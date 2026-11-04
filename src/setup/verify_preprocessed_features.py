#!/usr/bin/env python3
"""Independent checks of saved targets, split membership and training statistics."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--language',choices=['english','telugu'],required=True)
    parser.add_argument('--feature-version')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root/'src'))
    language=args.language
    features=root/'data/features'
    if args.feature_version:features=features/args.feature_version
    features=features/language
    stats=json.loads((features/'stats.json').read_text())
    speakers=json.loads((features/'speakers.json').read_text())
    actual={}; source={}; distributions={}; metadata_texts={}
    norm_sum={name:0.0 for name in ['pitch','energy']}
    norm_square={name:0.0 for name in norm_sum}; norm_count=Counter()
    from text import text_to_sequence
    allowed={phone for line in (root/'data/preprocessing'/language/'lexicon.txt').read_text().splitlines()
             for phone in line.split()[1:]} | {'sil','sp','spn'}
    for split,filename in [('train','train.txt'),('validation','val.txt')]:
        selected=[json.loads(line) for line in (root/'data/splits/v1'/language/(split+'-prepared.jsonl')).read_text().splitlines()]
        source[split]={r['sample_id']:r for r in selected}
        rows=[]; metadata_texts[split]=set()
        for line in (features/filename).read_text().splitlines():
            name,speaker,phones,text=line.split('|')
            metadata_texts[split].add(text)
            if name not in source[split]:raise ValueError('Feature row moved to a different split')
            original=source[split][name]
            assert original['speaker_id']==speaker
            if language=='english':
                assert original['speaker_id_estimated'] and original['source_speaker_id'] is None
                assert speaker.startswith('en_est_')
            assert speaker in speakers
            arrays={kind:np.load(features/kind/f'{speaker}-{kind}-{name}.npy')
                    for kind in ['duration','mel','pitch','energy']}
            tokens=text_to_sequence(phones,[])
            assert set(phones.strip('{}').split())<=allowed,'Alignment phone outside language lexicon'
            assert len(tokens)==len(phones.strip('{}').split())==len(arrays['duration'])
            assert len(tokens)==len(arrays['pitch'])==len(arrays['energy'])
            assert arrays['mel'].ndim==2 and arrays['mel'].shape[1]==80
            assert int(arrays['duration'].sum())==len(arrays['mel'])
            if args.feature_version:
                # Independent timeline check: full non-silence span, including
                # every internal gap, must survive to the end of the saved mel.
                import tgt
                tier=tgt.io.read_textgrid(str(features/'TextGrid'/speaker/(name+'.TextGrid')),
                                          include_empty_intervals=True).get_tier_by_name('phones')
                speech=[i for i in tier._objects if i.text.strip() not in {'','sil','sp','spn'}]
                expected_frames=int(np.round((speech[-1].end_time-speech[0].start_time)*22050/256))
                assert len(arrays['mel'])==expected_frames,'Acoustic timeline was truncated'
                expected_speech=[i.text.strip() for i in speech]
                metadata_speech=[p for p in phones.strip('{}').split() if p not in {'sil','sp','spn'}]
                assert expected_speech==metadata_speech,'Aligned speech phones changed'
                # Each gap must have a silence token covering its exact frame range.
                tokens_list=phones.strip('{}').split()
                cumulative=np.r_[0,np.cumsum(arrays['duration'])]
                spans=[(cumulative[i],cumulative[i+1]) for i,p in enumerate(tokens_list)
                       if p in {'sil','sp','spn'}]
                start=speech[0].start_time
                intervals=[i for i in tier._objects if i.end_time>start and i.start_time<speech[-1].end_time]
                for left,right in zip(intervals,intervals[1:]):
                    if right.start_time>left.end_time+1e-7:
                        a=round((left.end_time-start)*22050/256);b=round((right.start_time-start)*22050/256)
                        assert any(x<=a and y>=b for x,y in spans),'Implicit gap missing silence token'
                for interval in intervals:
                    if not interval.text.strip():
                        a=round((interval.start_time-start)*22050/256);b=round((interval.end_time-start)*22050/256)
                        assert any(x<=a and y>=b for x,y in spans),'Blank interval missing silence token'
            assert all(np.isfinite(v).all() for v in arrays.values())
            if split=='train':
                for kind in norm_sum:
                    values=arrays[kind].astype(np.float64)
                    q25,q75=np.percentile(values,[25,75]);gap=q75-q25
                    values=values[(values>=q25-1.5*gap)&(values<=q75+1.5*gap)]
                    norm_sum[kind]+=float(values.sum());norm_square[kind]+=float((values**2).sum())
                    norm_count[kind]+=len(values)
            rows.append({'sample_id':name,'speaker_id':speaker,'frames':len(arrays['mel'])})
        actual[split]=rows
        distributions[split]=dict(Counter(r['speaker_id'] for r in rows))
    train_ids={r['sample_id'] for r in actual['train']};val_ids={r['sample_id'] for r in actual['validation']}
    assert not train_ids&val_ids
    assert not metadata_texts['train']&metadata_texts['validation']
    assert not ({source['train'][i]['audio_sha256'] for i in train_ids}
                & {source['validation'][i]['audio_sha256'] for i in val_ids})
    if language=='english':
        assert not ({source['train'][i]['audio_path'].rsplit('_chunk_',1)[0] for i in train_ids}
                    & {source['validation'][i]['audio_path'].rsplit('_chunk_',1)[0] for i in val_ids})
    assert set(distributions['validation'])<=set(distributions['train'])
    normalization={}
    for kind in norm_sum:
        n=norm_count[kind];mean=norm_sum[kind]/n;variance=norm_square[kind]/n-mean**2
        assert abs(mean)<.001 and abs(variance-1)<.005,(kind,mean,variance)
        normalization[kind]={'filtered_training_normalized_mean':mean,
            'filtered_training_normalized_variance':variance,'training_phone_values':n}
    report={'status':'passed','language':language,'array_checks':'finite, matched phone lengths, 80-bin mels, duration sum equals mel frames, no silently dropped phones',
        'feature_rows':{s:len(rows) for s,rows in actual.items()},
        'source_selection_rows':{s:len(rows) for s,rows in source.items()},
        'feature_hours':{s:sum(r['frames'] for r in rows)*256/22050/3600 for s,rows in actual.items()},
        'no_train_validation_sample_overlap':True,'all_validation_speakers_seen_in_training':True,
        'no_normalized_transcript_overlap':True,'no_exact_audio_overlap':True,
        'no_english_recording_chunk_overlap':True if language=='english' else None,
        'english_speaker_ids_estimated':language=='english','normalization':normalization,
        'stats_sha256':hashlib.sha256((features/'stats.json').read_bytes()).hexdigest(),
        'train_metadata_sha256':hashlib.sha256((features/'train.txt').read_bytes()).hexdigest(),
        'validation_metadata_sha256':hashlib.sha256((features/'val.txt').read_bytes()).hexdigest()}
    report['feature_version']=args.feature_version or 'legacy'
    if args.feature_version:
        report['full_cropped_alignment_timeline_verified']=True
        report['all_internal_gaps_covered_by_silence_tokens']=True
        report['all_speech_phone_identities_preserved']=True
    out=root/'src/artifacts/Preprocessing'
    if args.feature_version:out=out/args.feature_version
    out.mkdir(parents=True,exist_ok=True)
    (out/(language+'-feature-verification.json')).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
