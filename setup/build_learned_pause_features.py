#!/usr/bin/env python3
"""Derive measured boundary targets locally; preserve all acoustic masters."""
import bisect
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
import numpy as np
import tgt
from preprocessor.alignment import alignment_to_phones
from setup.learned_pause_frontend import words_and_types, encode_words, BOUNDARY_TYPES

VERSION = 'learned-pauses-v1'


def remap_sample(old_phones, old_duration, old_pitch, old_energy, words, types, lexicon,
                 word_intervals, crop_start, mel_frames):
    expected, boundary_types = encode_words(words, types, lexicon)
    aligned_words = [x.text.casefold() for x in word_intervals]
    if aligned_words != words:
        raise ValueError('word_tier_transcript_mismatch')
    sil = {'sp', 'sil', 'spn'}
    lexical = [p for p in old_phones if p not in sil]
    reference = [p for w in words for p in lexicon[w]]
    if lexical != reference:
        raise ValueError('aligned_pronunciation_mismatch')
    ends = np.cumsum([len(lexicon[w]) for w in words]).tolist()
    at_end = {end: i for i, end in enumerate(ends)}
    speech, pauses, cursor = [], [[] for _ in words], 0
    for i, phone in enumerate(old_phones):
        if phone in sil:
            if cursor not in at_end:
                if old_duration[i] != 0:
                    raise ValueError('nonboundary_silence_requires_review')
            else:
                pauses[at_end[cursor]].append(i)
        else:
            speech.append(i); cursor += 1
    durations, pitches, energies, source_speech = [], [], [], []
    cursor = 0
    for wi, word in enumerate(words):
        for _ in lexicon[word]:
            oi = speech[cursor]; cursor += 1
            durations.append(int(old_duration[oi])); pitches.append(float(old_pitch[oi])); energies.append(float(old_energy[oi])); source_speech.append(oi)
        indexes = pauses[wi]
        frames = sum(int(old_duration[i]) for i in indexes)
        if wi < len(words) - 1:
            frame = lambda t: int(np.round((t-crop_start)*22050/256))
            measured = frame(word_intervals[wi+1].start_time)-frame(word_intervals[wi].end_time)
            if measured < 0 or measured != frames:
                raise ValueError('word_gap_frame_mismatch')
        elif frames:
            raise ValueError('unexpected_untrimmed_terminal_pause')
        durations.append(frames)
        if frames:
            pitches.append(sum(float(old_pitch[i])*int(old_duration[i]) for i in indexes)/frames)
            energies.append(sum(float(old_energy[i])*int(old_duration[i]) for i in indexes)/frames)
        else:
            # No frames exist for these slots. Their pitch/energy targets are
            # excluded from the loss; zero is only an unused normalized value.
            pitches.append(0.); energies.append(0.)
    result = [np.asarray(durations, dtype=np.int64), np.asarray(pitches, dtype=np.float32),
              np.asarray(energies, dtype=np.float32), np.asarray(boundary_types, dtype=np.int64)]
    d, p, e, b = result
    assert len(d) == len(expected) == len(p) == len(e) == len(b)
    assert int(d.sum()) == int(old_duration.sum()) == mel_frames
    assert np.array_equal(d[b == 0], old_duration[source_speech])
    assert np.array_equal(p[b == 0], old_pitch[source_speech])
    assert np.array_equal(e[b == 0], old_energy[source_speech])
    assert int(d[b > 0].sum()) == sum(int(old_duration[i]) for i, ph in enumerate(old_phones) if ph in sil)
    return expected, result


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    import argparse
    from setup.learned_pause_config import write_configs
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--configs-only',action='store_true',help='Reconstruct YAMLs from existing measured pause features')
    args=parser.parse_args()
    if args.configs_only:
        print(json.dumps({'config_directory':str(write_configs(ROOT).relative_to(ROOT))}))
        return
    old = ROOT/'data/features/gap-fixed-v2/english'
    out = ROOT/'data/features'/VERSION/'english'
    report_dir = ROOT/'Results/Preprocessing'/VERSION; report_dir.mkdir(parents=True, exist_ok=True)
    if (out/'stats.json').exists():
        raise ValueError('Completed features already exist; refuse overwrite')
    out.mkdir(parents=True, exist_ok=True)
    for kind in ('duration', 'pitch', 'energy', 'boundary', 'mel'):
        (out/kind).mkdir(exist_ok=True)
    lexicon = {p[0]: p[1:] for p in [l.split() for l in (ROOT/'data/preprocessing/english/lexicon.txt').read_text().splitlines()]}
    audits, split_info = [], {}
    for split, filename in [('train','train'), ('validation','val')]:
        sources = {r['sample_id']: r for r in [json.loads(l) for l in (ROOT/'data/splits/v1/english'/(split+'-prepared.jsonl')).read_text().splitlines()]}
        successful = []
        for line in (old/(filename+'.txt')).read_text().splitlines():
            sid, speaker, text, _ = line.split('|'); raw = sources[sid]['text']
            try:
                words, types = words_and_types(raw)
                grid = tgt.io.read_textgrid(str(old/'TextGrid'/speaker/(sid+'.TextGrid')), include_empty_intervals=True)
                wt = [i for i in grid.get_tier_by_name('words')._objects if i.text.strip()]
                phones, ds, start, end, _ = alignment_to_phones(grid.get_tier_by_name('phones'), 22050, 256)
                assert phones == text.strip('{}').split()
                arrays = {k: np.load(old/k/f'{speaker}-{k}-{sid}.npy') for k in ('duration','pitch','energy')}
                assert np.array_equal(arrays['duration'], ds)
                mel = old/'mel'/f'{speaker}-mel-{sid}.npy'
                frames = len(np.load(mel, mmap_mode='r'))
                new_phones, values = remap_sample(phones, arrays['duration'], arrays['pitch'], arrays['energy'],
                                                  words, types, lexicon, wt, start, frames)
                for kind, value in zip(('duration','pitch','energy','boundary'), values):
                    np.save(out/kind/f'{speaker}-{kind}-{sid}.npy', value)
                target = out/'mel'/mel.name
                if not target.exists():
                    shutil.copyfile(mel, target)
                assert sha(target) == sha(mel)
                raw_field = ' '.join(raw.replace('|',' ').split())
                successful.append('|'.join([sid, speaker, '{'+' '.join(new_phones)+'}', raw_field]))
                d, _, _, b = values; valid = b > 0; valid[-1] = False
                audits.append({'sample_id':sid,'split':split,'status':'passed','mel_frames':frames,
                    'boundary_slots':int(valid.sum()),'positive_pause_slots':int(((d>0)&valid).sum()),
                    'zero_pause_slots':int(((d==0)&valid).sum()),'pause_frames':int(d[valid].sum()),
                    'positive_pause_log_sum':float(np.log1p(d[(d>0)&valid]).sum()),
                    'max_pause_frames':int(d[valid].max()) if valid.any() else 0,
                    'boundary_type_counts':dict(Counter(map(int,b[b>0]))),
                    'source_mel_sha256':sha(mel),'source_text_sha256':hashlib.sha256(raw.encode()).hexdigest()})
            except (ValueError, AssertionError, KeyError, IndexError) as exc:
                audits.append({'sample_id':sid,'split':split,'status':'needs_review','reason':str(exc) or type(exc).__name__})
        (out/(filename+'.txt')).write_text('\n'.join(successful)+'\n')
        accepted = [r for r in audits if r['split']==split and r['status']=='passed']
        split_info[split] = {'accepted':len(accepted),'source_rows':len(sources),
            'feature_hours':sum(r['mel_frames'] for r in accepted)*256/22050/3600,
            'boundary_slots':sum(r['boundary_slots'] for r in accepted),
            'positive_pause_slots':sum(r['positive_pause_slots'] for r in accepted),
            'zero_pause_slots':sum(r['zero_pause_slots'] for r in accepted)}
        print(json.dumps({split:split_info[split]}),flush=True)
    (report_dir/'audit.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in audits))
    for name in ('stats.json','speakers.json'):
        shutil.copyfile(old/name,out/name); assert sha(old/name)==sha(out/name)
    train = [r for r in audits if r['split']=='train' and r['status']=='passed']
    positive = sum(r['positive_pause_slots'] for r in train); slots = sum(r['boundary_slots'] for r in train)
    schema = {'version':VERSION,'boundary_types':BOUNDARY_TYPES,'target_units':'integer mel frames measured from original word/phone alignments',
              'train_pause_prior':positive/slots,'train_pause_pos_weight':(slots-positive)/positive,
              'train_mean_positive_log_duration':sum(r['positive_pause_log_sum'] for r in train)/positive,
              'train_max_pause_frames':max(r['max_pause_frames'] for r in train),
              'terminal_boundary':'conditioning feature only; no trailing-silence target or pause loss',
              'fixed_pause_lengths':False,'statistics_fit':'training only; existing pitch/energy normalization frozen'}
    (out/'boundary-schema.json').write_text(json.dumps(schema,indent=2)+'\n')
    summary = {'feature_version':VERSION,'source_version':'gap-fixed-v2','splits':split_info,
               'status_counts':dict(Counter(r['status'] for r in audits)),
               'rejection_reasons':dict(Counter(r.get('reason') for r in audits if r['status']!='passed')),
               'checks':['word/phone/transcript identity','exact measured boundary frames','zero targets for absent pauses',
                         'full mel frame coverage','unchanged speech phone targets','byte-identical source mel arrays',
                         'frozen normalization and speaker maps'], 'boundary_schema':schema}
    (report_dir/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    summary['config_directory']=str(write_configs(ROOT).relative_to(ROOT))
    (report_dir/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == '__main__':
    main()
