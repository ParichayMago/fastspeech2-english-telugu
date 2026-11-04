#!/usr/bin/env python3
"""Extract upstream FastSpeech2 targets using fixed splits and training-only stats."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys

for _thread_env in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[_thread_env] = '1'

import numpy as np
import yaml

PROCESSOR = None


def initialize(config, root):
    global PROCESSOR
    os.chdir(root); sys.path.insert(0,str(Path(root)/'src'))
    import torch
    torch.set_num_threads(1)
    from preprocessor.preprocessor import Preprocessor
    PROCESSOR = Preprocessor(config)


def extract(task):
    split, row, in_dir = task
    PROCESSOR.in_dir = in_dir
    name, speaker = row['sample_id'], row['speaker_id']
    try:
        grid = Path(PROCESSOR.out_dir)/'TextGrid'/speaker/(name+'.TextGrid')
        if not grid.exists():
            return {'sample_id':name,'split':split,'status':'missing_alignment'}
        returned = PROCESSOR.process_utterance(speaker,name)
        if returned is None:
            return {'sample_id':name,'split':split,'status':'unusable_pitch_or_alignment'}
        info, _, _, frames = returned
        from text import text_to_sequence
        phone_text = info.split('|')[2]
        expected = phone_text.strip('{}').split()
        allowed=set(PROCESSOR.config['assignment_allowed_phones']) | {'sil','sp','spn'}
        if not set(expected)<=allowed:
            raise ValueError('Unexpected aligner phone labels: '+repr(sorted(set(expected)-allowed)))
        encoded = text_to_sequence(phone_text, [])
        if len(encoded) != len(expected):
            raise ValueError('Unknown phone was dropped from symbol vocabulary')
        out=Path(PROCESSOR.out_dir)
        duration=np.load(out/'duration'/f'{speaker}-duration-{name}.npy')
        mel=np.load(out/'mel'/f'{speaker}-mel-{name}.npy')
        pitch=np.load(out/'pitch'/f'{speaker}-pitch-{name}.npy')
        energy=np.load(out/'energy'/f'{speaker}-energy-{name}.npy')
        if not (mel.ndim==2 and mel.shape[1]==80 and len(duration)==len(expected)
                and int(duration.sum())==len(mel)==frames
                and len(pitch)==len(expected)==len(energy)):
            raise ValueError('Feature length mismatch')
        if not all(np.isfinite(x).all() for x in [mel,pitch,energy,duration]):
            raise ValueError('Nonfinite features')
        return {'sample_id':name,'speaker_id':speaker,'split':split,'status':'passed',
                'frames':frames,'metadata':info, 'phones':len(expected),
                'alignment':PROCESSOR.last_alignment_info}
    except Exception as e:
        return {'sample_id':name,'split':split,'status':'error',
                'reason':type(e).__name__+': '+str(e)[:250]}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--language',choices=['english','telugu'],required=True)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--limit',type=int)
    parser.add_argument('--feature-version',default='gap-fixed-v2')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[2]; os.chdir(root);sys.path.insert(0,str(root/'src'))
    language=args.language
    corpus=root/'data/preprocessing'/language
    if not args.feature_version.replace('-','').isalnum():
        raise ValueError('Feature version must be a simple directory name')
    output=root/'data/features'/args.feature_version/(language+'-smoke' if args.limit else language)
    if (output/'stats.json').exists():
        raise ValueError('Completed version already exists; choose a new feature version')
    output.mkdir(parents=True,exist_ok=True)
    for name in ['mel','duration','pitch','energy','TextGrid']:(output/name).mkdir(exist_ok=True)
    config=yaml.safe_load((root/'src/config/LJSpeech/preprocess.yaml').read_text())
    config['dataset']='SvarahEnglish' if language=='english' else 'IndicTTSTelugu'
    config['path']['raw_path']=str(corpus/'corpus/train')
    config['path']['preprocessed_path']=str(output)
    config['preprocessing']['pitch']['feature']='phoneme_level'
    config['preprocessing']['energy']['feature']='phoneme_level'
    config['preprocessing']['text']['text_cleaners']=[]
    config['assignment_allowed_phones']=sorted({phone for line in (corpus/'lexicon.txt').read_text().splitlines()
                                               for phone in line.split()[1:]})
    tasks=[]; members={}
    for split in ['train','validation']:
        rows=[json.loads(line) for line in (corpus/(split+'-normalized.jsonl')).read_text().splitlines()]
        if args.limit:
            if split=='validation':
                smoke_speakers={r['speaker_id'] for r in members.get('train',[])}
                rows=[r for r in rows if r['speaker_id'] in smoke_speakers]
            rows=rows[:args.limit]
        members[split]=rows
        for row in rows:
            speaker,name=row['speaker_id'],row['sample_id']
            grid=corpus/'alignments'/split/speaker/(name+'.TextGrid')
            if not grid.exists():
                # MFA can omit redundant single-speaker directories.
                alternative=corpus/'alignments'/split/(name+'.TextGrid')
                if alternative.exists():grid=alternative
            target=output/'TextGrid'/speaker/(name+'.TextGrid')
            if grid.exists():
                target.parent.mkdir(parents=True,exist_ok=True)
                if not target.exists():target.symlink_to(grid)
            tasks.append((split,row,str(corpus/'corpus'/split)))
    with ProcessPoolExecutor(max_workers=args.workers,
            mp_context=multiprocessing.get_context('spawn'),initializer=initialize,
            initargs=(config,str(root))) as pool:
        results=[]
        for result in pool.map(extract,tasks,chunksize=4):
            results.append(result)
            if len(results)%100==0:
                print(json.dumps({'processed':len(results),'total':len(tasks),
                    'passed':sum(r['status']=='passed' for r in results)}),flush=True)
    # All pitch/energy files are unnormalized at this point. Fit on training only.
    (output/'feature-audit.jsonl').write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in results))
    from sklearn.preprocessing import StandardScaler
    scalers={name:StandardScaler() for name in ['pitch','energy']}
    successes=[r for r in results if r['status']=='passed']
    if not any(r['split']=='train' for r in successes):
        raise ValueError('No valid training features; do not fabricate statistics')
    train_speakers={r['speaker_id'] for r in successes if r['split']=='train'}
    for row in successes:
        if row['split']=='validation' and row['speaker_id'] not in train_speakers:
            row['status']='no_training_speaker_after_alignment_filter'
    successes=[r for r in successes if r['status']=='passed']
    if not any(r['split']=='validation' for r in successes):
        raise ValueError('No valid validation features; smoke checks must cover both splits')
    for row in successes:
        if row['split']!='train':continue
        for name,scaler in scalers.items():
            values=np.load(output/name/f"{row['speaker_id']}-{name}-{row['sample_id']}.npy")
            # Match upstream outlier filtering for fitting, preserving full targets.
            q25,q75=np.percentile(values,[25,75]); gap=q75-q25
            fitted=values[(values>=q25-1.5*gap)&(values<=q75+1.5*gap)]
            if len(fitted):scaler.partial_fit(fitted.reshape(-1,1))
    stats={}
    for name,scaler in scalers.items():
        low,high=float('inf'),float('-inf')
        for row in successes:
            path=output/name/f"{row['speaker_id']}-{name}-{row['sample_id']}.npy"
            values=np.load(path)
            values=((values-scaler.mean_[0])/scaler.scale_[0]).astype(np.float32)
            if not np.isfinite(values).all():raise ValueError('Nonfinite normalized feature')
            np.save(path,values)
            if row['split']=='train':low=min(low,float(values.min()));high=max(high,float(values.max()))
        stats[name]=[low,high,float(scaler.mean_[0]),float(scaler.scale_[0])]
    (output/'stats.json').write_text(json.dumps(stats,indent=2)+'\n')
    speakers={speaker:i for i,speaker in enumerate(sorted({r['speaker_id'] for r in successes}))}
    (output/'speakers.json').write_text(json.dumps(speakers,indent=2)+'\n')
    for split,filename in [('train','train.txt'),('validation','val.txt')]:
        (output/filename).write_text(''.join(r['metadata']+'\n' for r in successes if r['split']==split))
    (output/'feature-audit.jsonl').write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in results))
    # Portable configs keep relative dataset paths for local and cloud checkouts.
    config['path']['raw_path']='data/preprocessing/'+language+'/corpus/train'
    config['path']['preprocessed_path']=str(output.relative_to(root))
    config['feature_version']=args.feature_version
    config['estimated_acoustic_speaker_labels']=language=='english'
    directory=root/'data/rebuild-config'/args.feature_version/('English' if language=='english' else 'Telugu')
    if args.limit:directory=directory/'smoke'
    directory.mkdir(parents=True,exist_ok=True)
    (directory/'preprocess.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    model=yaml.safe_load((root/'src/config/LJSpeech/model.yaml').read_text())
    model['multi_speaker']=language=='english'
    # Restored complete utterances can exceed the upstream 1000-frame cap.
    model['max_seq_len']=max(model['max_seq_len'],2048)
    assert max(r['frames'] for r in successes)<=model['max_seq_len']
    (directory/'model.yaml').write_text(yaml.safe_dump(model,sort_keys=False))
    training=yaml.safe_load((root/'src/config/LJSpeech/train.yaml').read_text())
    for k in ['ckpt_path','log_path','result_path']:
        training['path'][k]='output/'+language+'/'+k.removesuffix('_path')
    (directory/'train.yaml').write_text(yaml.safe_dump(training,sort_keys=False))
    report={'language':language,'scope':'small_subset_check' if args.limit else 'full_selected_splits',
        'status_counts':dict(Counter(r['status'] for r in results)),
        'requested_rows':len(results),'successful_rows':len(successes),
        'normalization_fit':'Training features only. Validation uses frozen training mean/std.',
        'normalization_training_rows':sum(r['split']=='train' for r in successes),
        'normalization_feature_samples':{name:int(scaler.n_samples_seen_) for name,scaler in scalers.items()},
        'split_rows':{s:sum(r['split']==s for r in successes) for s in members},
        'feature_hours':{s:sum(r['frames'] for r in successes if r['split']==s)*256/22050/3600 for s in members},
        'feature_version':args.feature_version,
        'feature_path':str(output.relative_to(root)),
        'rows_with_preserved_gaps':sum(r['alignment']['preserved_gap_seconds']>1e-7 for r in successes),
        'preserved_internal_gap_hours':sum(r['alignment']['preserved_gap_seconds'] for r in successes)/3600,
        'speaker_count':len(speakers),'english_speaker_labels_estimated':language=='english',
        'mel_bins':80,'sample_rate':22050,'hop_length':256,
        'failures':[{k:v for k,v in r.items() if k!='metadata'} for r in results if r['status']!='passed']}
    dest=root/'Results/Preprocessing'/args.feature_version;dest.mkdir(parents=True,exist_ok=True)
    suffix='-smoke' if args.limit else ''
    (dest/(language+'-features'+suffix+'.json')).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='failures'},indent=2),flush=True)


if __name__=='__main__':main()
