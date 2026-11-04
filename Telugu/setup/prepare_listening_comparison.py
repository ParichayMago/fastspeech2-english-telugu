#!/usr/bin/env python3
"""Generate matched, untreated English parent/learned-pause listening pairs on CPU."""
import copy
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'Telugu'))
OUT=ROOT/'Results/Listening/english-parent-vs-learned-pauses'
PARENT=dict(path='output/runs/english-finetune-pilot-20261104/english/best.pth.tar',
            sha256='b4955131de0b8915ede03eb2232fe4da9702136c1086d67bb31f5c0943f24aad',
            step=2500,model_type='fastspeech2',source_run='english-finetune-pilot-20261104')
MIT="i've reproduced the results from the MIT blog post today on weight-space programming. but one additional result i have to add: this is a new, step-change emerging capability that's appeared in just the latest set of models."


def label_orders(ids,seed=20261104):
    """Persist a reproducible, balanced label order independent of judgments."""
    rng=random.Random(seed);orders=[('parent','learned'),('learned','parent')]*(len(ids)//2)
    if len(ids)%2:orders.append(('parent','learned'))
    rng.shuffle(orders)
    return {key:dict(zip(('A','B'),order)) for key,order in zip(ids,orders)}


def main():
    import gc
    import torch
    import yaml
    from setup.final_inference import InferenceEngine,digest,test_cases
    config=yaml.safe_load((ROOT/'English/config.yaml').read_text())
    parent_path=ROOT/PARENT['path']
    if not parent_path.is_file():raise FileNotFoundError('Retained parent checkpoint is required: '+str(parent_path))
    if digest(parent_path)!=PARENT['sha256']:raise ValueError('Parent checkpoint identity mismatch')
    parent_checkpoint=torch.load(parent_path,map_location='cpu',weights_only=False)
    if parent_checkpoint['step']!=PARENT['step'] or parent_checkpoint['language']!='english':
        raise ValueError('Unexpected parent checkpoint metadata')
    # Same normalization and speaker-index lookup must be used by both systems.
    parent_features=ROOT/parent_checkpoint['configs'][0]['path']['preprocessed_path']
    for name in ('stats.json','speakers.json'):
        if digest(parent_features/name)!=digest(ROOT/config['resources']/name):
            raise ValueError('Parent and learned inference resources differ: '+name)
    del parent_checkpoint
    cases=[s for s in test_cases() if s['language']=='english']+[dict(id='MIT',text=MIT)]
    orders=label_orders([s['id'] for s in cases]);outputs={}
    frozen_before={p: digest(p) for p in (ROOT/'Results/English/test_generated_audios').glob('*') if p.is_file()}
    for system in ('parent','learned'):
        selected=copy.deepcopy(config)
        if system=='parent':selected['checkpoint']={**PARENT,'bytes':parent_path.stat().st_size}
        engine=InferenceEngine('english',config=selected,parent_english=system=='parent')
        outputs[system]={}
        for case in cases:
            info=engine.synthesize(case['text'],sample_id=case['id'],output_dir=str((OUT/'source'/system).relative_to(ROOT)),plots=False)
            if system=='learned' and case['id']!='MIT':
                original=json.loads((ROOT/'English/artifacts/test_generated_audios'/(case['id']+'.json')).read_text())
                if info['audio_sha256']!=original['audio_sha256']:
                    raise ValueError('Learned comparison differs from released WAV: '+case['id'])
            outputs[system][case['id']]=info
            print(json.dumps(dict(system=system,id=case['id'],seconds=info['seconds'],sha256=info['audio_sha256'])),flush=True)
        del engine;gc.collect()
    pairs=[]
    for case in cases:
        key=case['id'];a=outputs['parent'][key];b=outputs['learned'][key]
        if a['normalized_words']!=b['normalized_words'] or a['pronunciations']!=b['pronunciations']:
            raise ValueError('Lexical pronunciations differ between systems: '+key)
        if a['phones']!=[p for p,k in zip(b['phones'],b['boundary_types']) if not k]:
            raise ValueError('Common lexical phone sequence changed: '+key)
        clips={}
        for variant,system in orders[key].items():
            info=outputs[system][key];target=OUT/'pairs'/key/(variant+'.wav');target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(ROOT/info['audio_path'],target)
            clips[variant]=dict(audio_path=str(target.relative_to(ROOT)),audio_sha256=digest(target),seconds=info['seconds'])
        pairs.append(dict(id=key,text=case['text'],clips=clips))
    if any(digest(p)!=sha for p,sha in frozen_before.items()):raise ValueError('Frozen submission output changed')
    manifest=dict(version=1,comparison='English parent versus learned-pause system',speaker=config['speaker'],
        pairs=pairs,system_key=orders,checkpoints=dict(parent=PARENT,learned=config['checkpoint']),
        controls=dict(same_text=True,same_lexical_pronunciations=True,same_speaker=True,
            same_normalization_statistics=True,same_vocoder=True,vocoder_sha256=config['vocoder']['sha256'],
            sampling_rate=config['sampling_rate'],duration=1.,pitch=1.,energy=1.,
            denoising=False,external_silence_samples=0,waveform_edits=[],sentence_splitting=False),
        interpretation='System comparison: common lexical phones; parent omits text-derived boundary slots, learned model predicts their presence/duration. Not an isolated pause-head ablation.',
        order='Balanced A/B mapping fixed before judgments; browser displays labels only. Mapping is available here for later audit, so this is not a secured clinical blind test.',
        assessment='No listening judgments collected or quality winner established',cloud_resources_started=False,additional_cloud_compute_cost_usd=0,
        released_english_wavs_matched=5,released_outputs_unchanged=True)
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/'comparison.json';path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print('Comparison ready: '+str(path),flush=True)


if __name__=='__main__':main()
