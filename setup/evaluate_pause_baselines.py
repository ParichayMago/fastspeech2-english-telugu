#!/usr/bin/env python3
"""Occurrence baselines on held-out internal boundaries; no audio modification."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent.parent


def metrics(tp,fp,fn):
    return dict(tp=tp,fp=fp,fn=fn,precision=tp/max(1,tp+fp),recall=tp/max(1,tp+fn),
                f1=2*tp/max(1,2*tp+fp+fn))


def main():
    features=ROOT/'data/features/learned-pauses-v1/english'
    count={key:[0,0,0] for key in ('every_internal_boundary','punctuation_only')}
    digest=hashlib.sha256();rows=0
    for line in (features/'val.txt').read_text().splitlines():
        sid,speaker,*_=line.split('|');rows+=1
        boundary=np.load(features/'boundary'/f'{speaker}-boundary-{sid}.npy')
        duration=np.load(features/'duration'/f'{speaker}-duration-{sid}.npy')
        digest.update(sid.encode());digest.update(boundary.tobytes());digest.update(duration.tobytes())
        valid=boundary>0;valid[-1]=False;actual=duration>0
        for name,predicted in [('every_internal_boundary',valid),('punctuation_only',boundary>=2)]:
            values=[int((predicted&actual&valid).sum()),int((predicted&~actual&valid).sum()),int((~predicted&actual&valid).sum())]
            count[name]=[a+b for a,b in zip(count[name],values)]
    analysis=json.loads((ROOT/'Results/Assignment/analysis/measured-analysis.json').read_text())
    selected=analysis['training_runs']['english-learned-pauses-pilot-20261104/english']['best']
    p=selected['pause_metrics']
    result={'validation_utterances':rows,'boundary_targets_sha256':digest.hexdigest(),
            'mask':'All text-derived internal word boundaries; terminal slots excluded',
            'baselines':{key:metrics(*value) for key,value in count.items()},
            'selected_model':{'step':selected['step'],**metrics(p['tp'],p['fp'],p['fn'])},
            'model_measurement_source':'Saved complete validation, not a new GPU evaluation',
            'interpretation':'Occurrence classifiers only; no pause lengths, waveform edits or auditory quality claims'}
    result['F1_gain_over_every_boundary_percentage_points']=100*(result['selected_model']['f1']-result['baselines']['every_internal_boundary']['f1'])
    path=ROOT/'Results/Assignment/analysis/pause-baselines.json'
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
