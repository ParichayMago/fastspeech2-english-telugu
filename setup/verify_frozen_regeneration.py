#!/usr/bin/env python3
"""Regenerate all frozen tests locally and require exact WAV/array equality."""
import gc
import hashlib
import json
from pathlib import Path
import tempfile
import sys
import numpy as np

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from setup.final_inference import InferenceEngine


def main():
    samples=json.loads((ROOT/'setup/assignment_test_sentences.json').read_text())['samples']
    checks=[]
    with tempfile.TemporaryDirectory(prefix='.regeneration-',dir=ROOT/'Results') as temporary:
        for language in ('english','telugu'):
            engine=InferenceEngine(language)
            for case in [s for s in samples if s['language']==language]:
                directory=ROOT/'Results'/language.title()/'test_generated_audios'
                original=json.loads((directory/(case['id']+'.json')).read_text())
                actual=engine.synthesize(case['text'],case['id'],required=True,plots=False,
                                         output_dir=Path(temporary)/language)
                if actual['audio_sha256']!=original['audio_sha256']:
                    raise ValueError('Frozen WAV changed: '+case['id'])
                with np.load(ROOT/original['predictions_path']) as before,np.load(ROOT/actual['predictions_path']) as after:
                    if set(before.files)!=set(after.files):raise ValueError('Prediction keys differ')
                    arrays={key:bool(np.array_equal(before[key],after[key])) for key in before.files}
                    if not all(arrays.values()):raise ValueError('Prediction arrays changed: '+case['id'])
                checks.append(dict(id=case['id'],same_audio_sha256=True,same_arrays=arrays))
            del engine;gc.collect()
    result={'samples':checks,'same_audio_count':len(checks),'all_arrays_identical':True,
            'device':'cpu','trained_weights_changed':False,'cloud_resources_used':False,
            'software_scope':'Numerical regression, not a new listening assessment'}
    (ROOT/'Results/Assignment/frozen-regeneration-verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
