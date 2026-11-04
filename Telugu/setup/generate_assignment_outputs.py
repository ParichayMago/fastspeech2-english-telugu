#!/usr/bin/env python3
"""Generate exactly five source-preserving test samples per language and 30 plots."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'Telugu'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--language',choices=['english','telugu','both'],default='both')
    args=p.parse_args()
    from setup.final_inference import InferenceEngine
    manifest=json.loads((ROOT/'Telugu/setup/assignment_test_sentences.json').read_text())
    if (ROOT/manifest['source']).exists():
        assert hashlib.sha256((ROOT/manifest['source']).read_bytes()).hexdigest()==manifest['source_sha256']
    report={'source_manifest':'Telugu/setup/assignment_test_sentences.json','source_pdf_sha256':manifest['source_sha256'],
            'source_transcription_note':manifest['transcription_note'],'device':'cpu',
            'additional_cloud_compute_cost_usd':0,'samples':[]}
    target=ROOT/'English/artifacts/Assignment/generation.json';target.parent.mkdir(exist_ok=True)
    for language in ('english','telugu'):
        if args.language not in ('both',language):continue
        cases=[s for s in manifest['samples'] if s['language']==language]
        assert len(cases)==5
        engine=InferenceEngine(language)
        for sample in cases:
            info=engine.synthesize(sample['text'],sample['id'],required=True,plots=True)
            report['samples'].append(info)
            target.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
            print(json.dumps({'id':sample['id'],'seconds':info['seconds'],'audio':info['audio_path'],
                              'plots':len(info['plots']),'checkpoint_step':info['checkpoint_step']}),flush=True)
        del engine;gc.collect()
    report['generated_audio_count']=len(report['samples'])
    report['generated_variance_plot_count']=sum(len(s['plots']) for s in report['samples'])
    report['status']='generated_awaiting_final_verification'
    target.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':main()
