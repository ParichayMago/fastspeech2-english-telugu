#!/usr/bin/env python3
"""Check the assignment's required tree and a clean exported project root."""
import argparse
import json
import re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
ALLOWED={'English','Telugu','Results','Report','src','README.md','requirements.txt','LICENSE','.gitignore','.gitattributes'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--strict',action='store_true',help='Reject extra top-level entries in an extracted package')
    args=parser.parse_args();root=args.root.resolve()
    required=['Report/Technical_Report.pdf','README.md','requirements.txt',
              'src/model/fastspeech2.py','src/setup/train_assignment.py','src/audio_review/server.py']
    for language in ('English','Telugu'):
        required += [language+'/'+p for p in ('train.py','preprocess.py','inference.py','config.yaml','checkpoints/checkpoint.json')]
        required += ['Results/'+language+'/'+p for p in ('training_loss.png','validation_loss.png')]
        audios=list((root/'Results'/language/'test_generated_audios').glob('*.wav'))
        assert len(audios)==5,(language,'expected five test audios')
        for kind in ('duration','pitch','energy'):
            figures=list((root/'Results'/language/'variance_adaptor_plots'/kind).glob('*.png'))
            assert len(figures)==5,(language,kind,'expected five plots')
    missing=[p for p in required if not (root/p).is_file()]
    if missing:raise ValueError('Missing required files: '+repr(missing))
    results_files=[p.relative_to(root).as_posix() for p in (root/'Results').rglob('*') if p.is_file()]
    pattern=r'Results/(English|Telugu)/(?:(?:training|validation)_loss\.png|test_generated_audios/[^/]+\.wav|variance_adaptor_plots/(?:duration|pitch|energy)/[^/]+\.png)'
    extras=[p for p in results_files if not re.fullmatch(pattern,p)]
    extra_dirs={p.name for p in (root/'Results').iterdir()}-{'English','Telugu'}
    if extras or extra_dirs:raise ValueError('Unexpected Results entries: '+repr(extras+sorted(extra_dirs)))
    if len(results_files)!=44:raise ValueError('Results must contain the 44 required files')
    if args.strict:
        extras={p.name for p in root.iterdir()}-ALLOWED
        if extras:raise ValueError('Unexpected top-level entries: '+repr(extras))
        if (root/'src/setup/runpod-assignment.json').exists():
            raise ValueError('Local Runpod state must not be in a submission export')
    print(json.dumps(dict(status='passed',repository_root_is_project=True,
        shared_implementation='src/',required_audios=10,required_variance_plots=30,
        required_results_files=44,strict_export_check=args.strict)))


if __name__=='__main__':main()
