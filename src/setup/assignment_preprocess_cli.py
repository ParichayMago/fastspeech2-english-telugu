"""Shared command-line entrypoint for the two assignment preprocessing wrappers."""
import argparse
from pathlib import Path
import subprocess
import sys


def main(language):
    parser=argparse.ArgumentParser(description=f'{language} FastSpeech 2 preprocessing')
    parser.add_argument('--stage',choices=['corpus','align','features','check','all'],default='all')
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--feature-version',default='gap-fixed-v2')
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[2]
    commands={
        'corpus':[sys.executable,'src/setup/prepare_assignment_corpus.py','--language',language],
        'align':['bash','src/setup/run_assignment_alignment.sh',language],
        'features':[sys.executable,'src/setup/extract_assignment_features.py','--language',language,
                    '--workers',str(args.workers),'--feature-version',args.feature_version],
        'check':[sys.executable,'src/setup/check_real_preprocessed_batch.py','--language',language,
                 '--device',args.device,'--config-root','data/rebuild-config/'+args.feature_version,
                 '--report-version',args.feature_version]}
    for stage in commands if args.stage=='all' else [args.stage]:
        subprocess.run(commands[stage],cwd=root,check=True)
