#!/usr/bin/env python3
"""Frozen English submission inference, with explicit historical-run access."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def historical_main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--text', required=True)
    parser.add_argument('--run-id', default='english-finetune-pilot-20261104')
    parser.add_argument('--speaker', default='en_est_0155')
    parser.add_argument('--pacing', choices=['punctuation', 'legacy', 'learned'])
    parser.add_argument('--duration-control', type=float)
    args = parser.parse_args()
    metadata=ROOT/'output/runs'/args.run_id/'english/best.json'
    learned=metadata.exists() and json.loads(metadata.read_text()).get('model_type')=='learned_pause_v1'
    if args.pacing=='learned' or learned:
        if args.pacing not in (None,'learned') or args.duration_control not in (None,1.):
            parser.error('Learned-pause checkpoints use model predictions with no manual pacing control')
        from setup.synthesize_learned_pause import main as synthesize_learned
        return synthesize_learned(['--text',args.text,'--run-id',args.run_id,'--speaker',args.speaker])
    from setup.generate_reviewed_english_samples import main as synthesize
    argv = ['--text', args.text, '--run-id', args.run_id, '--speaker', args.speaker, '--pacing', args.pacing or 'punctuation']
    if args.duration_control is not None:
        argv += ['--duration-control', str(args.duration_control)]
    synthesize(argv)


def main():
    if any(arg=='--run-id' or arg.startswith('--run-id=') for arg in sys.argv[1:]):
        return historical_main()
    from setup.final_inference import main as frozen_inference
    return frozen_inference('english')


if __name__ == '__main__':main()
