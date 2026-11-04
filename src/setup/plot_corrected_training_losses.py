#!/usr/bin/env python3
"""Plot an immutable local metrics snapshot of a corrected training run."""
import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from plot_training_losses import ROOT, NAMES, plot_phase, plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', default='gap-fixed-v2-20261103')
    args = parser.parse_args()
    if args.run_id != Path(args.run_id).name:
        parser.error('run-id must be a directory name')
    out = ROOT / 'Results/Training' / args.run_id / 'plots'
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'figure.facecolor': 'white', 'savefig.facecolor': 'white',
                         'svg.fonttype': 'none'})
    summary = {'run_id': args.run_id,
               'created_utc': datetime.now(timezone.utc).isoformat(),
               'source': 'Local backup snapshots; active training may be ahead',
               'training': 'Raw logged minibatches and trailing mean of 15 logged points',
               'validation': 'Complete held-out evaluations only, unsmoothed',
               'languages': {}}
    datasets = {}
    for lang in ('english', 'telugu'):
        source = ROOT / 'output/runs' / args.run_id / lang / 'metrics.jsonl'
        raw = source.read_bytes()
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
        assert all(r['language'] == lang and len(r['losses']) == 6 and
                   all(math.isfinite(v) for v in r['losses']) for r in rows)
        datasets[lang] = rows
        directory = out / lang
        directory.mkdir(exist_ok=True)
        (directory / 'metrics-snapshot.jsonl').write_bytes(raw)
        with (directory / 'loss_history.csv').open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['utc', 'phase', 'step', *NAMES, 'examples', 'complete'])
            writer.writerows([r['utc'], r['phase'], r['step'], *r['losses'],
                              r.get('examples', ''), r.get('complete', '')] for r in rows)
        train = [r for r in rows if r['phase'] == 'train']
        val = [r for r in rows if r['phase'] == 'validation' and r.get('complete') is True]
        best = min(val, key=lambda r: r['losses'][0])
        summary['languages'][lang] = {
            'source': str(source), 'metrics_sha256': hashlib.sha256(raw).hexdigest(),
            'latest_logged_training_step': train[-1]['step'],
            'latest_logged_utc': rows[-1]['utc'],
            'latest_validation_step': val[-1]['step'],
            'latest_validation_loss': val[-1]['losses'][0],
            'best_validation_step': best['step'], 'best_validation_loss': best['losses'][0]}
        for phase in ('train', 'validation'):
            fig, ax = plt.subplots(figsize=(8, 4.8))
            draw(ax, rows, lang, phase)
            fig.suptitle('Corrected features | ' + args.run_id, fontsize=10)
            fig.tight_layout(rect=(0, .03, 1, .95))
            save(fig, directory / ('training_loss' if phase == 'train' else 'validation_loss'))
    steps = ' / '.join(f"{lang.title()}: {summary['languages'][lang]['latest_logged_training_step']:,}" for lang in datasets)
    for vertical in (False, True):
        fig, axes = plt.subplots(4, 1, figsize=(8, 18)) if vertical else plt.subplots(2, 2, figsize=(13, 8.8))
        for ax, (lang, phase) in zip(axes.flat, ((l, p) for l in datasets for p in ('train', 'validation'))):
            draw(ax, datasets[lang], lang, phase)
        fig.suptitle('FastSpeech 2 | Corrected training run', fontsize=18, fontweight='bold', y=.985)
        fig.text(.08, .94 if not vertical else .955, f'{args.run_id}\nLocal log snapshot | {steps} logged steps', fontsize=10, color='#425466')
        fig.text(.08, .012, 'Training: raw minibatches + 15-point trailing mean. Validation: complete held-out splits.\n'
                 'Active run may be ahead. English speaker labels are estimated. Loss does not establish speech quality.', fontsize=8, color='#425466')
        fig.tight_layout(rect=(.02, .055, .99, .90 if not vertical else .935), h_pad=3)
        save(fig, out / ('loss_overview_vertical' if vertical else 'loss_overview'))
    (out / 'plot-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


def draw(ax, rows, language, phase):
    plot_phase(ax, rows, language, phase)
    # A running snapshot has a latest observation, not a final training result.
    for annotation in ax.texts:
        annotation.set_text(annotation.get_text().replace('Final:', 'Latest:'))


def save(fig, stem):
    fig.savefig(stem.with_suffix('.png'), dpi=180)
    fig.savefig(stem.with_suffix('.svg'))
    plt.close(fig)


if __name__ == '__main__':
    main()
