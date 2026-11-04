#!/usr/bin/env python3
"""Plot actual saved minibatch and full-validation losses without cloud access.

Run locally with: uv run --with matplotlib python English/setup/plot_training_losses.py
"""
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, StrMethodFormatter

ROOT = Path(__file__).resolve().parents[2]
COLORS = {'english': '#147D92', 'telugu': '#AE4C24'}
NAMES = ['total', 'mel', 'mel_postnet', 'pitch', 'energy', 'duration']
WINDOW = 15


def trailing_mean(values):
    return [sum(values[max(0, i-WINDOW+1):i+1]) / min(i+1, WINDOW)
            for i in range(len(values))]


def style(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', color='#D9E1E8', linewidth=.7, alpha=.8)
    ax.set_axisbelow(True)
    ax.xaxis.set_major_locator(MaxNLocator(6, integer=True))
    ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    ax.set_xlabel('Optimizer step')
    ax.set_ylabel('Total loss')
    ax.set_xlim(left=0)
    ax.tick_params(colors='#425466')


def plot_phase(ax, rows, language, phase):
    selected = [r for r in rows if r['phase'] == phase and
                (phase == 'train' or r.get('complete') is True)]
    assert selected
    xs = [r['step'] for r in selected]
    ys = [r['losses'][0] for r in selected]
    assert xs == sorted(set(xs))
    color = COLORS[language]
    if phase == 'train':
        ax.plot(xs, ys, color=color, alpha=.28, linewidth=.85,
                label='Logged minibatch loss')
        ax.plot(xs, trailing_mean(ys), color=color, linewidth=1.8,
                label=f'Trailing mean ({WINDOW} logged points)')
        ax.set_ylim(bottom=0)
    else:
        ax.plot(xs, ys, color=color, linewidth=1.8, marker='o', markersize=4,
                label='Full held-out validation')
        best = min(selected, key=lambda r:r['losses'][0])
        ax.scatter([best['step']], [best['losses'][0]], color='#273B50',
                   marker='*', s=120, zorder=5,
                   label=f"Best observed: {best['losses'][0]:.3f} at {best['step']:,}")
        ax.annotate(f'Final: {ys[-1]:.3f}', (xs[-1], ys[-1]),
                    xytext=(-5, 12), textcoords='offset points', ha='right',
                    fontsize=10, fontweight='semibold', color=color)
        span = max(ys)-min(ys)
        ax.set_ylim(min(ys)-max(.10,span*.15), max(ys)+max(.12,span*.24))
    ax.set_title(f'{language.title()} | {"Training" if phase == "train" else "Validation"}',
                 loc='left', fontsize=13, fontweight='bold', pad=12)
    style(ax)
    ax.legend(frameon=False, fontsize=8.5, loc='upper right')
    return selected


def main():
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10,
                         'axes.labelcolor':'#273B50', 'text.color':'#172B3A',
                         'figure.facecolor':'white', 'axes.facecolor':'white',
                         'savefig.facecolor':'white', 'svg.fonttype':'none'})
    datasets = {}
    summary = {'source':'Saved logs from the completed bounded training session',
               'training_smoothing':'Trailing arithmetic mean of up to 15 logged minibatch points; raw points also shown',
               'validation':'Only complete held-out evaluations; no smoothing',
               'limitations':['Loss alone does not establish generated speech quality',
                              'English conditioning labels are estimated acoustic clusters',
                              'Best observed validation step does not imply its checkpoint was retained'],
               'languages':{}}
    for language in ('english','telugu'):
        source = ROOT/'output'/language/'bounded/metrics.jsonl'
        rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
        assert all(r['language']==language and len(r['losses'])==6 and
                   all(math.isfinite(v) for v in r['losses']) for r in rows)
        datasets[language]=rows
        directory=ROOT/'Results'/language.title()
        directory.mkdir(parents=True, exist_ok=True)
        with (ROOT/language.title()/'artifacts'/'loss_history.csv').open('w',newline='') as f:
            writer=csv.writer(f)
            writer.writerow(['utc','phase','step',*NAMES,'examples','complete'])
            writer.writerows([r['utc'],r['phase'],r['step'],*r['losses'],
                              r.get('examples',''),r.get('complete','')] for r in rows)
        tr=[r for r in rows if r['phase']=='train']
        va=[r for r in rows if r['phase']=='validation' and r.get('complete')]
        best=min(va,key=lambda r:r['losses'][0])
        summary['languages'][language]={'metrics_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'logged_train_points':len(tr),'complete_validation_points':len(va),
            'last_logged_training_step':tr[-1]['step'],
            'last_logged_training_total_loss':tr[-1]['losses'][0],
            'final_validation_step':va[-1]['step'],
            'final_validation_total_loss':va[-1]['losses'][0],
            'best_observed_validation_step':best['step'],
            'best_observed_validation_total_loss':best['losses'][0]}
        for phase in ('train','validation'):
            fig,ax=plt.subplots(figsize=(8,4.8))
            plot_phase(ax,rows,language,phase)
            fig.text(.12,.015,'Bounded initial session | English speaker labels estimated | Speech quality not yet evaluated',
                     fontsize=7,color='#5B6875')
            fig.tight_layout(rect=(0,.045,1,1))
            name='training_loss' if phase=='train' else 'validation_loss'
            fig.savefig(directory/(name+'.png'),dpi=200)
            fig.savefig(ROOT/language.title()/'artifacts'/(name+'.svg'))
            plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(13,8.5))
    for row,language in enumerate(('english','telugu')):
        for col,phase in enumerate(('train','validation')):
            plot_phase(axes[row,col],datasets[language],language,phase)
    fig.suptitle('FastSpeech 2 | English and Telugu training progress',x=.06,ha='left',
                 fontsize=19,fontweight='bold',y=.985)
    fig.text(.06,.94,'Completed initial session: 8,324 English updates / 8,323 Telugu updates',
             color='#5B6875',fontsize=11)
    fig.text(.06,.025,'Training: logged minibatches plus a trailing mean. Validation: complete fixed held-out splits.',
             color='#5B6875',fontsize=9)
    fig.text(.06,.006,'English speaker labels are estimated acoustic clusters. Loss curves do not establish generated speech quality.',
             color='#5B6875',fontsize=9)
    fig.tight_layout(rect=(.025,.05,1,.92),h_pad=2.5,w_pad=2.2)
    out=ROOT/'English/artifacts/Training'
    out.mkdir(parents=True,exist_ok=True)
    fig.savefig(out/'loss_overview.png',dpi=200)
    fig.savefig(out/'loss_overview.svg')
    plt.close(fig)
    # Portrait version keeps all four original series in one readable column.
    fig, axes = plt.subplots(4, 1, figsize=(8, 19))
    for ax, (language, phase) in zip(axes, (
        ('english', 'train'), ('english', 'validation'),
        ('telugu', 'train'), ('telugu', 'validation'),
    )):
        plot_phase(ax, datasets[language], language, phase)
    english_steps = summary['languages']['english']['final_validation_step']
    telugu_steps = summary['languages']['telugu']['final_validation_step']
    fig.suptitle('FastSpeech 2\nEnglish and Telugu loss curves', x=.12, ha='left',
                 fontsize=21, fontweight='bold', y=.987)
    fig.text(.12, .936,
             f'Initial session | English: {english_steps:,} steps | Telugu: {telugu_steps:,} steps',
             fontsize=10, color='#5B6875')
    fig.text(.12, .028, 'Training: logged minibatches and a 15-point trailing mean.\n'
             'Validation: complete fixed held-out splits; no smoothing.',
             fontsize=8, color='#5B6875', linespacing=1.5)
    fig.text(.12, .009, 'English speaker labels are estimated acoustic clusters.\n'
             'Loss curves do not establish generated speech quality.',
             fontsize=8, color='#5B6875', linespacing=1.5)
    fig.tight_layout(rect=(.025, .055, .995, .925), h_pad=3)
    fig.savefig(out/'loss_overview_vertical.png', dpi=200)
    fig.savefig(out/'loss_overview_vertical.svg')
    plt.close(fig)
    (out/'plot-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
