#!/usr/bin/env python3
"""Print-readable report figures from measured logs and frozen test predictions."""
import json
from pathlib import Path
import shutil
import sys
import csv

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))


def main():
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'axes.titlesize':10,
        'axes.labelsize':8.5,'xtick.labelsize':7.5,'ytick.labelsize':7.5,'axes.spines.top':False,
        'axes.spines.right':False,'savefig.facecolor':'white'})
    d=json.loads((ROOT/'Results/Assignment/analysis/measured-analysis.json').read_text())
    out=ROOT/'Report/figures';out.mkdir(exist_ok=True)
    def smooth(y):return [np.mean(y[max(0,i-24):i+1]) for i in range(len(y))]
    def losses(ax,run,validation_only=False):
        t=run['train'];v=run['validation']
        if not validation_only:
            xs=[x['step'] for x in t];ys=[x['losses'][0] for x in t]
            ax.plot(xs,ys,color='#9ea9b4',alpha=.28,lw=.5,label='Minibatch (raw)')
            ax.plot(xs,smooth(ys),color='#225c8a',lw=1.3,label='Train, 25-record mean')
        ax.plot([x['step'] for x in v],[x['losses'][0] for x in v],color='#0b7f72',marker='o',ms=2.4,lw=1.1,label='Complete validation')
        b=run['best'];ax.axvline(b['step'],color='#bc772b',ls='--',lw=.9)
        ax.scatter([b['step']],[b['losses'][0]],color='#bc772b',s=24,zorder=5,label='Selected best')
        ax.grid(alpha=.2);ax.set_xlabel('Optimizer updates');ax.set_ylabel('Logged total loss')
        ax.legend(fontsize=6.6,frameon=False)
    fig,axs=plt.subplots(2,2,figsize=(7.1,5.5),layout='constrained')
    for col,language in enumerate(('english','telugu')):
        run=d['training_runs']['gap-fixed-v2-20261103/'+language]
        losses(axs[0,col],run);losses(axs[1,col],run,True)
        axs[0,col].set_title(language.title()+' corrected training')
        axs[1,col].set_title(language.title()+' validation detail')
    fig.savefig(out/'corrected_training.png',dpi=220);plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(7.1,5.5),layout='constrained')
    for col,runid in enumerate(('english-finetune-pilot-20261104','english-learned-pauses-pilot-20261104')):
        run=d['training_runs'][runid+'/english'];losses(axs[0,col],run);losses(axs[1,col],run,True)
        axs[0,col].set_title('English fixed-LR fine-tune' if col==0 else 'English learned-pause pilot')
        axs[1,col].set_title('Original objective' if col==0 else 'New objective (extra pause losses)')
        axs[0,col].set_xlabel('Additional pilot updates');axs[1,col].set_xlabel('Additional pilot updates')
    fig.savefig(out/'english_pilots.png',dpi=220);plt.close(fig)
    run=d['training_runs']['english-learned-pauses-pilot-20261104/english']
    fig,axs=plt.subplots(1,2,figsize=(7.1,2.9),layout='constrained')
    rows=run['validation'];x=[r['step'] for r in rows]
    for key,label in [('precision','Precision'),('recall','Recall'),('f1','F1')]:
        axs[0].plot(x,[r['pause_metrics'][key] for r in rows],marker='o',ms=3,label=label,lw=1.2)
    axs[0].set(ylim=(-.02,1.02),xlabel='Additional pilot updates',ylabel='Pause detection score',title='Held-out pause detection')
    axs[0].legend(fontsize=7,frameon=False);axs[0].grid(alpha=.2)
    axs[1].plot(x,[r['pause_metrics']['positive_pause_mae_seconds']*1000 for r in rows],marker='o',ms=3,color='#225c8a')
    axs[1].axvline(1000,color='#bc772b',ls='--',lw=.9)
    axs[1].set(xlabel='Additional pilot updates',ylabel='Positive-pause MAE (ms)',title='Pause length did not improve clearly')
    axs[1].grid(alpha=.2);fig.savefig(out/'pause_validation.png',dpi=220);plt.close(fig)
    # Replot every variance curve with a numeric phone-index axis so that all
    # thirty curves are readable on portrait report pages; original full-label
    # figures and raw values remain unchanged in Results/.
    targets={s: [(w['start'],w['end']) for w in v['target_word_stats']] for s,v in d['samples'].items()}
    for sid,sample in d['samples'].items():
        lang=sample['language'];directory=ROOT/'Results'/lang.title()/'test_generated_audios'
        info=json.loads((directory/(sid+'.json')).read_text());a=np.load(ROOT/info['predictions_path']);dur=a['duration_frames'];n=len(dur);x=np.arange(1,n+1)
        fig,axs=plt.subplots(3,1,figsize=(7.1,4.6),sharex=True,layout='constrained')
        vals=[dur*256/22050,a['predicted_f0_hz'],a['predicted_spectral_energy']]
        ylabels=['Duration (s)','F0 conditioning (Hz)','Spectral energy (L2)']
        for index,(ax,values,label) in enumerate(zip(axs,vals,ylabels)):
            if index==0:ax.bar(x,values,width=.8,color='#225c8a')
            else:ax.plot(x,np.where(dur>0,values,np.nan),color='#225c8a',lw=1,marker='o',ms=1.9)
            for start,end in targets[sid]:ax.axvspan(start+.5,end+.5,color='#0b7f72',alpha=.12,zorder=0)
            for i,b in enumerate(info['boundary_types']):
                if b:ax.axvspan(i+.6,i+1.4,color='#d7a56f',alpha=.1,zorder=0)
            ax.set_ylabel(label);ax.grid(axis='y',alpha=.18);ax.set_xlim(.2,n+.8)
        ticks=np.unique(np.r_[1,np.arange(10,n+1,10),n]);axs[-1].set_xticks(ticks)
        axs[-1].set_xlabel('Input token index (all points shown; green = highlighted test word)')
        fig.savefig(out/(sid+'_variance.png'),dpi=220);plt.close(fig)
    # Canonical requested loss files now describe the selected runs. Preserve
    # prior root-level bounded-run plots and CSV before replacement.
    for language,runid in [('english','english-learned-pauses-pilot-20261104'),('telugu','gap-fixed-v2-20261103')]:
        root=ROOT/'Results'/language.title();archive=ROOT/'Results/Training/legacy-initial-bounded-loss-plots'/language.title();archive.mkdir(parents=True,exist_ok=True)
        for file in ('training_loss.png','validation_loss.png','training_loss.svg','validation_loss.svg','loss_history.csv'):
            p=root/file
            if p.exists() and not (archive/file).exists():shutil.copyfile(p,archive/file)
        run=d['training_runs'][runid+'/'+language]
        for kind in ('training','validation'):
            fig,ax=plt.subplots(figsize=(8,4.6),layout='constrained')
            if kind=='training':
                t=run['train'];xs=[r['step'] for r in t];ys=[r['losses'][0] for r in t]
                ax.plot(xs,ys,color='#b0b7c0',alpha=.4,lw=.6,label='Raw minibatch loss')
                ax.plot(xs,smooth(ys),color='#225c8a',label='25-record moving mean',lw=1.4)
            else:losses(ax,run,True)
            ax.set(title=f'{language.title()} {kind} loss | {runid}',xlabel='Additional pilot updates' if language=='english' else 'Optimizer updates',ylabel='Logged total loss')
            ax.grid(alpha=.2);ax.legend(fontsize=8)
            for ext in ('png','svg'):fig.savefig(root/(kind+'_loss.'+ext),dpi=180)
            plt.close(fig)
        with (root/'loss_history.csv').open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['run_id','phase','step','total','mel','mel_postnet','pitch','energy','duration','pause_presence','pause_duration','complete'])
            for phase in ('train','validation'):
                for row in run[phase]:w.writerow([runid,phase,row['step'],*row['losses'],*(['']*(8-len(row['losses']))),row.get('complete','')])
    print(json.dumps({'report_figures':len(list(out.glob('*.png'))),'canonical_loss_plots_updated':True,
                      'old_root_loss_artifacts_preserved':True}))


if __name__=='__main__':main()
