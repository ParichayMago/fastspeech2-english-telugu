#!/usr/bin/env python3
"""One-off local credit guard and asynchronous verified backup for one run."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import fcntl
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2]
CLI='/Users/parichay/.local/bin/runpodctl'
REMOTE='/workspace/fastspeech2-english-telugu'


def call(argv,timeout=60):
    return subprocess.run(argv,cwd=ROOT,check=True,text=True,stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE,timeout=timeout).stdout


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def verify_snapshot(run):
    root=ROOT/'output/runs'/run
    snapshot=json.loads((root/'snapshot.json').read_text())
    assert snapshot['run_id']==run
    proof={}
    for language,entries in snapshot['checkpoints'].items():
        assert language in ('english','telugu')
        proof[language]={}
        for name,entry in entries.items():
            p=ROOT/entry['path']
            assert p.parent==root/language
            assert p.name in ('latest.pth.tar','best.pth.tar')
            assert p.stat().st_size==entry['bytes'] and digest(p)==entry['sha256']
            proof[language][name]=entry
    return {'checkpoints':proof,'observed_status':snapshot['observed_status'],
            'verified_utc':datetime.now(timezone.utc).isoformat()}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runtime',required=True);args=p.parse_args()
    cfg=json.loads(Path(args.runtime).read_text())
    pod,run=cfg['pod_id'],cfg['run_id']
    assert run.replace('-','').replace('_','').isalnum()
    assert cfg['maximum_total_assignment_spend_usd']==20
    lock=(ROOT/'.git/assignment-training-supervisor.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    ssh=['ssh','-p',str(cfg['ssh_port']),'-i',str(Path.home()/'.ssh/id_ed25519'),
         '-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','ConnectTimeout=15',
         '-o','StrictHostKeyChecking=yes','-o','UserKnownHostsFile='+str(ROOT/'.git/runpod_known_hosts')]
    dest='root@'+cfg['ssh_host']
    report=ROOT/'English/artifacts/Training'/run/'supervisor.json';report.parent.mkdir(parents=True,exist_ok=True)
    remote_session=REMOTE+'/output/runs/'+run+'/session'
    activated=datetime.fromisoformat(cfg['activated_utc'])
    state={'status':'monitoring_setup','pod_id':pod,'run_id':run,'pid':__import__('os').getpid(),
           'runtime':cfg,'automatic_top_up':False}
    pool=ThreadPoolExecutor(max_workers=1);future=None;backup=None;requested_stop=False

    def save():
        state['updated_utc']=datetime.now(timezone.utc).isoformat()
        tmp=report.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(report)

    def backup_run():
        call(ssh+[dest,'/workspace/.venvs/fastspeech2/bin/python '+REMOTE+
                  '/Telugu/setup/snapshot_training_run.py --run-id '+shlex.quote(run)],timeout=120)
        local=ROOT/'output/runs'/run;local.mkdir(parents=True,exist_ok=True)
        call(['rsync','-az','--timeout=60','--exclude=*.tmp','--exclude=*.lock',
              '-e',shlex.join(ssh),dest+':'+REMOTE+'/output/snapshots/'+run+'/',str(local)+'/'],timeout=600)
        return verify_snapshot(run)

    save()
    try:
        while True:
            try:
                now=datetime.now(timezone.utc)
                elapsed=max(0,(now-activated).total_seconds())
                spend=cfg['conservative_prior_assignment_spend_usd']+elapsed/3600*cfg['total_running_rate_usd_per_hour']
                account=json.loads(call([CLI,'user']))
                state.update(credit_usd=account['clientBalance'],
                    account_spend_per_hour=account['currentSpendPerHr'],
                    conservative_total_assignment_spend_usd=round(spend,4),
                    billing_elapsed_seconds=round(elapsed,1))
                reasons=[]
                if account['clientBalance']<=cfg['credit_reserve_usd']:reasons.append('credit_reserve')
                if spend>=cfg['assignment_stop_threshold_usd']:reasons.append('assignment_budget_reserve')
                if elapsed>=cfg['maximum_elapsed_minutes']*60:reasons.append('elapsed_limit')
                if reasons:
                    requested_stop=True;state['stop_reasons']=reasons
                    try:call(ssh+[dest,'mkdir -p '+remote_session+' && touch '+remote_session+'/STOP'])
                    except Exception:pass
                try:
                    training=json.loads(call(ssh+[dest,'cat '+remote_session+'/status.json']))
                    state['training']=training;state['status']='monitoring_training'
                    state.pop('status_read_error',None)
                except Exception as exc:
                    training={};state['status_read_error']=str(exc)
                terminal=training.get('status') in ('completed_bounded_session','failed')
                if future and future.done():
                    try:
                        backup=future.result();state['verified_backup']=backup;state.pop('backup_error',None)
                    except Exception as exc:state['backup_error']=str(exc)
                    future=None
                final_verified=terminal and backup and backup['observed_status'].get('status') in (
                    'completed_bounded_session','failed') and all(
                    backup['checkpoints'].get(lang,{}).get('checkpoint',{}).get('step')==details['checkpoint']['step']
                    for lang,details in training.get('languages',{}).items() if 'checkpoint' in details)
                if final_verified:
                    rebuild=json.loads((ROOT/'English/artifacts/Preprocessing/gap-fixed-v2/rebuild-summary.json').read_text())
                    assert rebuild['new_artifacts']['status']=='passed'
                    result=json.loads(call([CLI,'pod','stop',pod]))
                    # Pod responses contain environment secrets; persist a whitelist only.
                    state['stop_result']={k:result.get(k) for k in ('id','desiredStatus','runtimeStatus')}
                    result=json.loads(call([CLI,'pod','get',pod]))
                    state['pod_state_after_stop']={k:result.get(k) for k in ('id','desiredStatus','runtimeStatus')}
                    state['status']='completed_and_assignment_pod_stopped';save();return
                if requested_stop and not training and elapsed>cfg['maximum_elapsed_minutes']*60+300:
                    # No training status was ever produced: no new model work exists
                    # to lose. All feature masters remain on the Mac.
                    if 'training' not in state:
                        result=json.loads(call([CLI,'pod','stop',pod]))
                        state['status']='setup_timed_out_and_pod_stopped';save();return
                if training and future is None:
                    future=pool.submit(backup_run)
                state.pop('last_error',None)
            except Exception as exc:state['last_error']=type(exc).__name__+': '+str(exc)
            save();time.sleep(30)
    finally:
        pool.shutdown(wait=True)


if __name__=='__main__':main()
