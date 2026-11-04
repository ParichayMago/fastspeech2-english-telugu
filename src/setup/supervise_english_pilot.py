#!/usr/bin/env python3
"""One English pilot: independent credit/deadline guard and verified backups.

Hard cost/deadline stop takes priority over a failed final backup. The initialized
parent checkpoint is already verified locally before this supervisor is started.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import shlex
import subprocess
import time

from supervise_training_run import verify_snapshot

ROOT = Path(__file__).resolve().parents[2]
CLI = '/Users/parichay/.local/bin/runpodctl'
REMOTE = '/workspace/fastspeech2-english-telugu'


def call(argv, timeout=15):
    return subprocess.run(argv, cwd=ROOT, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, check=True, timeout=timeout).stdout


def limits(cfg, elapsed, credit=None):
    estimate = elapsed / 3600 * cfg['running_rate_usd_per_hour']
    credit_delta = max(0, cfg['starting_credit_usd'] - credit) if credit is not None else 0
    charged = max(estimate, credit_delta)
    hard = (charged >= cfg['hard_pilot_stop_usd'] or elapsed >= cfg['hard_elapsed_minutes']*60
            or cfg['prior_assignment_spend_usd'] + estimate >= 19
            or (credit is not None and credit <= 1))
    soft = charged >= cfg['soft_pilot_stop_usd'] or elapsed >= cfg['soft_elapsed_minutes']*60
    return estimate, credit_delta, soft, hard


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', required=True)
    args = parser.parse_args()
    runtime_path = Path(args.runtime).resolve()
    cfg = json.loads(runtime_path.read_text())
    assert cfg['pod_id'] == cfg['authorized_pod_id']
    assert cfg['maximum_total_assignment_spend_usd'] == 20
    budget = cfg.get('pilot_budget_usd', 1)
    assert 0 < budget <= 2
    assert 0 < cfg['soft_pilot_stop_usd'] < cfg['hard_pilot_stop_usd'] <= budget-.15
    assert 0 < cfg['running_rate_usd_per_hour'] <= .73
    assert cfg['hard_elapsed_minutes'] <= 180
    assert cfg['hard_elapsed_minutes']/60*cfg['running_rate_usd_per_hour'] < cfg['hard_pilot_stop_usd']-.05
    lock = (ROOT / '.git/assignment-training-supervisor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    run, pod = cfg['run_id'], cfg['pod_id']
    assert run.replace('-', '').replace('_', '').isalnum()
    activated = datetime.fromisoformat(cfg['activated_utc'])
    report = ROOT / 'src/artifacts/Training' / run / 'supervisor.json'
    report.parent.mkdir(parents=True, exist_ok=True)
    state = {'status': 'guarding_setup', 'pod_id': pod, 'run_id': run,
             'pid': __import__('os').getpid(), 'runtime': cfg, 'started_utc': cfg['activated_utc']}
    pool = ThreadPoolExecutor(max_workers=1)
    future = None

    def save():
        state['updated_utc'] = datetime.now(timezone.utc).isoformat()
        tmp = report.with_suffix('.tmp'); tmp.write_text(json.dumps(state, indent=2) + '\n'); tmp.replace(report)

    def ssh_command(config):
        return ['ssh', '-p', str(config['ssh_port']), '-i', str(Path.home()/'.ssh/id_ed25519'),
                '-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8',
                '-o', 'StrictHostKeyChecking=yes', '-o', 'UserKnownHostsFile='+str(ROOT/'.git/runpod_known_hosts'),
                'root@'+config['ssh_host']]

    def backup(config):
        ssh = ssh_command(config)
        call(ssh + ['/workspace/.venvs/fastspeech2/bin/python '+REMOTE+'/src/setup/snapshot_training_run.py --run-id '+shlex.quote(run)], 45)
        local = ROOT / 'output/runs' / run; local.mkdir(parents=True, exist_ok=True)
        call(['rsync','-az','--timeout=30','--exclude=*.tmp','--exclude=*.lock','-e',shlex.join(ssh[:-1]),
              ssh[-1]+':'+REMOTE+'/output/snapshots/'+run+'/',str(local)+'/'], 180)
        return verify_snapshot(run)

    def stop(reason):
        state['stop_reason'] = reason
        result = json.loads(call([CLI, 'pod', 'get', pod], 20))
        if result.get('desiredStatus') not in ('EXITED','STOPPED'):
            result = json.loads(call([CLI, 'pod', 'stop', pod], 20))
            state['stop_result'] = {k: result.get(k) for k in ('id','desiredStatus','runtimeStatus')}
            result = json.loads(call([CLI, 'pod', 'get', pod], 20))
        state['pod_state_after_stop'] = {k: result.get(k) for k in ('id','desiredStatus','runtimeStatus')}
        if result.get('desiredStatus') not in ('EXITED','STOPPED'):
            raise RuntimeError('Pod stop not yet confirmed')
        state['status'] = ('completed_and_assignment_pod_stopped' if reason == 'final_backup_verified'
                           else 'budget_or_safety_stop')
        save()

    save()
    try:
        while True:
            cfg = json.loads(runtime_path.read_text())
            elapsed = max(0, (datetime.now(timezone.utc)-activated).total_seconds())
            credit = None
            try:
                user = json.loads(call([CLI,'user']))
                credit = user['clientBalance']; state['credit_usd'] = credit
                state.pop('credit_error', None)
            except Exception as exc:
                state['credit_error'] = str(exc)
            estimate, delta, soft, hard = limits(cfg, elapsed, credit)
            state.update(elapsed_seconds=round(elapsed,1), estimated_pilot_spend_usd=round(estimate,4),
                         account_credit_decrease_usd=round(delta,4),
                         conservative_total_assignment_spend_usd=round(cfg['prior_assignment_spend_usd']+estimate,4))
            if cfg.get('abort_requested'):
                state['abort_reason'] = cfg['abort_requested']; hard = True
            try:
                if hard:
                    stop('hard_pilot_budget_or_setup_abort'); return
                training = {}
                if cfg.get('ssh_ready'):
                    ssh = ssh_command(cfg)
                    if soft:
                        call(ssh + ['mkdir -p '+REMOTE+'/output/runs/'+run+'/session && touch '+REMOTE+'/output/runs/'+run+'/session/STOP'])
                        state['graceful_stop_requested'] = True
                    try:
                        training = json.loads(call(ssh + ['cat '+REMOTE+'/output/runs/'+run+'/session/status.json']))
                        state['training'] = training; state['status'] = 'monitoring_training'
                        state.pop('status_read_error',None)
                    except Exception as exc:
                        state['status_read_error'] = str(exc)
                if not training and 'training' not in state and elapsed > cfg['setup_timeout_minutes']*60:
                    stop('setup_timeout'); return
                if future and future.done():
                    try:
                        state['verified_backup'] = future.result(); state.pop('backup_error',None)
                    except Exception as exc:
                        state['backup_error'] = str(exc)
                    future = None
                proof = state.get('verified_backup',{})
                terminal = training.get('status') in ('completed_bounded_session','failed')
                if training.get('status') == 'failed' and not training.get('languages'):
                    stop('training_failed_before_checkpoint'); return
                final_verified = (terminal and proof.get('observed_status',{}).get('status') in ('completed_bounded_session','failed')
                    and proof.get('checkpoints',{}).get('english',{}).get('checkpoint',{}).get('step')
                    == training.get('languages',{}).get('english',{}).get('checkpoint',{}).get('step')
                    and 'checkpoint' in proof.get('checkpoints',{}).get('english',{}))
                if final_verified:
                    stop('final_backup_verified'); return
                if training and future is None:
                    future = pool.submit(backup, dict(cfg))
                state.pop('last_error',None)
            except Exception as exc:
                state['last_error'] = type(exc).__name__+': '+str(exc)
            save(); time.sleep(10)
    finally:
        pool.shutdown(wait=False)


if __name__ == '__main__':
    main()
