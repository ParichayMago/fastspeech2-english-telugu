#!/usr/bin/env python3
"""One-off local guard: back up this assignment, check credit, stop its Pod.

This is not a recurring GPU availability automation. It needs the Mac awake,
network access, and existing CLI credentials; it never funds an account.
"""
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
POD = '2z7xskg1vc616m'
CLI = '/Users/parichay/.local/bin/runpodctl'
RATE = 0.28694444444444445
CREATED = datetime.fromisoformat('2026-11-02T16:44:01.067+00:00')
SSH = ['ssh', '-p', '22117', '-i', str(Path.home()/'.ssh/id_ed25519'),
       '-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
       '-o', 'StrictHostKeyChecking=yes', '-o', 'UserKnownHostsFile='+str(ROOT/'.git/runpod_known_hosts')]
DEST = 'root@157.157.221.29'
REMOTE = '/workspace/fastspeech2-english-telugu'
REPORT = ROOT / 'English/artifacts/Training/supervisor.json'


def call(argv, timeout=90):
    return subprocess.run(argv, cwd=ROOT, check=True, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout).stdout


def write(value):
    value['updated_utc'] = datetime.now(timezone.utc).isoformat()
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    tmp = REPORT.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2)+'\n')
    tmp.replace(REPORT)


def sync_and_verify():
    import shlex
    call(SSH+[DEST, '/workspace/.venvs/fastspeech2/bin/python '+REMOTE+'/English/setup/snapshot_training.py'])
    # No --delete. All keys and unrelated directories remain outside the transfer.
    call(['rsync', '-az', '--timeout=60', '-e', shlex.join(SSH),
          '--exclude=*.tmp', '--exclude=*.lock',
          DEST+':'+REMOTE+'/output/training-backup/', str(ROOT/'output')+'/'], timeout=180)
    proof = {}
    for lang in ('english', 'telugu'):
        directory = ROOT/'output'/lang/'bounded'
        entry = json.loads((directory/'checkpoint.json').read_text())
        path = ROOT/entry['path']
        assert path == directory/'latest.pth.tar'
        assert path.stat().st_size == entry['bytes']
        hasher = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                hasher.update(chunk)
        digest = hasher.hexdigest()
        assert digest == entry['sha256'], 'Checkpoint changed during sync; retry'
        assert entry['step'] > 0
        proof[lang] = entry
    return proof


def main():
    lock = (ROOT/'.git/assignment-training-supervisor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    started = time.monotonic()
    state = {'status': 'monitoring', 'pod_id': POD, 'pid': __import__('os').getpid(),
             'credit_reserve_usd': 0.18, 'assignment_stop_threshold_usd': 19.25,
             'max_session_minutes': 48, 'automatic_top_up': False,
             'stop_policy': 'request trainer stop, verify local SHA256 checkpoints, then stop only approved Pod',
             'scope': 'one_off_training_guard_not_gpu_stock_automation'}
    write(state)
    while True:
        try:
            account = json.loads(call([CLI, 'user']))
            credit = float(account['clientBalance'])
            elapsed = (datetime.now(timezone.utc)-CREATED).total_seconds()/3600
            spend = 0.1654 + max(0, elapsed)*RATE
            state.update(credit_usd=credit, current_account_spend_per_hour=account['currentSpendPerHr'],
                         estimated_total_assignment_spend_usd=round(spend,4))
            reasons=[]
            if credit <= .18:
                reasons.append('credit_reserve')
            if spend >= 19.25:
                reasons.append('assignment_budget_reserve')
            if time.monotonic()-started >= 48*60:
                reasons.append('supervisor_time_limit')
            if reasons:
                call(SSH+[DEST, 'touch '+REMOTE+'/output/training-session/STOP'])
                state['stop_reasons'] = reasons
            try:
                status = json.loads(call(SSH+[DEST, 'cat '+REMOTE+'/output/training-session/status.json']))
                state['training'] = status
            except Exception as exc:
                status={}
                state['status_read_error'] = str(exc)
            terminal = status.get('status') in ('completed_bounded_session', 'failed')
            try:
                state['verified_local_checkpoints'] = sync_and_verify()
                state['backup_verified_utc'] = datetime.now(timezone.utc).isoformat()
            except Exception as exc:
                state['last_backup_error'] = str(exc)
                write(state)
                time.sleep(30)
                continue
            if terminal:
                # Final immutable files now verified; preprocessing was independently backed up.
                preprocessing = json.loads((ROOT/'English/artifacts/Preprocessing/backup-verification.json').read_text())
                assert preprocessing['status'] == 'passed'
                state['training_final_backup_verified'] = True
                state['stop_result'] = json.loads(call([CLI, 'pod', 'stop', POD]))
                pod = json.loads(call([CLI, 'pod', 'get', POD]))
                state['pod_state_after_stop'] = {k:pod.get(k) for k in ('id','desiredStatus','runtimeStatus','costPerHr','volumeInGb')}
                state['status'] = 'completed_and_assignment_pod_stopped'
                write(state)
                return
            state['status'] = 'monitoring'
            state.pop('last_error', None)
        except Exception as exc:
            state['last_error'] = type(exc).__name__+': '+str(exc)
        write(state)
        time.sleep(45)


if __name__ == '__main__':
    main()
