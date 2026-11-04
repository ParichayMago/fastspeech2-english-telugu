#!/usr/bin/env python3
"""Launch one isolated learned-pause trainer only after local and CUDA gates."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
RUN = 'english-learned-pauses-pilot-20261104'
PARENT_SHA = 'b4955131de0b8915ede03eb2232fe4da9702136c1086d67bb31f5c0943f24aad'


def main():
    import torch
    parent = ROOT/'output/initialization/english-learned-pause-parent.pth.tar'
    assert hashlib.sha256(parent.read_bytes()).hexdigest() == PARENT_SHA
    evidence = ROOT/'src/artifacts/Training'/RUN
    transfer = json.loads((evidence/'transfer-verification.json').read_text())
    local = json.loads((evidence/'checks/local-verification.json').read_text())
    gpu = json.loads((evidence/'checks/gpu-readiness.json').read_text())
    assert transfer['status'] == local['status'] == gpu['status'] == 'passed'
    assert transfer['manifest_sha256'] == local['manifest_sha256']
    assert transfer['files'] == local['files'] == 18278
    assert gpu['parent_sha256'] == PARENT_SHA and gpu['smoke_weights_discarded']
    assert torch.cuda.is_available()
    session = ROOT/'output/runs'/RUN/'session'
    if (session/'status.json').exists():
        raise ValueError('Run already has training status; refusing a duplicate trainer')
    session.mkdir(parents=True, exist_ok=True)
    command = [sys.executable,'-u',str(ROOT/'src/setup/train_assignment.py'),
               '--run-id',RUN,'--fresh','--language','english','--learned-pauses',
               '--config-root','data/rebuild-config/learned-pauses-v1',
               '--initialize-from',str(parent.relative_to(ROOT)),
               '--initialize-sha256',PARENT_SHA,'--learning-rate','0.00002',
               '--new-head-learning-rate','0.0002','--minutes','90',
               '--max-steps','20000','--batch-size','8','--checkpoint-every','100',
               '--validation-every','500','--early-stop-patience','8']
    request = {'command':command, 'requested_utc':datetime.now(timezone.utc).isoformat(),
               'gpu':torch.cuda.get_device_name(0), 'parent_step':2500,
               'step_semantics':'additional learned-pause pilot updates from zero'}
    with (session/'launch-request.json').open('x') as f:
        f.write(json.dumps(request, indent=2)+'\n')
    with (session/'console.log').open('ab') as log:
        process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    request['pid'] = process.pid
    (session/'launch-request.json').write_text(json.dumps(request, indent=2)+'\n')
    print(json.dumps(request), flush=True)


if __name__ == '__main__':
    main()
