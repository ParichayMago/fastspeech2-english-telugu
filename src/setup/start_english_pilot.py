#!/usr/bin/env python3
"""Start exactly one prepared English pilot on the approved remote checkout."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
RUN = 'english-finetune-pilot-20261104'
PARENT_SHA256 = 'dd8193a7d69fd89b1a9209f028ef22bba6eca22477298d8e90b9120cdc2b6f2b'


def main():
    import torch
    parent = ROOT / 'output/initialization/english-best-12000.pth.tar'
    assert hashlib.sha256(parent.read_bytes()).hexdigest() == PARENT_SHA256
    report = json.loads((ROOT/'src/artifacts/Training'/RUN/'transfer-verification.json').read_text())
    assert report['status'] == 'passed' and report['files'] == 14621
    assert torch.cuda.is_available()
    session = ROOT / 'output/runs' / RUN / 'session'
    if (session/'status.json').exists():
        raise ValueError('Pilot already has a session status; refusing duplicate launch')
    session.mkdir(parents=True, exist_ok=True)
    command = [sys.executable,'-u',str(ROOT/'src/setup/train_assignment.py'),'--run-id',RUN,
               '--fresh','--language','english','--initialize-from',str(parent.relative_to(ROOT)),
               '--initialize-sha256',PARENT_SHA256,'--learning-rate','0.00002',
               '--minutes','45','--max-steps','12000','--batch-size','8',
               '--checkpoint-every','100','--validation-every','500','--early-stop-patience','5']
    request = {'command': command, 'requested_utc': datetime.now(timezone.utc).isoformat(),
               'gpu': torch.cuda.get_device_name(0), 'parent_step': 12000}
    with (session/'launch-request.json').open('x') as f:
        f.write(json.dumps(request, indent=2)+'\n')
    with (session/'console.log').open('ab') as log:
        process = subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,
                                   stderr=subprocess.STDOUT,start_new_session=True)
    request['pid'] = process.pid
    (session/'launch-request.json').write_text(json.dumps(request,indent=2)+'\n')
    print(json.dumps(request),flush=True)


if __name__ == '__main__':
    main()
