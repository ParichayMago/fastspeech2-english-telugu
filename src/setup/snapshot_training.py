#!/usr/bin/env python3
"""Create stable hard-linked checkpoint pairs for a concurrent local backup."""
import hashlib
import json
from pathlib import Path
import shutil
import time

ROOT = Path(__file__).resolve().parents[2]


def main():
    output = ROOT/'output'
    snapshot = output/'training-backup'
    proof = {}
    for lang in ('english', 'telugu'):
        src = output/lang/'bounded'
        dst = snapshot/lang/'bounded'
        dst.mkdir(parents=True, exist_ok=True)
        for attempt in range(10):
            entry = json.loads((src/'checkpoint.json').read_text())
            target = dst/'latest.pth.tar'
            target.unlink(missing_ok=True)
            target.hardlink_to(src/'latest.pth.tar')
            hasher = hashlib.sha256()
            with target.open('rb') as stream:
                for chunk in iter(lambda: stream.read(8*1024*1024), b''):
                    hasher.update(chunk)
            if hasher.hexdigest() == entry['sha256']:
                (dst/'checkpoint.json').write_text(json.dumps(entry, indent=2)+'\n')
                proof[lang] = entry['step']
                break
            time.sleep(.3)
        else:
            raise RuntimeError('Could not snapshot a consistent checkpoint pair')
        for name in ('metrics.jsonl', 'run-config.json'):
            shutil.copyfile(src/name, dst/name)
        if (src/'tensorboard').exists():
            shutil.copytree(src/'tensorboard', dst/'tensorboard', dirs_exist_ok=True)
    session = snapshot/'training-session'
    session.mkdir(parents=True, exist_ok=True)
    for name in ('status.json', 'console.log'):
        shutil.copyfile(output/'training-session'/name, session/name)
    print(json.dumps({'stable_snapshot_steps': proof}))


if __name__ == '__main__':
    main()
