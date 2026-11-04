#!/usr/bin/env python3
"""Create one stable, read-only snapshot of an isolated training run for backup."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

ROOT=Path(__file__).resolve().parent.parent


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-id',required=True);args=p.parse_args()
    assert args.run_id.replace('-','').replace('_','').isalnum()
    source=ROOT/'output/runs'/args.run_id
    target=ROOT/'output/snapshots'/args.run_id
    target.mkdir(parents=True,exist_ok=True)
    proof={}
    status=json.loads((source/'session/status.json').read_text())
    for lang in status['languages']:
        assert lang in ('english','telugu')
        src=source/lang;dst=target/lang;dst.mkdir(exist_ok=True)
        proof[lang]={}
        for name in ('checkpoint','best'):
            metadata=src/(name+'.json')
            if not metadata.exists():continue
            for attempt in range(10):
                entry=json.loads(metadata.read_text())
                checkpoint=ROOT/entry['path']
                assert checkpoint.parent==src and checkpoint.name in ('latest.pth.tar','best.pth.tar')
                output=dst/checkpoint.name;output.unlink(missing_ok=True);output.hardlink_to(checkpoint)
                if hashlib.sha256(output.read_bytes()).hexdigest()==entry['sha256']:
                    (dst/(name+'.json')).write_text(json.dumps(entry,indent=2)+'\n')
                    proof[lang][name]=entry;break
                time.sleep(.2)
            else:raise RuntimeError('Could not create stable checkpoint pair')
        for name in ('run-config.json','metrics.jsonl'):
            if (src/name).exists():shutil.copyfile(src/name,dst/name)
        if (src/'tensorboard').exists():shutil.copytree(src/'tensorboard',dst/'tensorboard',dirs_exist_ok=True)
    session=target/'session';session.mkdir(exist_ok=True)
    for name in ('status.json','console.log'):
        if (source/'session'/name).exists():shutil.copyfile(source/'session'/name,session/name)
    (target/'snapshot.json').write_text(json.dumps({'run_id':args.run_id,'observed_status':status,
                                                'checkpoints':proof},indent=2)+'\n')
    print(json.dumps({'run_id':args.run_id,'checkpoints':{k:{n:e['step'] for n,e in v.items()} for k,v in proof.items()}}))


if __name__=='__main__':main()
