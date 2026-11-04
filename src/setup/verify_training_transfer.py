#!/usr/bin/env python3
"""Verify the exact corrected training arrays uploaded from the Mac."""
import hashlib
import argparse
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',default='data/features/gap-fixed-v2/training-transfer-sha256.json')
    parser.add_argument('--report',default='src/artifacts/Training/gap-fixed-v2-20261103/transfer-verification.json')
    args=parser.parse_args()
    manifest=(ROOT/args.manifest).resolve()
    assert manifest.is_relative_to(ROOT/'data/features')
    obj=json.loads(manifest.read_text());assert obj['feature_version'] in ('gap-fixed-v2','learned-pauses-v1')
    for name,entry in obj['files'].items():
        p=ROOT/name
        assert name.startswith(('data/features/gap-fixed-v2/english/',
                                'data/features/gap-fixed-v2/telugu/',
                                'data/features/learned-pauses-v1/english/',
                                'data/rebuild-config/learned-pauses-v1/English/'))
        assert p.stat().st_size==entry['bytes'],name
        assert hashlib.sha256(p.read_bytes()).hexdigest()==entry['sha256'],name
    result={'status':'passed','feature_version':obj['feature_version'],
            'files':len(obj['files']),'bytes':sum(x['bytes'] for x in obj['files'].values()),
            'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest()}
    path=(ROOT/args.report).resolve();assert path.is_relative_to(ROOT/'src/artifacts/Training')
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
