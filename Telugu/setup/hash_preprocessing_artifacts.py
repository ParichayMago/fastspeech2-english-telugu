#!/usr/bin/env python3
"""Hash immutable preprocessing artifacts, or verify a local restored copy."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['write','verify'])
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[2]
    manifest=root/'data/preprocessing/artifact-sha256.json'
    if args.action=='write':
        files={}
        for language in ['english','telugu']:
            for base in [root/'data/features'/language,root/'data/preprocessing'/language]:
                if not base.exists():raise ValueError('Missing completed artifact directory')
                for p in sorted(base.rglob('*')):
                    if p.is_file():
                        if not p.resolve().is_relative_to(root/'data'):
                            raise ValueError('Artifact symlink escapes assignment data')
                        files[str(p.relative_to(root))]={'sha256':digest(p),'bytes':p.stat().st_size}
        for p in [root/'Telugu/text/assignment_phones.json'] + [p for language in ['English','Telugu']
                for p in sorted((root/'Telugu/config'/language).glob('*.yaml'))]:
            files[str(p.relative_to(root))]={'sha256':digest(p),'bytes':p.stat().st_size}
        manifest.write_text(json.dumps({'hash':'SHA-256','files':files},sort_keys=True,indent=2)+'\n')
        print(json.dumps({'status':'hashed','files':len(files),'bytes':sum(x['bytes'] for x in files.values())}),flush=True)
    else:
        obj=json.loads(manifest.read_text());files=obj['files']
        for name,entry in files.items():
            p=root/name
            if not p.is_file() or p.stat().st_size!=entry['bytes'] or digest(p)!=entry['sha256']:
                raise ValueError('Local artifact differs from GPU hash: '+name)
        report={'status':'passed','files':len(files),'bytes':sum(x['bytes'] for x in files.values()),
                'algorithm':'SHA-256','artifact_manifest_sha256':digest(manifest),
                'scope':'Local files verified against the manifest generated on the assignment GPU.'}
        out=root/'English/artifacts/Preprocessing/backup-verification.json'
        out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
