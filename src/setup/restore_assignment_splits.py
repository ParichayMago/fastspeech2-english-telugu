#!/usr/bin/env python3
"""Rebuild selected manifests from authorized dataset audits and public membership."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    members = [json.loads(s) for s in (ROOT/'src/artifacts/DataSplits/selection-membership.jsonl').read_text().splitlines()]
    report = {'selection_identity':'Published source rows, partitions and conditioning labels',
              'metadata_note':'Restored metadata may differ from historical manifests; original raw texts/audio come from the authorized source.',
              'partitions':{}}
    for language in ('english','telugu'):
        audit = json.loads((ROOT/f'src/artifacts/DatasetAudit/{language}-audit.json').read_text())
        source = {(r['shard'],r['row_in_shard']):r for r in
                  [json.loads(s) for s in (ROOT/audit['row_manifest']).read_text().splitlines()]}
        for split in ('train','validation'):
            rows=[]
            for member in members:
                if member['language']!=language or member['partition']!=split:
                    continue
                if audit['revision']!=member['dataset_revision']:
                    raise ValueError('Audit revision differs from the published selection')
                row=dict(source[(member['shard'],member['row_in_shard'])])
                if row['audio_sha256']!=member['audio_sha256']:
                    raise ValueError('Audited audio differs from selected source row')
                row.update({k:v for k,v in member.items() if k not in ('language','partition')})
                row['source_speaker_id']=source[(member['shard'],member['row_in_shard'])].get('speaker_id')
                row['speaker_id_provenance']='Restored frozen estimated acoustic label' if language=='english' else 'Source male conditioning label'
                rows.append(row)
            rows.sort(key=lambda r:r['sample_id'])
            content=''.join(json.dumps(r,sort_keys=True,ensure_ascii=False)+'\n' for r in rows)
            path=ROOT/f'data/splits/v1/{language}/{split}.jsonl'
            report['partitions'][language+'/'+split]={'rows':len(rows),'sha256':hashlib.sha256(content.encode()).hexdigest()}
            if not args.check_only:
                if path.exists() and path.read_text()!=content:
                    raise ValueError('Existing split differs; refusing overwrite: '+str(path))
                path.parent.mkdir(parents=True,exist_ok=True);path.write_text(content)
    if not args.check_only:
        (ROOT/'src/artifacts/DataSplits/restoration.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'checked_only':args.check_only,**report},indent=2))


if __name__=='__main__':
    main()
