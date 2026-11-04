#!/usr/bin/env python3
"""Restore only hash-verified model files from the public submission archive."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, help='Use an already downloaded submission ZIP')
    parser.add_argument('--check-only', action='store_true', help='Verify local weights without downloading')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'Telugu/setup/checkpoint_manifest.json').read_text())
    pending = []
    for entry in manifest['files']:
        destination = ROOT / entry['path']
        if destination.exists():
            if destination.stat().st_size != entry['bytes'] or digest(destination) != entry['sha256']:
                raise ValueError('Existing model differs; preserve or relocate it first: ' + entry['path'])
        else:
            pending.append(entry)
    if args.check_only:
        if pending:
            parser.error('Missing model files: ' + ', '.join(e['path'] for e in pending))
        print(f'All {len(manifest["files"])} model files passed SHA256 verification.')
        return
    if not pending:
        print(f'All {len(manifest["files"])} model files already match; no download needed.')
        return
    with tempfile.TemporaryDirectory(prefix='fs2-models-') as temporary:
        archive = args.archive.resolve() if args.archive else Path(temporary) / 'submission.zip'
        if not args.archive:
            print('Downloading the complete submission archive; only the three model files will be restored.', flush=True)
            with urllib.request.urlopen(manifest['archive_url'], timeout=60) as response, archive.open('wb') as stream:
                shutil.copyfileobj(response, stream)
        with zipfile.ZipFile(archive) as bundle:
            for entry in pending:
                destination = ROOT / entry['path']
                destination.parent.mkdir(parents=True, exist_ok=True)
                # Read only fixed manifest paths; never extract arbitrary archive paths.
                member = 'project/' + entry['path']
                if bundle.getinfo(member).file_size != entry['bytes']:
                    raise ValueError('Model size mismatch: ' + member)
                with tempfile.NamedTemporaryFile(dir=destination.parent, prefix='.restore-', delete=False) as stream:
                    staged = Path(stream.name)
                    try:
                        with bundle.open(member) as source:
                            shutil.copyfileobj(source, stream)
                        stream.close()
                        if digest(staged) != entry['sha256']:
                            raise ValueError('Model SHA256 mismatch: ' + member)
                        staged.replace(destination)
                    finally:
                        staged.unlink(missing_ok=True)
                print('Verified: ' + entry['path'], flush=True)


if __name__ == '__main__':
    main()
