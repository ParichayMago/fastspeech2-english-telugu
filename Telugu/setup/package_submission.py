#!/usr/bin/env python3
"""Build and verify the complete assignment ZIP with an explicit file manifest."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import zipfile

ROOT=Path(__file__).resolve().parents[2]


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def submission_files():
    files={ROOT/name for name in ('README.md','requirements.txt','LICENSE','.gitignore','.gitattributes')}
    for language in ('English','Telugu'):
        base=ROOT/language
        for path in base.rglob('*'):
            relative=path.relative_to(base)
            if relative.parts[:2]==('artifacts','Submission'):continue
            if relative.parts[:1]==('setup',) and path.suffix=='.json' and path.name not in ('assignment_test_sentences.json','checkpoint_manifest.json'):continue
            if relative.parts[:2] in (('reference','demo'),('reference','preprocessed_data')):continue
            files.add(path)
    files.update((ROOT/'Report').rglob('*'))
    files.update((ROOT/'Results').rglob('*'))
    return sorted(p for p in files if p.is_file() and '__pycache__' not in p.parts
                  and p.name not in ('.DS_Store','.gitkeep') and p.suffix not in ('.pyc','.pyo','.log')
                  and (not p.name.endswith('.pth.tar') or p.relative_to(ROOT).as_posix() in {
                      'English/checkpoints/final.pth.tar','Telugu/checkpoints/final.pth.tar',
                      'English/hifigan/generator_universal.pth.tar','Telugu/hifigan/generator_universal.pth.tar'})
                  and (p.suffix!='.zip' or p==ROOT/'English/artifacts/Assignment/assignment_test_outputs.zip'))


def main():
    destination=ROOT/'English/artifacts/Submission';destination.mkdir(exist_ok=True)
    files=submission_files()
    required=['English/train.py','Telugu/train.py','English/preprocess.py','Telugu/preprocess.py',
              'English/inference.py','Telugu/inference.py','English/config.yaml','Telugu/config.yaml',
              'English/checkpoints/final.pth.tar','Telugu/checkpoints/final.pth.tar',
              'English/hifigan/generator_universal.pth.tar','Telugu/hifigan/generator_universal.pth.tar','Report/Technical_Report.pdf','README.md','requirements.txt',
              'English/setup/training_recovery.py','English/setup/learned_pause_config.py',
              'English/setup/analyze_speaker_coverage.py','English/artifacts/Assignment/analysis/english-speaker-coverage.json',
              'English/docs/REPRODUCIBILITY.md',
              'English/audio_review/server.py','English/audio_review/static/index.html','English/audio_review/static/app.js',
              'English/config/learned-pauses-v1/English/preprocess.yaml','English/config/learned-pauses-v1/English/model.yaml',
              'English/config/learned-pauses-v1/English/train.yaml']
    paths={p.relative_to(ROOT).as_posix() for p in files}
    allowed_roots={'English','Telugu','Results','Report','README.md','requirements.txt','LICENSE','.gitignore','.gitattributes'}
    if {p.split('/')[0] for p in paths}-allowed_roots:
        raise ValueError('Unexpected top-level submission path')
    if not set(required)<=paths:raise ValueError('Required artifacts missing: '+repr(set(required)-paths))
    for language in ('English','Telugu'):
        assert len([p for p in paths if p.startswith(f'Results/{language}/test_generated_audios/') and p.endswith('.wav')])==5
        assert len([p for p in paths if p.startswith(f'Results/{language}/variance_adaptor_plots/') and p.endswith('.png')])==15
    patterns=[rb'gh[pousr]_[A-Za-z0-9]{25,}',rb'hf_[A-Za-z0-9]{25,}',rb'-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----',rb'rpa_[A-Za-z0-9]{25,}']
    for p in files:
        if p.stat().st_size<10_000_000 and p.suffix in ('.py','.sh','.json','.jsonl','.txt','.yaml','.md','.html'):
            raw=p.read_bytes()
            if any(re.search(pattern,raw) for pattern in patterns):
                raise ValueError('Credential pattern in selected file: '+str(p.relative_to(ROOT)))
    release=json.loads((ROOT/'English/setup/checkpoint_manifest.json').read_text())['release_tag']
    manifest={'created_utc':datetime.now(timezone.utc).isoformat(),'release_tag':release,
              'archive_layout':'project/','audios':10,'variance_plots':30,
              'exclusions':['raw datasets','credentials','account runtime state','environments','historical optimizer checkpoints','unrelated upstream demos'],
              'files':{str(p.relative_to(ROOT)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in files}}
    path=destination/'package-manifest.json';path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    files.append(path)
    archive=destination/('fastspeech2-'+release+'.zip')
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=4,allowZip64=True) as bundle:
        for p in files:bundle.write(p,'project/'+str(p.relative_to(ROOT)))
    with zipfile.ZipFile(archive) as bundle:
        assert bundle.testzip() is None
        for p,entry in manifest['files'].items():
            with bundle.open('project/'+p) as stream:
                h=hashlib.sha256()
                for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
                if h.hexdigest()!=entry['sha256']:raise ValueError('Archive content differs: '+p)
    (destination/'SHA256SUMS').write_text(sha(archive)+'  '+archive.name+'\n')
    print(json.dumps({'archive':str(archive.relative_to(ROOT)),'bytes':archive.stat().st_size,
                      'files':len(files),'sha256':sha(archive),'verified':'CRC and every manifest SHA256'},indent=2))


if __name__=='__main__':main()
