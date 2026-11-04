#!/usr/bin/env python3
"""Build and verify the complete assignment ZIP with an explicit file manifest."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import zipfile

ROOT=Path(__file__).resolve().parent.parent


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def submission_files():
    files={ROOT/'README.md',ROOT/'requirements.txt',ROOT/'LICENSE',ROOT/'.gitignore'}
    files.update(ROOT.glob('*.py'))
    for name in ('English','Telugu','audio','model','preprocessor','transformer','utils','text','config','lexicon','docs','img','audio_review'):
        files.update((ROOT/name).rglob('*'))
    files.update((ROOT/'hifigan').glob('*.py'))
    files.update((ROOT/'hifigan').glob('*.json'))
    files.add(ROOT/'hifigan/LICENSE')
    for pattern in ('*.py','*.sh','requirements*.txt','assignment_test_sentences.json','checkpoint_manifest.json'):
        files.update((ROOT/'setup').glob(pattern))
    files.update((ROOT/'Report').rglob('*'))
    for language in ('English','Telugu'):
        files.update((ROOT/'Results'/language/'test_generated_audios').rglob('*'))
        files.update((ROOT/'Results'/language/'variance_adaptor_plots').rglob('*'))
        for pattern in ('*loss.png','*loss.svg','loss_history.csv'):
            files.update((ROOT/'Results'/language).glob(pattern))
    files.update((ROOT/'Results/Assignment/analysis').glob('*.json'))
    for name in ('index.html','verification.json','generation.json','checkpoint-selection.json','cli-verification.json','training-entrypoint-verification.json','variance_contact_sheet.png','assignment_test_outputs.zip'):
        files.add(ROOT/'Results/Assignment'/name)
    for name in ('DataSplits','DatasetAudit','Preprocessing'):
        files.update((ROOT/'Results'/name).rglob('*.json'))
        files.update((ROOT/'Results'/name).rglob('requirements*.txt'))
    files.add(ROOT/'Results/DataSplits/selection-membership.jsonl')
    files.add(ROOT/'Results/InitialSpeechReview/review.json')
    for run,language in [('gap-fixed-v2-20261103','english'),('gap-fixed-v2-20261103','telugu'),('english-finetune-pilot-20261104','english'),('english-learned-pauses-pilot-20261104','english')]:
        for name in ('metrics.jsonl','run-config.json'):
            files.add(ROOT/'Results/Training'/run/language/name)
    for p in (ROOT/'Results/Training').rglob('*'):
        if 'plots' in p.parts and p.suffix in ('.png','.svg','.csv'):
            files.add(p)
    files.add(ROOT/'hifigan/generator_universal.pth.tar')
    return sorted(p for p in files if p.is_file() and '__pycache__' not in p.parts
                  and p.name not in ('.DS_Store','.gitkeep') and p.suffix not in ('.pyc','.pyo','.log')
                  and (p.suffix!='.zip' or p==ROOT/'Results/Assignment/assignment_test_outputs.zip'))


def main():
    destination=ROOT/'Results/Submission';destination.mkdir(exist_ok=True)
    files=submission_files()
    required=['English/train.py','Telugu/train.py','English/preprocess.py','Telugu/preprocess.py',
              'English/inference.py','Telugu/inference.py','English/config.yaml','Telugu/config.yaml',
              'English/checkpoints/final.pth.tar','Telugu/checkpoints/final.pth.tar',
              'hifigan/generator_universal.pth.tar','Report/Technical_Report.pdf','README.md','requirements.txt']
    paths={p.relative_to(ROOT).as_posix() for p in files}
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
    manifest={'created_utc':datetime.now(timezone.utc).isoformat(),'release_tag':'assignment-v1',
              'archive_layout':'project/','audios':10,'variance_plots':30,
              'exclusions':['raw datasets','credentials','account runtime state','environments','historical optimizer checkpoints','unrelated upstream demos'],
              'files':{str(p.relative_to(ROOT)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in files}}
    path=destination/'package-manifest.json';path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    files.append(path)
    archive=destination/'fastspeech2-assignment-v1.zip'
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
