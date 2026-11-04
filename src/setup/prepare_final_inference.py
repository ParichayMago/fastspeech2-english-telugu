#!/usr/bin/env python3
"""Freeze verified local checkpoints and portable inference resources; no cloud."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
SELECTIONS={
 'english':dict(run='english-learned-pauses-pilot-20261104',sha256='74ce8b28183ec37f98a001bfcd5b484cb660a2eea2d1a780ef6dc8ea7e6a38bc',step=1000,speaker='en_est_0155',model_type='learned_pause_v1'),
 'telugu':dict(run='gap-fixed-v2-20261103',sha256='2268f94c05c0c90d9a4b85780971b40dd07297bc39a16271b375311d4b7fcfb8',step=18000,speaker='te_male',model_type='fastspeech2')}


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-nltk',action='store_true')
    parser.add_argument('--nltk-only',action='store_true',help='Prepare text resources for an already packaged checkout')
    args=parser.parse_args();os.chdir(ROOT)
    if args.nltk_only and args.skip_nltk:parser.error('--nltk-only and --skip-nltk conflict')
    import torch,yaml
    from text.symbols import symbols
    if not args.skip_nltk:
        directory=ROOT/'data/inference/nltk_data';directory.mkdir(parents=True,exist_ok=True)
        os.environ['NLTK_DATA']=str(directory)
        import nltk
        nltk.data.path.insert(0,str(directory))
        for resource in ('averaged_perceptron_tagger','averaged_perceptron_tagger_eng','cmudict'):
            assert nltk.download(resource,download_dir=str(directory),quiet=True),resource
    if args.nltk_only:
        print(json.dumps({'status':'NLTK resources ready','directory':'data/inference/nltk_data'}));return
    summary={}
    for language,selection in SELECTIONS.items():
        source=ROOT/'output/runs'/selection['run']/language
        original=source/'best.pth.tar';metadata=json.loads((source/'best.json').read_text())
        assert digest(original)==metadata['sha256']==selection['sha256']
        assert metadata['step']==selection['step']
        target=ROOT/language.title();resources=target/'resources';resources.mkdir(exist_ok=True)
        destination=target/'checkpoints/final.pth.tar';destination.parent.mkdir(exist_ok=True)
        if destination.exists() and digest(destination)!=selection['sha256']:
            raise ValueError('An existing frozen checkpoint differs; refuse overwrite')
        if not destination.exists():shutil.copyfile(original,destination)
        assert digest(destination)==selection['sha256']
        ckpt=torch.load(destination,map_location='cpu',weights_only=False)
        assert ckpt['language']==language and ckpt['step']==selection['step']
        feature=ROOT/ckpt['configs'][0]['path']['preprocessed_path']
        for name in ('stats.json','speakers.json'):
            shutil.copyfile(feature/name,resources/name)
        lexicon={}
        inputs=[ROOT/'data/preprocessing'/language/'lexicon.txt']
        if language=='english':inputs.append(ROOT/'src/lexicon/librispeech-lexicon.txt')
        for path in inputs:
            for line in path.read_text().splitlines():
                parts=line.split()
                if parts:lexicon.setdefault(parts[0].lower(),parts[1:])
        (resources/'lexicon.tsv').write_text(''.join(w+'\t'+' '.join(p)+'\n' for w,p in sorted(lexicon.items())))
        if language=='telugu':
            shutil.copyfile(ROOT/'data/preprocessing/telugu/ipa-to-phone.json',resources/'ipa-to-phone.json')
        vocoder=ROOT/'src/hifigan/generator_universal.pth.tar'
        config={'language':language,'checkpoint':{'path':str(destination.relative_to(ROOT)),
            'sha256':selection['sha256'],'bytes':destination.stat().st_size,
            'step':selection['step'],'model_type':selection['model_type'],
            'source_run':selection['run'],'validation_total_loss':metadata['validation_total_loss']},
            'speaker':selection['speaker'],'speaker_labels_estimated':language=='english',
            'resources':str(resources.relative_to(ROOT)),
            'resource_sha256':{p.name:digest(p) for p in sorted(resources.iterdir()) if p.is_file()},
            'trained_symbols_sha256':hashlib.sha256(json.dumps(symbols).encode()).hexdigest(),
            'vocoder':{'weights':str(vocoder.relative_to(ROOT)),'sha256':digest(vocoder),
                       'config':'src/hifigan/config.json','config_sha256':digest(ROOT/'src/hifigan/config.json')},
            'sampling_rate':22050,'hop_length':256,'device':'cpu',
            'duration_control':1.0,'pitch_control':1.0,'energy_control':1.0,
            'manual_silence_samples':0,'denoising':False,
            'test_manifest':'src/setup/assignment_test_sentences.json'}
        path=target/'config.yaml'
        path.write_text(yaml.safe_dump(config,sort_keys=False,allow_unicode=True))
        proof={'frozen_utc':datetime.now(timezone.utc).isoformat(),**config['checkpoint'],
               'source_checkpoint':str(original.relative_to(ROOT)),
               'selection_basis':'Lowest complete validation loss within the selected run; no cross-objective loss ranking',
               'quality_status':'Human evaluation on the exact assignment test set still required',
               'original_checkpoints_preserved':True}
        (target/'checkpoints/checkpoint.json').write_text(json.dumps(proof,indent=2)+'\n')
        summary[language]=proof
        print(json.dumps({'language':language,'frozen_checkpoint':str(destination.relative_to(ROOT)),
                          'step':selection['step'],'sha256':selection['sha256']}),flush=True)
    out=ROOT/'src/artifacts/Assignment';out.mkdir(exist_ok=True)
    (out/'checkpoint-selection.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':main()
