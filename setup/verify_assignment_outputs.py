#!/usr/bin/env python3
"""Verify final WAV/data/plot contracts and build a local review page and ZIP."""
from datetime import datetime,timezone
import csv
import hashlib
import html
import json
import os
from pathlib import Path
import sys
import zipfile

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    import numpy as np,soundfile as sf,yaml
    from PIL import Image,ImageOps,ImageDraw
    out=ROOT/'Results/Assignment';out.mkdir(exist_ok=True)
    cases=json.loads((ROOT/'setup/assignment_test_sentences.json').read_text())['samples']
    expected={s['id']:s for s in cases};assert len(expected)==10
    verification={'status':'passed','verified_utc':datetime.now(timezone.utc).isoformat(),
                  'scope':'File/shape/unit/provenance checks; not perceptual quality assessment',
                  'samples':[],'checkpoints':{},'source_sha256':{}}
    for pattern in ('setup/final_*frontend.py','setup/final_inference.py','setup/generate_assignment_outputs.py',
                    'setup/prepare_final_inference.py','setup/verify_assignment_outputs.py',
                    'English/inference.py','Telugu/inference.py',
                    'setup/assignment_test_sentences.json','model/learned_pause.py','model/fastspeech2.py',
                    'transformer/Models.py','text/symbols.py','text/assignment_phones.json'):
        for p in ROOT.glob(pattern):verification['source_sha256'][str(p.relative_to(ROOT))]=sha(p)
    infos=[]
    for lang in ('english','telugu'):
        config=yaml.safe_load((ROOT/lang.title()/'config.yaml').read_text())
        cp=config['checkpoint'];assert sha(ROOT/cp['path'])==cp['sha256']
        verification['checkpoints'][lang]=cp
        directory=ROOT/'Results'/lang.title()/'test_generated_audios'
        assert {p.stem for p in directory.glob('*.wav')}=={k for k,v in expected.items() if v['language']==lang}
        for case in [s for s in cases if s['language']==lang]:
            sid=case['id'];info=json.loads((directory/(sid+'.json')).read_text());infos.append(info)
            assert info['source_text']==case['text']==(directory/(sid+'.txt')).read_text().removesuffix('\n')
            assert info['checkpoint_sha256']==cp['sha256'] and info['checkpoint_step']==cp['step']
            assert info['external_silence_samples']==0 and not info['denoising'] and not info['sentence_splitting']
            assert info['duration_control']==info['pitch_control']==info['energy_control']==1.
            wav=ROOT/info['audio_path'];audio,sr=sf.read(wav)
            assert sha(wav)==info['audio_sha256'] and sr==22050 and sf.info(wav).subtype=='PCM_16'
            assert audio.ndim==1 and np.isfinite(audio).all() and np.max(np.abs(audio))>0
            a=np.load(ROOT/info['predictions_path']);d=a['duration_frames'];n=len(info['phones'])
            assert len(d)==n==len(info['sequence'])==len(a['normalized_pitch'])==len(a['normalized_energy'])
            assert (d>=0).all() and len(audio)==int(d.sum())*256==info['frames']*256
            assert a['mel'].shape==(int(d.sum()),80) and all(np.isfinite(a[k]).all() for k in a.files)
            stats=json.loads((ROOT/config['resources']/'stats.json').read_text())
            np.testing.assert_allclose(a['predicted_f0_hz'],a['normalized_pitch']*stats['pitch'][3]+stats['pitch'][2],rtol=1e-6)
            np.testing.assert_allclose(a['predicted_spectral_energy'],a['normalized_energy']*stats['energy'][3]+stats['energy'][2],rtol=1e-6)
            with (ROOT/info['variance_csv']).open() as f:assert len(list(csv.DictReader(f)))==n
            assert set(info['plots'])=={'duration','pitch','energy'}
            images={}
            for kind,path in info['plots'].items():
                with Image.open(ROOT/path) as im:
                    assert im.format=='PNG' and im.width>=1000 and im.height>=500
                    assert np.asarray(im.convert('RGB')).std()>5
                    images[kind]={'path':path,'sha256':sha(ROOT/path),'pixels':[im.width,im.height]}
            verification['samples'].append({'id':sid,'language':lang,'seconds':len(audio)/sr,
                'audio_sha256':sha(wav),'phonemes':n,'duration_frames':int(d.sum()),
                'zero_duration_phones':int((d==0).sum()),'plots':images,
                'decoded_peak':float(np.abs(audio).max()),'decoded_rms':float(np.sqrt(np.mean(audio**2)))})
    verification['audio_count']=len(infos);verification['variance_plot_count']=sum(len(i['plots']) for i in infos)
    if (out/'cli-verification.json').exists():
        verification['cli_smoke']=json.loads((out/'cli-verification.json').read_text())
    assert verification['audio_count']==10 and verification['variance_plot_count']==30
    (out/'verification.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2)+'\n')
    # Contact sheet is for layout inspection; full-resolution originals remain separate.
    sheet=Image.new('RGB',(2160,10*250),(238,241,245));draw=ImageDraw.Draw(sheet)
    for row,info in enumerate(infos):
        for col,kind in enumerate(('duration','pitch','energy')):
            with Image.open(ROOT/info['plots'][kind]) as im:
                thumb=ImageOps.contain(im.convert('RGB'),(710,220))
                sheet.paste(thumb,(col*720+(710-thumb.width)//2,row*250+25))
            draw.text((col*720+12,row*250+6),info['id']+' / '+kind,fill=(22,34,48))
    sheet.save(out/'variance_contact_sheet.png')
    def rel(path):return html.escape(os.path.relpath(ROOT/path,out),quote=True)
    cards=[]
    for info in infos:
        text=html.escape(info['source_text']);lang=info['language'];sid=info['id']
        images=''.join(f'<a href="{rel(path)}" target="_blank"><span>{kind.title()}</span><img src="{rel(path)}" alt="{sid} {kind} plot"></a>' for kind,path in info['plots'].items())
        cards.append(f'''<article data-language="{lang}"><h2>{sid} <small>{lang.title()} · {info['seconds']:.2f}s · checkpoint {info['checkpoint_step']}</small></h2><p lang="{'te' if lang=='telugu' else 'en'}">{text}</p><audio controls preload="none" src="{rel(info['audio_path'])}"></audio><p class="links"><a href="{rel(info['audio_path'])}" download>Download WAV</a> · <a href="{rel(str(Path(info['audio_path']).with_suffix('.txt')))}">Exact source text</a> · <a href="{rel(str(Path(info['audio_path']).with_suffix('.json')))}">Generation details</a></p><div class="plots">{images}</div></article>''')
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Assignment audio and variance plots</title><style>body{font:16px system-ui,sans-serif;background:#eef2f7;color:#17253a;max-width:1120px;margin:32px auto;padding:0 20px}h1{font-size:30px}header p{line-height:1.6}button{background:white;border:1px solid #b8c6d7;padding:9px 20px;border-radius:8px;cursor:pointer;margin:0 6px 12px 0}article{background:white;border:1px solid #d5deea;border-radius:14px;padding:22px;margin:18px 0}h2{margin-top:0}small{font-size:14px;font-weight:400;color:#54657b;margin-left:12px}article>p{font-size:19px;line-height:1.7}audio{width:100%}.links{font-size:14px!important}a{color:#195e9d}.plots{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.plots a{min-width:0}.plots span{display:block;font-size:14px;margin-bottom:8px}.plots img{width:100%;height:125px;object-fit:contain;border:1px solid #e0e6ee}@media(max-width:700px){.plots{grid-template-columns:1fr}small{display:block;margin:4px 0}.plots img{height:auto}}</style><header><h1>Assignment audio and variance plots</h1><p>Five English and five Telugu test samples. Exact source text is retained. Speech uses model-predicted prosody with no denoising, added silence or duration scaling. Click each plot to inspect the full-resolution figure.</p><p>Software verification passed. Listening quality and sentence-level interpretation still need review.</p><button onclick="filter('all')">All 10</button><button onclick="filter('english')">English</button><button onclick="filter('telugu')">Telugu</button></header>'''+''.join(cards)+'''<script>function filter(lang){document.querySelectorAll('article').forEach(c=>c.hidden=lang!=='all'&&c.dataset.language!==lang)}document.querySelectorAll('audio').forEach(a=>a.addEventListener('play',()=>document.querySelectorAll('audio').forEach(b=>{if(b!==a)b.pause()})))</script></html>'''
    (out/'index.html').write_text(page)
    files={out/'index.html',out/'verification.json',out/'generation.json',out/'variance_contact_sheet.png'}
    if (ROOT/'Report/Technical_Report.pdf').exists():files.add(ROOT/'Report/Technical_Report.pdf')
    for info in infos:
        wav=ROOT/info['audio_path'];files.update([wav,wav.with_suffix('.txt'),wav.with_suffix('.json'),ROOT/info['predictions_path'],ROOT/info['variance_csv']]);files.update(ROOT/p for p in info['plots'].values())
    files.update([ROOT/'setup/assignment_test_sentences.json',ROOT/'setup/requirements-inference.txt'])
    for path in (out/'checkpoint-selection.json',out/'cli-verification.json',
                 ROOT/'English/config.yaml',ROOT/'Telugu/config.yaml',
                 ROOT/'Report/Final_Inference_and_Test_Outputs.txt'):
        if path.exists():files.add(path)
    archive=out/'assignment_test_outputs.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for path in sorted(files):z.write(path,str(path.relative_to(ROOT)))
    print(json.dumps({'status':'passed','audios':10,'variance_plots':30,'review_page':str((out/'index.html').relative_to(ROOT)),
                      'archive':str(archive.relative_to(ROOT)),'archive_bytes':archive.stat().st_size,'checkpoints_verified':True}))


if __name__=='__main__':main()
