#!/usr/bin/env python3
"""One model-predicted utterance: no inserted silence or duration adjustment."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'English'))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--text',required=True)
    parser.add_argument('--run-id',default='english-learned-pauses-pilot-20261104')
    parser.add_argument('--speaker',default='en_est_0155')
    parser.add_argument('--checkpoint',choices=['best','latest'],default='best')
    args=parser.parse_args(argv)
    if not args.run_id.replace('-','').replace('_','').isalnum():parser.error('Invalid run ID')
    os.chdir(ROOT)
    import torch,numpy as np,soundfile as sf
    import hifigan
    from model.learned_pause import LearnedPauseFastSpeech2
    from setup.learned_pause_frontend import words_and_types,encode_words
    from text import text_to_sequence
    torch.set_num_threads(4)
    base=ROOT/'output/runs'/args.run_id/'english'
    metadata=json.loads((base/('best.json' if args.checkpoint=='best' else 'checkpoint.json')).read_text())
    raw=(base/(args.checkpoint+'.pth.tar')).read_bytes();assert hashlib.sha256(raw).hexdigest()==metadata['sha256']
    checkpoint=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=False)
    assert checkpoint['model_type']=='learned_pause_v1'
    pre,config,_=checkpoint['configs'];model=LearnedPauseFastSpeech2(pre,config).eval();model.load_state_dict(checkpoint['model'],strict=True)
    speakers=json.loads((ROOT/pre['path']['preprocessed_path']/'speakers.json').read_text());assert args.speaker in speakers
    lexicon={}
    for filename in ['data/preprocessing/english/lexicon.txt','English/lexicon/librispeech-lexicon.txt']:
        for line in (ROOT/filename).read_text().splitlines():
            parts=line.split()
            if parts:lexicon.setdefault(parts[0].lower(),parts[1:])
    words,types=words_and_types(args.text);phones,boundaries=encode_words(words,types,lexicon)
    seq=text_to_sequence('{'+' '.join(phones)+'}',[]);assert len(seq)==len(phones)
    weights=ROOT/'English/hifigan/generator_universal.pth.tar'
    vocoder=hifigan.Generator(hifigan.AttrDict(json.loads((ROOT/'English/hifigan/config.json').read_text())))
    vocoder.load_state_dict(torch.load(weights,map_location='cpu',weights_only=False)['generator']);vocoder.eval();vocoder.remove_weight_norm()
    with torch.inference_mode():
        pred,aux=model(torch.tensor([speakers[args.speaker]]),torch.tensor([seq]),torch.tensor([len(seq)]),len(seq),
                       boundary_types=torch.tensor([boundaries]),return_aux=True)
        mel=pred[1];wave=vocoder(mel.transpose(1,2)).squeeze().numpy()
    assert np.isfinite(wave).all() and len(wave)==int(pred[5].sum())*256
    sid='custom-'+hashlib.sha256((args.text+'|'+args.speaker).encode()).hexdigest()[:12]
    out=ROOT/'English/artifacts/Training'/args.run_id/'learned_english_samples'/(args.checkpoint+'-'+str(metadata['step']))/sid;out.mkdir(parents=True,exist_ok=True)
    path=out/'preview.wav';sf.write(path,wave,22050,subtype='PCM_16')
    np.savez(out/'predictions.npz',mel=mel[0].numpy(),duration=pred[5][0].numpy(),
             boundary_types=np.array(boundaries),pause_probability=torch.sigmoid(aux['pause_logits'])[0].numpy(),
             normalized_pitch=pred[2][0].numpy(),normalized_energy=pred[3][0].numpy())
    info={'text':args.text,'normalized_text':' '.join(words),'phones':phones,'boundary_types':boundaries,
          'checkpoint_step':metadata['step'],'checkpoint_sha256':metadata['sha256'],'run_id':args.run_id,
          'speaker_label':args.speaker,'speaker_label_estimated':True,'audio_path':str(path.relative_to(ROOT)),
          'sampling_rate':22050,'seconds':len(wave)/22050,'duration_control':1.,'external_silence_samples':0,
          'waveform_edits':[],'sentence_splitting':False,'denoising':False,
          'predicted_pause_frames':[int(pred[5][0,i]) for i,b in enumerate(boundaries) if b],
          'method':'One full model inference; model predicts pause presence, duration, pitch, energy and acoustic frames'}
    (out/'preview.json').write_text(json.dumps(info,ensure_ascii=False,indent=2)+'\n');print(json.dumps(info,ensure_ascii=False))


if __name__=='__main__':main()
