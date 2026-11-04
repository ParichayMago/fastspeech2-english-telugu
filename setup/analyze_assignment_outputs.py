#!/usr/bin/env python3
"""Measured loss, waveform and model-variance analysis; no perceptual ratings."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def edit_counts(ref,hyp):
    # S/D/I are exact alignment counts, not a pronunciation or naturalness rating.
    d=[[(i+j,0,i,j) for j in range(len(hyp)+1)] for i in range(len(ref)+1)]
    for i in range(1,len(ref)+1):
        for j in range(1,len(hyp)+1):
            if ref[i-1]==hyp[j-1]:d[i][j]=d[i-1][j-1]
            else:
                a=d[i-1][j-1];b=d[i-1][j];c=d[i][j-1]
                d[i][j]=min((a[0]+1,a[1]+1,a[2],a[3]),
                             (b[0]+1,b[1],b[2]+1,b[3]),
                             (c[0]+1,c[1],c[2],c[3]+1))
    n,s,de,ins=d[-1][-1]
    return {'errors':n,'substitutions':s,'deletions':de,'insertions':ins,
            'reference_words':len(ref),'WER':n/max(1,len(ref))}


def analyze():
    import numpy as np,soundfile as sf
    from scipy.signal import welch
    out=ROOT/'Results/Assignment/analysis';out.mkdir(exist_ok=True)
    data={'created_utc':datetime.now(timezone.utc).isoformat(),'human_listening_performed':False,
          'cloud_compute_used':False,'training_runs':{},'samples':{},'source_files':{}}
    runs=[('gap-fixed-v2-20261103','english'),('gap-fixed-v2-20261103','telugu'),
          ('english-finetune-pilot-20261104','english'),('english-learned-pauses-pilot-20261104','english')]
    for run,language in runs:
        path=ROOT/'output/runs'/run/language/'metrics.jsonl'
        if not path.exists():
            path=ROOT/'Results/Training'/run/language/'metrics.jsonl'
        rows=[json.loads(x) for x in path.read_text().splitlines()]
        train={r['step']:r for r in rows if r['phase']=='train'}
        valid={r['step']:r for r in rows if r['phase']=='validation' and r.get('complete')}
        key=run+'/'+language;vv=[valid[k] for k in sorted(valid)];tt=[train[k] for k in sorted(train)]
        best=min(vv,key=lambda r:r['losses'][0]);last=vv[-1]
        cfg=json.loads((path.parent/'run-config.json').read_text())
        data['training_runs'][key]={'run_id':run,'language':language,'train':tt,'validation':vv,
             'best':best,'first':vv[0],'last':last,'settings':cfg['session'],'configs':cfg['configs'],
             'recorded_minibatch_size_range':[min(r['batch_size'] for r in tt),max(r['batch_size'] for r in tt)],
             'final_minus_best_validation_percent':(last['losses'][0]/best['losses'][0]-1)*100,
             'best_vs_first_validation_improvement_percent':(1-best['losses'][0]/vv[0]['losses'][0])*100}
        data['source_files'][str(path.relative_to(ROOT))]=sha(path)
    manifest=json.loads((ROOT/'setup/assignment_test_sentences.json').read_text())
    target_words={'E1':['antidisestablishmentarianism'],'E2a':['conduct'],'E2b':['conduct'],
                  'E3':['rhyth','m'],'E4':['andaroo','twelfth'],
                  'T1':['న్యూరోఫార్మకోలాజికల్'],'T2a':['పట్టు'],'T2b':['పట్టు'],
                  'T3':['అడ','విలో'],'T4':['పన్నెండవ']}
    for case in manifest['samples']:
        sid=case['id'];directory=ROOT/'Results'/case['language'].title()/'test_generated_audios'
        info=json.loads((directory/(sid+'.json')).read_text());a=np.load(ROOT/info['predictions_path'])
        wave,sr=sf.read(ROOT/info['audio_path']);d=a['duration_frames'];f0=a['predicted_f0_hz'];energy=a['predicted_spectral_energy']
        assert info['source_text']==case['text'] and sha(ROOT/info['audio_path'])==info['audio_sha256']
        active=d>0;lex=active & (np.array(info['boundary_types'])==0)
        freq,power=welch(wave,fs=sr,nperseg=1024)
        high=float(power[freq>=6000].sum()/max(power.sum(),1e-20))
        n=int(sr*.02);hop=int(sr*.01)
        rms=np.array([np.sqrt(np.mean(wave[i:i+n]**2)) for i in range(0,len(wave)-n+1,hop)])
        words=[]
        for span in info['word_spans']:
            i,j=span['start'],span['end'];wd=d[i:j];p=f0[i:j];e=energy[i:j]
            words.append({'word':span['word'],'start':i,'end':j,'duration_frames':int(wd.sum()),
              'seconds':float(wd.sum()*256/22050),'phones':info['phones'][i:j],
              'phone_durations_frames':wd.tolist(),'predicted_F0_hz':p.tolist(),
              'weighted_F0_hz':float(np.average(p,weights=wd)) if wd.sum() else None,
              'weighted_spectral_energy':float(np.average(e,weights=wd)) if wd.sum() else None})
        boundaries=[]
        for i,kind in enumerate(info['boundary_types']):
            if kind:
                preceding=max((w for w in info['word_spans'] if w['end']<=i),key=lambda w:w['end'])
                boundaries.append({'word':preceding['word'],'type':kind,'phone_index':i,
                                   'duration_frames':int(d[i]),'seconds':float(d[i]*256/22050),
                                   'terminal':i==len(d)-1})
        result={'id':sid,'language':case['language'],'source_text':case['text'],
          'normalized_words':info['normalized_words'],'duration_seconds':len(wave)/sr,
          'sampling_rate':sr,'mel_frames':int(d.sum()),'input_tokens':len(d),
          'lexical_phone_count':int((np.array(info['boundary_types'])==0).sum()),
          'spoken_word_count':len(info['normalized_words']),
          'spoken_tokens_per_minute':len(info['normalized_words'])/(len(wave)/sr)*60,
          'peak':float(np.abs(wave).max()),'rms_dbfs':float(20*np.log10(max(np.sqrt(np.mean(wave**2)),1e-12))),
          'clipped_fraction':float(np.mean(np.abs(wave)>=.999)),
          'high_frequency_power_fraction_6kHz_to_Nyquist':high,
          'low_level_20ms_frame_fraction_below_minus45dBFS':float(np.mean(rms<10**(-45/20))),
          'lexical_F0_hz_min_max':[float(f0[lex].min()),float(f0[lex].max())],
          'lexical_F0_hz_weighted_mean':float(np.average(f0[lex],weights=d[lex])),
          'lexical_spectral_energy_min_max':[float(energy[lex].min()),float(energy[lex].max())],
          'lexical_spectral_energy_weighted_mean':float(np.average(energy[lex],weights=d[lex])),
          'first_to_last_quintile_F0_hz_difference':float(np.mean(f0[lex][-max(1,int(lex.sum()/5)):])-np.mean(f0[lex][:max(1,int(lex.sum()/5))])),
          'total_predicted_pause_seconds':sum(x['seconds'] for x in boundaries),
          'predicted_pause_boundaries':boundaries,
          'target_word_stats':[w for w in words if w['word'] in target_words[sid]],
          'word_statistics':words,'normalization_changes':info['normalization_changes'],
          'pronunciations':info['pronunciations'],'audio_sha256':info['audio_sha256'],
          'interpretation_limits':['Pitch is model phoneme-level conditioning, not measured voiced waveform F0',
              'Spectral energy is a model prediction in linear L2 units, not dB',
              'High-frequency power can include legitimate consonants and is not a noise/SNR rating',
              'Spoken-token rate depends on normalization and is not comparable across languages']}
        data['samples'][sid]=result
        data['source_files'][str((directory/(sid+'.json')).relative_to(ROOT))]=sha(directory/(sid+'.json'))
        print(json.dumps({'id':sid,'seconds':round(result['duration_seconds'],3),'rms_dbfs':round(result['rms_dbfs'],2),
                          'F0_range':result['lexical_F0_hz_min_max'],'pause_seconds':result['total_predicted_pause_seconds']}),flush=True)
    data['data_splits']=json.loads((ROOT/'Results/DataSplits/summary.json').read_text())
    data['feature_rebuild']=json.loads((ROOT/'Results/Preprocessing/gap-fixed-v2/rebuild-summary.json').read_text())
    for name in ('english','telugu'):
        p=ROOT/'Results/DatasetAudit'/(name+'-audit.json');x=json.loads(p.read_text())
        data.setdefault('dataset_audits',{})[name]={k:x.get(k) for k in ('dataset','revision','decoded_audio_hours','candidate_audio_hours','sample_rates','counters','speaker_identity_note')}
    (out/'measured-analysis.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    return data


def asr(data):
    import torch,whisper
    from setup.final_text_frontend import english_spoken_form
    from setup.prepare_assignment_corpus import normalize_text
    torch.set_num_threads(4)
    out=ROOT/'Results/Assignment/analysis';path=out/'english-asr.json'
    report={'recognizer':'Whisper-small multilingual (local)','language':'en','prompt':None,
      'condition_on_previous_text':False,'temperature':0,'human_listening_performed':False,
      'limitations':['ASR disagreement is not a phonetic correctness or MOS score',
        'Only five synthetic test sentences and three natural controls; not a corpus benchmark',
        'Literal split words and code-mixing can inflate recognition disagreement'],
      'natural_controls':[],'generated':[]}
    if path.exists():report=json.loads(path.read_text())
    recognizer=whisper.load_model('small',device='cpu',download_root=str(ROOT/'data/evaluation/whisper'))
    def normalized(text):return normalize_text(english_spoken_form(text)[0],'english').split()
    def transcribe(key,wav,text,kind):
        digest=sha(wav)
        cached=next((x for x in report[kind] if x['id']==key and x['audio_sha256']==digest),None)
        if cached:return cached
        result=recognizer.transcribe(str(wav),language='en',temperature=0,fp16=False,
                 condition_on_previous_text=False,verbose=None)
        ref=normalized(text);hyp=normalized(result['text'])
        row={'id':key,'path':str(wav.relative_to(ROOT)),'audio_sha256':digest,'reference_text':text,
             'ASR_text':result['text'],'normalized_reference':ref,'normalized_hypothesis':hyp,
             **edit_counts(ref,hyp)}
        report[kind]=[x for x in report[kind] if x['id']!=key]+[row]
        path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps({'ASR_id':key,'text':result['text'],'WER':row['WER']}),flush=True)
        return row
    # Three held-out recordings from the same estimated English cluster serve
    # only as recognizer checks. They are not a calibrated quality benchmark.
    controls=[json.loads(l) for l in (ROOT/'data/splits/v1/english/validation-prepared.jsonl').read_text().splitlines()]
    controls=[x for x in controls if x['speaker_id']=='en_est_0155' and 5<=len(x['text'].split())<=12]
    controls=sorted(controls,key=lambda x:(abs(x['duration_seconds']-4),x['sample_id']))[:3]
    assert len(controls)==3
    for x in controls:transcribe(x['sample_id'],ROOT/x['prepared_audio_path'],x['text'],'natural_controls')
    for sid in ('E1','E2a','E2b','E3','E4'):
        x=data['samples'][sid];transcribe(sid,ROOT/'Results/English/test_generated_audios'/(sid+'.wav'),x['source_text'],'generated')
    for kind in ('natural_controls','generated'):
        rows=report[kind];report[kind+'_aggregate']={'errors':sum(x['errors'] for x in rows),
          'reference_words':sum(x['reference_words'] for x in rows),
          'WER':sum(x['errors'] for x in rows)/sum(x['reference_words'] for x in rows)}
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--asr',action='store_true');args=p.parse_args()
    data=analyze()
    if args.asr:asr(data)
