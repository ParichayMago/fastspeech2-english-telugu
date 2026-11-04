#!/usr/bin/env python3
"""Local objective signal/ASR review, not a human listening score."""
import hashlib
import json
from pathlib import Path
import re
import unicodedata

ROOT=Path(__file__).resolve().parents[2]


def normalized(text):
    return ' '.join(''.join(c if unicodedata.category(c)[0] in 'LMN' else ' '
                            for c in unicodedata.normalize('NFC',text).casefold()).split())


def distance(a,b):
    old=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        new=[i]
        for j,y in enumerate(b,1):
            new.append(min(new[-1]+1,old[j]+1,old[j-1]+(x!=y)))
        old=new
    return old[-1]


def main():
    import numpy as np
    import soundfile as sf
    import torch
    import whisper
    torch.set_num_threads(4)
    out=ROOT/'Results/InitialSpeechReview';out.mkdir(parents=True,exist_ok=True)
    report={'method':'signal measurements and local multilingual Whisper-small without text prompts',
            'human_listening_performed':False,'cloud_compute_used':False,
            'limitations':['ASR errors are not direct pronunciation or naturalness scores',
                           'Two short generated sentences do not represent all test cases',
                           'Natural reference controls test the recognizer on these languages'],
            'samples':{}}
    model=whisper.load_model('small',device='cpu',download_root=str(ROOT/'data/evaluation/whisper'))
    for lang,code in [('english','en'),('telugu','te')]:
        folder=ROOT/'Results'/lang.title()/'initial_generated_audios'
        preview=json.loads((folder/'preview.json').read_text())
        candidates=[json.loads(line) for line in (ROOT/'data/splits/v1'/lang/'validation-prepared.jsonl').read_text().splitlines()]
        control=min((r for r in candidates if len(r['text'].split())>=4),
                    key=lambda r:abs(r['duration_seconds']-3.5))
        for kind,path,expected in [('generated',folder/'preview.wav',preview['text']),
                                   ('natural_control',ROOT/control['prepared_audio_path'],control['text'])]:
            audio,rate=sf.read(path,dtype='float32')
            rms=float(np.sqrt(np.mean(audio**2)))
            result=model.transcribe(str(path),language=code,task='transcribe',fp16=False,
                                    temperature=0,condition_on_previous_text=False,verbose=None)
            ref,hyp=normalized(expected),normalized(result['text'])
            rw,hw=ref.split(),hyp.split()
            entry={'language':lang,'kind':kind,'path':str(path.relative_to(ROOT)),
                   'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                   'expected_text':expected,'asr_text':result['text'],
                   'duration_seconds':len(audio)/rate,'sampling_rate':rate,
                   'peak':float(np.abs(audio).max()),'rms_dbfs':20*np.log10(max(rms,1e-12)),
                   'full_scale_sample_fraction':float(np.mean(np.abs(audio)>=.999)),
                   'normalized_word_error_rate':distance(rw,hw)/max(1,len(rw)),
                   'normalized_character_error_rate':distance(ref.replace(' ',''),hyp.replace(' ',''))/max(1,len(ref.replace(' ',''))),
                   'segments':[{k:s[k] for k in ['start','end','text','avg_logprob','no_speech_prob']} for s in result['segments']]}
            report['samples'][lang+'_'+kind]=entry
            (out/'review.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
            print(json.dumps(entry,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
