#!/usr/bin/env python3
"""CPU inference with frozen weights and exported model variance predictions."""
import argparse
import copy
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
os.environ.setdefault('MPLBACKEND','Agg')


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def test_cases():
    return json.loads((ROOT/'setup/assignment_test_sentences.json').read_text())['samples']


class InferenceEngine:
    def __init__(self,language,*,config=None,parent_english=False):
        os.chdir(ROOT)
        import numpy as np
        import torch,yaml
        import hifigan
        from text.symbols import symbols
        from model import FastSpeech2
        from model.learned_pause import LearnedPauseFastSpeech2
        from setup.final_text_frontend import TextFrontend
        self.language=language
        self.config=copy.deepcopy(config) if config is not None else yaml.safe_load((ROOT/language.title()/'config.yaml').read_text())
        c=self.config
        assert c['language']==language and c['device']=='cpu'
        assert hashlib.sha256(json.dumps(symbols).encode()).hexdigest()==c['trained_symbols_sha256']
        self.resources=ROOT/c['resources']
        for name,sha in c['resource_sha256'].items():
            assert digest(self.resources/name)==sha, 'Frozen inference resource changed: '+name
        path=ROOT/c['checkpoint']['path'];raw=path.read_bytes()
        assert len(raw)==c['checkpoint']['bytes'] and hashlib.sha256(raw).hexdigest()==c['checkpoint']['sha256']
        ckpt=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=False)
        assert ckpt['language']==language and ckpt['step']==c['checkpoint']['step']
        pre,model_config,_=copy.deepcopy(ckpt['configs'])
        # Only location changes. Frozen stats/speaker files are byte-identical
        # to training; raw datasets are not required to instantiate the model.
        pre['path']['preprocessed_path']=str(self.resources)
        self.learned=c['checkpoint']['model_type']=='learned_pause_v1'
        if parent_english and (language!='english' or self.learned):
            raise ValueError('Parent inputs require the plain English parent checkpoint')
        self.parent_english=parent_english
        model_class=LearnedPauseFastSpeech2 if self.learned else FastSpeech2
        torch.set_num_threads(4)
        self.model=model_class(pre,model_config).eval()
        self.model.load_state_dict(ckpt['model'],strict=True)
        self.stats=json.loads((self.resources/'stats.json').read_text())
        self.speakers=json.loads((self.resources/'speakers.json').read_text())
        assert c['speaker'] in self.speakers
        self.frontend=TextFrontend(language,self.resources)
        self.ipa={v:k for k,v in json.loads((self.resources/'ipa-to-phone.json').read_text()).items()} if language=='telugu' else {}
        v=c['vocoder'];weights=ROOT/v['weights']
        assert digest(weights)==v['sha256'] and digest(ROOT/v['config'])==v['config_sha256']
        self.vocoder=hifigan.Generator(hifigan.AttrDict(json.loads((ROOT/v['config']).read_text())))
        self.vocoder.load_state_dict(torch.load(weights,map_location='cpu',weights_only=False)['generator'])
        self.vocoder.eval();self.vocoder.remove_weight_norm()

    def synthesize(self,text,sample_id=None,required=False,output_dir=None,plots=True):
        import torch,numpy as np,soundfile as sf
        c=self.config
        encoded=self.frontend.encode(text)
        if self.parent_english:
            from setup.final_text_frontend import parent_english_inputs
            encoded=parent_english_inputs(encoded)
        if sample_id is None:
            sample_id='custom-'+hashlib.sha256((text+'|'+c['checkpoint']['sha256']).encode()).hexdigest()[:12]
        if not re.fullmatch(r'[A-Za-z0-9_-]+',sample_id):raise ValueError('Invalid sample ID')
        out=(ROOT/output_dir if output_dir else ROOT/'Results'/self.language.title()/
             ('test_generated_audios' if required else 'custom_generated_audios')).resolve()
        if not out.is_relative_to(ROOT/'Results'):raise ValueError('Output must stay within assignment Results/')
        out.mkdir(parents=True,exist_ok=True)
        info_path=out/(sample_id+'.json')
        if info_path.exists():
            old=json.loads(info_path.read_text())
            if old['source_text']!=text or old['checkpoint_sha256']!=c['checkpoint']['sha256']:
                raise ValueError('Existing sample has different source/weights; use a separate output directory')
        sequence=encoded['sequence'];n=len(sequence)
        inputs=(torch.tensor([self.speakers[c['speaker']]]),torch.tensor([sequence],dtype=torch.long),torch.tensor([n]),n)
        with torch.inference_mode():
            if self.learned:
                pred,aux=self.model(*inputs,boundary_types=torch.tensor([encoded['boundary_types']]),return_aux=True)
            else:
                pred=self.model(*inputs);aux=None
            frames=int(pred[9][0]);dur=pred[5][0].cpu().numpy().astype(np.int64)
            if not 0<frames<=6000 or int(dur.sum())!=frames:
                raise ValueError('Invalid predicted acoustic length')
            mel=pred[1][:,:frames]
            wave=self.vocoder(mel.transpose(1,2)).squeeze().cpu().numpy()
        pitch=pred[2][0].detach().cpu().numpy();energy=pred[3][0].detach().cpu().numpy()
        assert len(pitch)==len(energy)==len(dur)==len(encoded['phones'])
        assert len(wave)==frames*c['hop_length'] and np.isfinite(wave).all() and np.max(np.abs(wave))>0
        f0=pitch*self.stats['pitch'][3]+self.stats['pitch'][2]
        physical_energy=energy*self.stats['energy'][3]+self.stats['energy'][2]
        assert np.isfinite(f0).all() and np.isfinite(physical_energy).all()
        audio_path=out/(sample_id+'.wav');sf.write(audio_path,wave,c['sampling_rate'],subtype='PCM_16')
        (out/(sample_id+'.txt')).write_text(text+'\n')
        arrays_path=out/(sample_id+'_predictions.npz')
        arrays=dict(mel=mel[0].numpy(),duration_frames=dur,duration_seconds=dur*c['hop_length']/c['sampling_rate'],
                    normalized_pitch=pitch,predicted_f0_hz=f0,normalized_energy=energy,
                    predicted_spectral_energy=physical_energy,boundary_types=np.array(encoded['boundary_types']))
        if aux:
            arrays['pause_probability']=torch.sigmoid(aux['pause_logits'])[0].cpu().numpy()
            arrays['pause_log_duration']=aux['pause_log_duration'][0].cpu().numpy()
        np.savez(arrays_path,**arrays)
        word_for_phone=['']*n
        for span in encoded['word_spans']:
            for i in range(span['start'],span['end']):word_for_phone[i]=span['word']
        csv_path=out/(sample_id+'_variance.csv')
        with csv_path.open('w',newline='') as f:
            writer=csv.writer(f);writer.writerow(['phone_index','phone','display_phone','word','boundary_type','used_in_audio','duration_frames','duration_seconds','predicted_f0_hz','predicted_spectral_energy','normalized_pitch','normalized_energy'])
            for i,phone in enumerate(encoded['phones']):
                writer.writerow([i,phone,self.ipa.get(phone,phone),word_for_phone[i],encoded['boundary_types'][i],bool(dur[i]),
                    int(dur[i]),float(dur[i]*c['hop_length']/c['sampling_rate']),float(f0[i]),float(physical_energy[i]),float(pitch[i]),float(energy[i])])
        image_paths=self.plot_variance(sample_id,encoded,dur,f0,physical_energy) if plots else {}
        info={**encoded,'id':sample_id,'language':self.language,'generated_utc':datetime.now(timezone.utc).isoformat(),
              'required_assignment_sample':required,'checkpoint_step':c['checkpoint']['step'],
              'checkpoint_sha256':c['checkpoint']['sha256'],'checkpoint_path':c['checkpoint']['path'],
              'model_type':c['checkpoint']['model_type'],'source_run':c['checkpoint']['source_run'],
              'speaker':c['speaker'],'speaker_labels_estimated':c['speaker_labels_estimated'],
              'vocoder_sha256':c['vocoder']['sha256'],'sampling_rate':c['sampling_rate'],'hop_length':c['hop_length'],
              'frames':frames,'seconds':len(wave)/c['sampling_rate'],
              'audio_path':str(audio_path.relative_to(ROOT)),'audio_sha256':digest(audio_path),
              'predictions_path':str(arrays_path.relative_to(ROOT)),'variance_csv':str(csv_path.relative_to(ROOT)),
              'plots':image_paths,'duration_control':1.0,'pitch_control':1.0,'energy_control':1.0,
              'denoising':False,'external_silence_samples':0,'waveform_edits':[],'sentence_splitting':False,
              'device':'cpu','cloud_resources_started':False,'additional_cloud_compute_cost_usd':0,
              'prediction_summary':{'zero_duration_phones':int((dur==0).sum()),
                  'negative_predicted_f0_values':int((f0<0).sum()),'negative_predicted_energy_values':int((physical_energy<0).sum()),
                  'peak_waveform':float(np.abs(wave).max()),'clipping_fraction':float(np.mean(np.abs(wave)>=1.0))},
              'quality_assessment':'Software checks only; native listening and per-sentence analysis pending',
              'variance_units':{'duration':'integer mel frames and seconds','pitch':'denormalized phoneme-level F0 conditioning in Hz; no voiced/unvoiced classifier',
                                'energy':'denormalized spectral L2 energy, linear units; not dB'},
              'plot_note':'Zero-duration tokens remain in raw exports but are omitted from pitch/energy curves because they generate no frames'}
        info_path.write_text(json.dumps(info,ensure_ascii=False,indent=2)+'\n')
        return info

    def plot_variance(self,sample_id,encoded,duration,f0,energy):
        import numpy as np
        from matplotlib import pyplot as plt
        labels=[self.ipa.get(p,p) for p in encoded['phones']];x=np.arange(len(labels))
        directory=ROOT/'Results'/self.language.title()/'variance_adaptor_plots'
        result={}
        for kind,values,ylabel in [('duration',duration*self.config['hop_length']/self.config['sampling_rate'],'Predicted duration (seconds)'),
                                   ('pitch',f0,'Predicted F0 conditioning (Hz)'),
                                   ('energy',energy,'Predicted spectral L2 energy (linear units)')]:
            fig,ax=plt.subplots(figsize=(max(12,min(36,.23*len(labels))),4.8))
            if kind=='duration':ax.bar(x,values,color=['#b97522' if p=='sp' else '#286090' for p in encoded['phones']],width=.8)
            else:
                curve=np.where(duration>0,values,np.nan)
                ax.plot(x,curve,color='#286090',marker='o',markersize=2.8,linewidth=1)
            for i,phone in enumerate(encoded['phones']):
                if phone=='sp':ax.axvspan(i-.45,i+.45,color='#d19a44',alpha=.12)
            ax.set_xticks(x);ax.set_xticklabels(labels,rotation=90,fontsize=7)
            ax.set_xlim(-.7,len(labels)-.3);ax.set_ylabel(ylabel);ax.set_xlabel('Input phoneme (sp = word boundary/pause slot)')
            ax.set_title(f'{sample_id} | {self.language.title()} | {kind.title()} prediction | checkpoint step {self.config["checkpoint"]["step"]}',fontsize=12)
            ax.grid(axis='y',alpha=.22);ax.set_axisbelow(True)
            fig.text(.01,.01,'Model predictions, unity controls; no target prosody or waveform edits. Zero-duration tokens have no pitch/energy curve point.',fontsize=8)
            fig.tight_layout(rect=(0,.055,1,1))
            path=directory/kind/(sample_id+'.png');path.parent.mkdir(parents=True,exist_ok=True)
            fig.savefig(path,dpi=180,facecolor='white');plt.close(fig)
            result[kind]=str(path.relative_to(ROOT))
        return result


def main(language=None):
    p=argparse.ArgumentParser(description=__doc__)
    if language is None:p.add_argument('--language',choices=['english','telugu'],required=True)
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--text');source.add_argument('--test-id')
    p.add_argument('--output-dir');p.add_argument('--no-plots',action='store_true')
    args=p.parse_args();language=language or args.language
    if args.test_id:
        matches=[s for s in test_cases() if s['id']==args.test_id and s['language']==language]
        if len(matches)!=1:p.error('Test ID does not belong to this language')
        text=matches[0]['text']
    else:text=args.text
    info=InferenceEngine(language).synthesize(text,sample_id=args.test_id,required=bool(args.test_id),
                                           output_dir=args.output_dir,plots=not args.no_plots)
    print(json.dumps({k:info[k] for k in ('id','language','source_text','audio_path','seconds','plots','checkpoint_step')},ensure_ascii=False))


if __name__=='__main__':main()
