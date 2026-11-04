#!/usr/bin/env python3
"""Isolate HiFi-GAN: saved ground-truth mel to waveform, without FastSpeech2."""
import hashlib
import json
import os
from pathlib import Path
import sys
import argparse

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'Telugu'));os.chdir(ROOT)
os.environ.setdefault('MPLBACKEND','Agg')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--feature-version')
    args=parser.parse_args()
    import numpy as np
    import soundfile as sf
    import torch
    import librosa
    import tgt
    import whisper
    import hifigan
    from matplotlib import pyplot as plt
    from setup.review_initial_speech import normalized,distance
    torch.set_num_threads(4)
    output=ROOT/'Results/VocoderComparison'
    if args.feature_version:output=output/args.feature_version
    output.mkdir(parents=True,exist_ok=True)
    controls=json.loads((ROOT/'English/artifacts/InitialSpeechReview/review.json').read_text())['samples']
    config=json.loads((ROOT/'Telugu/hifigan/config.json').read_text())
    vocoder=hifigan.Generator(hifigan.AttrDict(config))
    weights=ROOT/'Telugu/hifigan/generator_universal.pth.tar'
    vocoder.load_state_dict(torch.load(weights,map_location='cpu',weights_only=False)['generator'])
    vocoder.eval();vocoder.remove_weight_norm()
    recognizer=whisper.load_model('small',device='cpu',download_root=str(ROOT/'data/evaluation/whisper'))
    basis=librosa.filters.mel(22050,1024,80,0,8000)

    def mel_of(wav):
        # Equivalent analysis settings to TacotronSTFT, avoiding its CUDA-only call.
        spectrum=librosa.stft(wav,n_fft=1024,hop_length=256,win_length=1024,
                             window='hann',center=True,pad_mode='reflect')
        return np.log(np.maximum(basis@np.abs(spectrum),1e-5))

    report={'method':'Natural aligned audio versus universal HiFi-GAN reconstruction from saved and freshly extracted full-length ground-truth mels',
            'fastspeech2_used_in_reconstruction':False,'cloud_compute_used':False,
            'feature_version':args.feature_version or 'legacy',
            'vocoder_sha256':hashlib.sha256(weights.read_bytes()).hexdigest(),
            'asr':'Local Whisper-small, forced source language, no transcript prompts',
            'human_listening_performed':False,'languages':{}}
    for lang,code in [('english','en'),('telugu','te')]:
        control=controls[lang+'_natural_control']
        sample=Path(control['path']).stem
        feature_dir=ROOT/'data/features'
        if args.feature_version:feature_dir=feature_dir/args.feature_version
        feature_dir=feature_dir/lang
        metadata=next(line.split('|') for line in (feature_dir/'val.txt').read_text().splitlines()
                      if line.startswith(sample+'|'))
        speaker=metadata[1]
        grid=tgt.io.read_textgrid(str(feature_dir/'TextGrid'/speaker/(sample+'.TextGrid')))
        intervals=[i for i in grid.get_tier_by_name('phones')._objects if i.text not in ['sil','sp','spn']]
        start,end=intervals[0].start_time,intervals[-1].end_time
        source=ROOT/'data/preprocessing'/lang/'corpus/validation'/speaker/(sample+'.wav')
        audio,rate=sf.read(source,dtype='float32');assert rate==22050
        natural=audio[int(start*rate):int(end*rate)]
        mel_path=feature_dir/'mel'/f'{speaker}-mel-{sample}.npy'
        target=np.load(mel_path).T
        recomputed=mel_of(natural)
        n=min(target.shape[1],recomputed.shape[1])
        extraction_mae=float(np.abs(target[:,:n]-recomputed[:,:n]).mean())
        assert extraction_mae<.02, ('Ground-truth extraction mismatch',extraction_mae)
        with torch.inference_mode():
            reconstructed=vocoder(torch.from_numpy(target.copy()).unsqueeze(0)).squeeze().numpy()
            full_reconstructed=vocoder(torch.from_numpy(recomputed.copy()).unsqueeze(0)).squeeze().numpy()
        assert np.isfinite(reconstructed).all()
        directory=output/lang;directory.mkdir(exist_ok=True)
        info={'text':control['expected_text'],'sample_id':sample,'speaker_label':speaker,
              'ground_truth_mel_path':str(mel_path.relative_to(ROOT)),
              'ground_truth_mel_sha256':hashlib.sha256(mel_path.read_bytes()).hexdigest(),
              'analysis_settings_match_mae':extraction_mae,'trim_seconds':[start,end],
              'samples':{}}
        info['saved_mel_frames']=target.shape[1]
        info['fresh_full_mel_frames']=recomputed.shape[1]
        info['saved_mel_missing_tail_seconds']=max(0,len(natural)/rate-len(reconstructed)/rate)
        for name,wav in [('natural',natural),('reconstructed',reconstructed),
                         ('full_mel_reconstructed',full_reconstructed)]:
            path=directory/(name+'.wav');sf.write(path,wav,rate,subtype='PCM_16')
            result=recognizer.transcribe(str(path),language=code,task='transcribe',fp16=False,
                                        temperature=0,condition_on_previous_text=False,verbose=None)
            ref,hyp=normalized(info['text']).split(),normalized(result['text']).split()
            info['samples'][name]={'path':str(path.relative_to(ROOT)),
                'duration_seconds':len(wav)/rate,'peak':float(np.abs(wav).max()),
                'full_scale_sample_fraction':float(np.mean(np.abs(wav)>=.999)),
                'rms_dbfs':float(20*np.log10(max(float(np.sqrt(np.mean(wav**2))),1e-12))),
                'asr_text':result['text'],'normalized_word_error_rate':distance(ref,hyp)/max(1,len(ref))}
        recmel=mel_of(reconstructed)
        n=min(target.shape[1],recmel.shape[1])
        info['reconstructed_log_mel_mae']=float(np.abs(target[:,:n]-recmel[:,:n]).mean())
        full_recmel=mel_of(full_reconstructed)
        nfull=min(recomputed.shape[1],full_recmel.shape[1])
        info['full_reconstruction_log_mel_mae']=float(np.abs(recomputed[:,:nfull]-full_recmel[:,:nfull]).mean())
        info['metric_note']='Log-mel error measures spectral reconstruction, not perceptual quality or pronunciation.'
        fig,axes=plt.subplots(4,1,figsize=(10,11),layout='constrained')
        vmin,vmax=np.percentile(target,[5,99])
        for ax,arr,title in zip(axes,[recomputed,target,recmel,full_recmel],
                ['Fresh full-length natural mel','Saved training mel',
                 'Reconstruction from saved mel','Reconstruction from fresh full-length mel']):
            ax.imshow(arr,origin='lower',aspect='auto',extent=[0,arr.shape[1]*256/22050,0,80],
                      vmin=vmin,vmax=vmax,cmap='magma')
            ax.set(title=title,xlabel='Time (seconds)',ylabel='Mel bin')
        fig.suptitle(lang.title()+' | Ground-truth-mel vocoder comparison')
        fig.savefig(directory/'spectrogram_comparison.png',dpi=170);plt.close(fig)
        report['languages'][lang]=info
        (output/'comparison.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
        print(json.dumps({lang:info},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
