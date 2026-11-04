#!/usr/bin/env python3
"""Local hiss-cleanup candidates; preserve the model output and source audio."""
import hashlib
import argparse
import json
import os
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'English'))
os.chdir(ROOT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--denoise-strength', choices=['gentle', 'moderate'], default='gentle')
    args = parser.parse_args()
    import numpy as np
    import soundfile as sf
    import torch
    import whisper
    import hifigan
    from scipy import signal, ndimage
    from setup.review_initial_speech import normalized, distance
    torch.set_num_threads(4)
    root = ROOT / 'English/artifacts/Training/gap-fixed-v2-20261103'
    source = root / 'english_diagnostics/preview_best_duration_1.3.wav'
    mel_path = root / 'english_diagnostics/preview_best_duration_1.3-mel.npy'
    out = root / 'english_hiss_comparison'
    out.mkdir(parents=True, exist_ok=True)
    x, sr = sf.read(source, dtype='float64')
    assert sr == 22050 and np.isfinite(x).all()
    rms = lambda y: float(np.sqrt(np.mean(y * y)))
    _, _, z = signal.stft(x, fs=sr, nperseg=512, noverlap=384)
    power = np.abs(z) ** 2
    quiet = np.argsort(power.sum(axis=0))[:max(3, power.shape[1] // 10)]
    noise = np.median(power[:, quiet], axis=1, keepdims=True)
    # The floor comes from quiet frames, not verified noise-only audio.
    # Bound attenuation to 6 dB (gentle) or 12 dB (moderate).
    amount, floor = (.75, .5) if args.denoise_strength == 'gentle' else (1.25, .25)
    gain = np.clip(1 - amount * noise / (power + 1e-16), floor, 1)
    gain = ndimage.gaussian_filter(gain, sigma=(1, 1))
    _, clean = signal.istft(z * gain, fs=sr, nperseg=512, noverlap=384)
    clean = clean[:len(x)]
    clean *= rms(x) / max(rms(clean), 1e-12)
    assert np.max(np.abs(clean)) < .99
    cleaned = out / f'universal_{args.denoise_strength}_denoise.wav'
    sf.write(cleaned, clean, sr, subtype='PCM_16')
    weights = ROOT / 'English/hifigan/generator_LJSpeech.pth.tar'
    if not weights.exists():
        with zipfile.ZipFile(str(weights) + '.zip') as archive:
            weights.write_bytes(archive.read(weights.name))
    vocoder = hifigan.Generator(hifigan.AttrDict(json.loads((ROOT / 'English/hifigan/config.json').read_text())))
    vocoder.load_state_dict(torch.load(weights, map_location='cpu', weights_only=False)['generator'])
    vocoder.eval(); vocoder.remove_weight_norm()
    mel = np.load(mel_path)
    with torch.inference_mode():
        alternate = vocoder(torch.from_numpy(mel)[None].transpose(1, 2)).squeeze().numpy()
    assert len(alternate) == len(x) and np.isfinite(alternate).all()
    alternate *= rms(x) / max(rms(alternate), 1e-12)
    assert np.max(np.abs(alternate)) < .99
    alternate_path = out / 'ljspeech_vocoder_rms_matched.wav'
    sf.write(alternate_path, alternate, sr, subtype='PCM_16')
    model = whisper.load_model('small', device='cpu', download_root=str(ROOT / 'data/evaluation/whisper'))
    text = 'The meeting will start on time.'
    report = {'user_report': 'hissing or static', 'human_listening_performed': False,
              'checkpoint_step': 12000, 'duration_control': 1.3,
              'denoise_strength': args.denoise_strength,
              'maximum_spectral_attenuation_db': float(-20*np.log10(floor)),
              'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
              'mel_sha256': hashlib.sha256(mel_path.read_bytes()).hexdigest(),
              'alternate_vocoder_sha256': hashlib.sha256(weights.read_bytes()).hexdigest(),
              'method': 'Mild STFT spectral attenuation, or alternate LJSpeech HiFi-GAN on identical mel; candidate RMS matched to original',
              'limitations': ['Quietest frames may contain speech; denoising may alter consonants',
                              'ASR does not measure hiss or perceptual quality',
                              'LJSpeech vocoder is an alternative, not a confirmed improvement for this voice',
                              'No training settings or original audio modified'],
              'samples': {}}
    for name, path in [('original_universal', source), (args.denoise_strength+'_denoise', cleaned), ('alternate_vocoder', alternate_path)]:
        y, rate = sf.read(path)
        transcript = model.transcribe(str(path), language='en', fp16=False, temperature=0,
                                      condition_on_previous_text=False, verbose=None)['text']
        ref, hyp = normalized(text).split(), normalized(transcript).split()
        entry = {'path': str(path), 'text': text, 'asr_text': transcript,
                 'word_errors': distance(ref, hyp), 'reference_words': len(ref),
                 'seconds': len(y)/rate, 'rms_dbfs': float(20*np.log10(max(rms(y), 1e-12))),
                 'peak': float(np.abs(y).max()), 'clipping_fraction': float(np.mean(np.abs(y) >= .999)),
                 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        report['samples'][name] = entry
        print(json.dumps({'sample': name, **entry}), flush=True)
    report_name = 'comparison.json' if args.denoise_strength == 'gentle' else 'comparison_moderate.json'
    (out / report_name).write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
