#!/usr/bin/env python3
"""Local matched-sentence comparison; no human listening or cloud requests."""
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import whisper

from review_initial_speech import normalized, distance

ROOT = Path(__file__).resolve().parent.parent
RUN = 'gap-fixed-v2-20261103'


def main():
    torch.set_num_threads(4)
    output = ROOT / 'Results/Training' / RUN / 'speech_comparison'
    output.mkdir(parents=True, exist_ok=True)
    generated = json.loads((output.parent / 'speech_samples/generation.json').read_text())
    model = whisper.load_model('small', device='cpu', download_root=str(ROOT / 'data/evaluation/whisper'))
    report = {'method': 'Matched-text signal measurements and unprompted Whisper-small English transcription',
              'human_listening_performed': False, 'cloud_compute_used': False,
              'limitations': ['One sentence per language; checkpoints have different training step counts',
                             'ASR errors do not measure naturalness or prove human intelligibility',
                             'Telugu ASR previously failed on natural controls; no Telugu WER judgment'],
              'languages': {}}
    for lang in ('english', 'telugu'):
        old_dir = ROOT / 'Results' / lang.title() / 'initial_generated_audios'
        old = json.loads((old_dir / 'preview.json').read_text())
        current = generated['samples'][lang]
        assert old['text'] == current['text']
        assert old['phones'] == current['phones']
        assert old['speaker_label'] == current['speaker_label']
        entries = {}
        for version, info, folder in [('previous', old, old_dir),
                                      ('corrected', current, (ROOT / current['audio_path']).parent)]:
            path = folder / 'preview.wav'
            audio, sr = sf.read(path, dtype='float32')
            assert audio.ndim == 1 and np.isfinite(audio).all()
            predictions = np.load(folder / 'predictions.npz')
            duration = predictions['duration'].tolist()
            entry = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                     'checkpoint_step': info['checkpoint_step'], 'seconds': len(audio) / sr,
                     'peak': float(np.abs(audio).max()),
                     'rms_dbfs': float(20*np.log10(max(float(np.sqrt(np.mean(audio**2))), 1e-12))),
                     'clipping_fraction': float(np.mean(np.abs(audio) >= .999)),
                     'predicted_phone_durations_frames': duration,
                     'zero_duration_phones': int(np.count_nonzero(np.asarray(duration) == 0))}
            if lang == 'english':
                result = model.transcribe(str(path), language='en', task='transcribe', fp16=False,
                                          temperature=0, condition_on_previous_text=False, verbose=None)
                ref, hyp = normalized(info['text']).split(), normalized(result['text']).split()
                entry['asr_text'] = result['text']
                entry['word_errors'] = distance(ref, hyp)
                entry['reference_words'] = len(ref)
                entry['word_error_rate'] = distance(ref, hyp) / len(ref)
            entries[version] = entry
            print(json.dumps({'language': lang, 'version': version, **entry}, ensure_ascii=False), flush=True)
        report['languages'][lang] = {'text': old['text'], 'speaker': old['speaker_label'],
                                     'speaker_estimated': lang == 'english', 'samples': entries}
    (output / 'comparison.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
