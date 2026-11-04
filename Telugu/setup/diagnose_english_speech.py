#!/usr/bin/env python3
"""Local English synthesis diagnostics; never modifies training or checkpoints."""
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time
from collections import Counter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'Telugu'))
os.chdir(ROOT)
os.environ.setdefault('MPLBACKEND', 'Agg')


def main():
    import numpy as np
    import soundfile as sf
    import torch
    import whisper
    import hifigan
    from model import FastSpeech2
    from text import text_to_sequence
    from setup.review_initial_speech import normalized, distance
    from matplotlib import pyplot as plt
    torch.set_num_threads(4)
    torch.manual_seed(20261103)
    run = 'gap-fixed-v2-20261103'
    source = ROOT / 'output/runs' / run / 'english'
    out = ROOT / 'English/artifacts/Training' / run / 'english_diagnostics'
    out.mkdir(parents=True, exist_ok=True)
    preview_dir = out.parent / 'speech_samples/english/12000'
    preview = json.loads((preview_dir / 'preview.json').read_text())
    feature = ROOT / preview['feature_path']
    stats = json.loads((feature / 'stats.json').read_text())
    speakers = json.loads((feature / 'speakers.json').read_text())
    train = [r.split('|') for r in (feature / 'train.txt').read_text().splitlines()]
    val = [r.split('|') for r in (feature / 'val.txt').read_text().splitlines()]
    speaker = preview['speaker_label']
    counts = Counter(r[1] for r in train)
    def duration_of(row):
        return int(np.load(feature / 'duration' / f'{row[1]}-duration-{row[0]}.npy').sum()) * 256 / 22050
    candidates = [r for r in val if r[1] == speaker and len(r[3].split()) >= 5
                  and len(set(r[3].split())) == len(r[3].split())]
    row = min(candidates, key=lambda r: abs(duration_of(r) - 3.5))
    sid, _, phones, text = row
    targets = {k: np.load(feature / k / f'{speaker}-{k}-{sid}.npy') for k in ('mel', 'pitch', 'energy', 'duration')}
    report = {'human_listening_performed': False, 'training_modified': False,
              'sample_scope': 'One preview plus one held-out utterance, same estimated speaker',
              'speaker': speaker, 'speaker_labels_estimated': True,
              'speaker_training_clips': counts[speaker],
              'speaker_training_minutes': sum(duration_of(r) for r in train if r[1] == speaker) / 60,
              'speaker_cluster_count': len(counts), 'median_clips_per_cluster': float(np.median(list(counts.values()))),
              'held_out_id': sid, 'held_out_text': text,
              'checkpoints': {}, 'samples': {}}
    vocoder = hifigan.Generator(hifigan.AttrDict(json.loads((ROOT / 'Telugu/hifigan/config.json').read_text())))
    vocoder.load_state_dict(torch.load(ROOT / 'Telugu/hifigan/generator_universal.pth.tar', map_location='cpu', weights_only=False)['generator'])
    vocoder.eval(); vocoder.remove_weight_norm()
    recognizer = whisper.load_model('small', device='cpu', download_root=str(ROOT / 'data/evaluation/whisper'))

    def save_sample(name, mel, expected, pred=None, target=None):
        assert torch.isfinite(mel).all()
        with torch.inference_mode():
            wav = vocoder(mel.transpose(1, 2)).squeeze().numpy()
        path = out / (name + '.wav')
        sf.write(path, wav, 22050, subtype='PCM_16')
        transcription = recognizer.transcribe(str(path), language='en', temperature=0,
                          fp16=False, condition_on_previous_text=False, verbose=None)['text']
        ref, hyp = normalized(expected).split(), normalized(transcription).split()
        arr = mel[0].numpy()
        entry = {'path': str(path), 'text': expected, 'asr_text': transcription,
                 'word_errors': distance(ref, hyp), 'reference_words': len(ref),
                 'word_error_rate': distance(ref, hyp) / len(ref),
                 'seconds': len(wav) / 22050, 'frames': arr.shape[0],
                 'peak': float(np.abs(wav).max()), 'clipping_fraction': float(np.mean(np.abs(wav) >= .999)),
                 'rms_dbfs': float(20 * np.log10(max(float(np.sqrt(np.mean(wav**2))), 1e-12))),
                 'mean_temporal_mel_std': float(arr.std(axis=0).mean())}
        if pred is not None:
            d = pred[5][0].numpy()
            entry.update(zero_duration_phones=int((d == 0).sum()), phone_durations=d.tolist(),
                         predicted_pitch_hz=(pred[2][0].numpy() * stats['pitch'][3] + stats['pitch'][2]).tolist())
        if target is not None and arr.shape == target.shape:
            entry['target_log_mel_mae'] = float(np.abs(arr - target).mean())
            entry['target_mean_temporal_mel_std'] = float(target.std(axis=0).mean())
        report['samples'][name] = entry
        np.save(out / (name + '-mel.npy'), arr)
        (out / 'diagnosis.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({'name': name, **{k:v for k,v in entry.items() if k not in ('phone_durations', 'predicted_pitch_hz')} }), flush=True)

    save_sample('heldout_ground_truth_vocoder', torch.from_numpy(targets['mel'])[None], text)
    for selection in ('best', 'latest'):
        for attempt in range(10):
            key = 'best' if selection == 'best' else 'checkpoint'
            candidates = [json.loads((source / (key + '.json')).read_text())]
            supervisor = json.loads((out.parent / 'supervisor.json').read_text())
            verified = supervisor.get('verified_backup', {}).get('checkpoints', {}).get('english', {}).get(key)
            if verified:
                candidates.append(verified)
            raw = (source / (selection + '.pth.tar')).read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            matching = [m for m in candidates if digest == m['sha256'] and len(raw) == m['bytes']]
            if matching:
                meta = matching[0]
                break
            time.sleep(.5)
        else:
            raise RuntimeError('Backup changing: retry after supervisor finishes its copy')
        ckpt = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=False)
        del raw
        assert ckpt['step'] == meta['step']
        pre, mod, _ = ckpt['configs']
        net = FastSpeech2(pre, mod).eval(); net.load_state_dict(ckpt['model'])
        report['checkpoints'][selection] = meta
        del ckpt
        def predict(phone_string, **kwargs):
            seq = text_to_sequence(phone_string, [])
            assert len(seq) == len(phone_string.strip('{}').split())
            with torch.inference_mode():
                return net(torch.tensor([speakers[speaker]]), torch.tensor([seq]), torch.tensor([len(seq)]), len(seq), **kwargs)
        for control in (1.0, 1.3):
            pred = predict('{' + ' '.join(preview['phones']) + '}', d_control=control)
            save_sample(f'preview_{selection}_duration_{control}', pred[1], preview['text'], pred)
        modes = ('free', 'duration_target', 'all_variance_targets') if selection == 'best' else ('free',)
        for mode in modes:
            kwargs = {}
            if mode != 'free':
                kwargs['d_targets'] = torch.from_numpy(targets['duration'])[None]
                kwargs['mel_lens'] = torch.tensor([len(targets['mel'])])
                kwargs['max_mel_len'] = len(targets['mel'])
            if mode == 'all_variance_targets':
                kwargs.update(p_targets=torch.from_numpy(targets['pitch'])[None], e_targets=torch.from_numpy(targets['energy'])[None])
            pred = predict(phones, **kwargs)
            save_sample(f'heldout_{selection}_{mode}', pred[1], text, pred, targets['mel'])
        del net
    # Shared color range permits inspecting spectral smoothing in the controlled test.
    names = ['heldout_ground_truth_vocoder', 'heldout_best_all_variance_targets',
             'heldout_best_duration_target', 'heldout_best_free']
    fig, axes = plt.subplots(4, 1, figsize=(11, 11), layout='constrained')
    lo, hi = np.percentile(targets['mel'], [5, 99])
    for ax, name in zip(axes, names):
        arr = np.load(out / (name + '-mel.npy'))
        ax.imshow(arr.T, origin='lower', aspect='auto', vmin=lo, vmax=hi, cmap='magma',
                  extent=[0, len(arr)*256/22050, 0, 80])
        ax.set(title=name.replace('_', ' '), xlabel='Seconds', ylabel='Mel bin')
    fig.savefig(out / 'heldout_mel_diagnostics.png', dpi=150)
    plt.close(fig)
    (out / 'diagnosis.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
