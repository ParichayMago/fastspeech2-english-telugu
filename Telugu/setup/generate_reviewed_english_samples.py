#!/usr/bin/env python3
"""Synthesize approved English source texts locally with matched speaker labels."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'Telugu'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit', type=int, default=4)
    parser.add_argument('--run-id', default='gap-fixed-v2-20261103')
    parser.add_argument('--reference-generation', help='Reuse exact source samples from an earlier generation report')
    parser.add_argument('--text', help='Generate one custom English sentence')
    parser.add_argument('--speaker', default='en_est_0155')
    parser.add_argument('--pacing', choices=['legacy','punctuation'],
                        help='Custom text defaults to punctuation-aware pacing; legacy reproduces prior previews')
    parser.add_argument('--duration-control', type=float,
                        help='Phoneme duration multiplier before rounding (default: 1.2 paced, 1.0 legacy)')
    args = parser.parse_args(argv)
    if not 1 <= args.limit <= 20:
        parser.error('Choose 1–20 samples')
    if not args.run_id.replace('-', '').replace('_', '').isalnum():
        parser.error('Run ID must be a simple directory name')
    if args.text is not None and (not args.text.strip() or args.reference_generation):
        parser.error('Provide nonempty custom text without --reference-generation')
    pacing = args.pacing or ('punctuation' if args.text is not None else 'legacy')
    duration_control = args.duration_control if args.duration_control is not None else (1.2 if pacing == 'punctuation' else 1.0)
    if not .5 <= duration_control <= 2:
        parser.error('Duration control must be between 0.5 and 2')
    os.chdir(ROOT)
    import numpy as np
    import soundfile as sf
    import torch
    import hifigan
    from model import FastSpeech2
    from text import text_to_sequence
    from setup.prepare_assignment_corpus import normalize_text
    from setup.english_inference_frontend import Chunk, plan_utterance, scaled_durations, assemble_audio
    torch.set_num_threads(4)
    torch.manual_seed(20261103)
    run = args.run_id
    base = ROOT / 'output/runs' / run / 'english'
    metadata = json.loads((base / 'best.json').read_text())
    path = base / 'best.pth.tar'
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == metadata['sha256'] and len(raw) == metadata['bytes']
    checkpoint = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=False)
    del raw
    assert checkpoint['step'] == metadata['step']
    pre, config, _ = checkpoint['configs']
    initialization = checkpoint.get('initialization')
    model = FastSpeech2(pre, config).eval(); model.load_state_dict(checkpoint['model'])
    del checkpoint
    feature = ROOT / pre['path']['preprocessed_path']
    speakers = json.loads((feature / 'speakers.json').read_text())
    lexicon = {parts[0]: parts[1:] for parts in [line.split() for line in
               (ROOT / 'data/preprocessing/english/lexicon.txt').read_text().splitlines()]}
    # Preserve assignment pronunciations; use the bundled English lexicon for
    # additional known words rather than dropping unknown phones or words.
    fallback = {}
    for line in (ROOT / 'Telugu/lexicon/librispeech-lexicon.txt').read_text().splitlines():
        parts = line.split()
        if parts:
            fallback.setdefault(parts[0].lower(), parts[1:])
    for word, phones in fallback.items():
        lexicon.setdefault(word, phones)
    source_rows = {r['prepared_audio_path']: r for r in [json.loads(line) for line in
                   (ROOT / 'data/splits/v1/english/train-prepared.jsonl').read_text().splitlines()]}
    selected = []
    if args.text is not None:
        if args.speaker not in speakers:
            raise ValueError('Unknown speaker label')
        sample_id = 'custom-' + hashlib.sha256((args.text+'|'+args.speaker).encode()).hexdigest()[:12]
        selected = [{'text': args.text, 'speaker_id': args.speaker,
                     'sample_id': sample_id, 'prepared_audio_path': None}]
    elif args.reference_generation:
        reference_path = (ROOT / args.reference_generation).resolve()
        assert reference_path.is_relative_to(ROOT / 'Results')
        reference = json.loads(reference_path.read_text())
        assert reference['vocoder'] == 'universal HiFi-GAN' and reference['duration_control'] == 1.0
        for sample in reference['samples']:
            row = source_rows[sample['source_audio_path']]
            assert row['speaker_id'] == sample['speaker_label'] and row['text'] == sample['text']
            selected.append(row)
    else:
        reviews = json.loads((ROOT / 'data/audio-review/ratings.json').read_text())['reviews']
        for review in reviews.values():
            if review['status'] != 'good' or review['language'] != 'english' or review['kind'] != 'dataset':
                continue
            row = source_rows.get(review['path'])
            if row is None:
                continue
            assert hashlib.sha256((ROOT / row['prepared_audio_path']).read_bytes()).hexdigest() == review['audio_sha256'], 'Reviewed source audio changed'
            selected.append(row)
    selected = selected[:args.limit]
    if not selected:
        raise ValueError('No approved English training recordings found')
    weights = ROOT / 'Telugu/hifigan/generator_universal.pth.tar'
    vocoder = hifigan.Generator(hifigan.AttrDict(json.loads((ROOT / 'Telugu/hifigan/config.json').read_text())))
    vocoder.load_state_dict(torch.load(weights, map_location='cpu', weights_only=False)['generator'])
    vocoder.eval(); vocoder.remove_weight_norm()
    category = 'custom_english_samples' if args.text is not None else 'reviewed_english_samples'
    if pacing == 'punctuation':
        category = 'paced_english_samples'
    out = ROOT / 'English/artifacts/Training' / run / category / f"best-{metadata['step']}"
    out.mkdir(parents=True, exist_ok=True)
    report = {'checkpoint': metadata, 'run_id': run, 'initialization': initialization,
              'reference_generation': args.reference_generation,
              'device': 'cpu', 'denoising': False, 'duration_control': duration_control,
              'pacing': pacing, 'boundary_fade_seconds': .005 if pacing == 'punctuation' else 0,
              'vocoder': 'universal HiFi-GAN', 'vocoder_sha256': hashlib.sha256(weights.read_bytes()).hexdigest(),
              'scope': ('Custom English text; user listening preview' if args.text is not None else
                        'User-approved source training texts; not independent held-out quality evaluation'), 'samples': []}
    for row in selected:
        text = normalize_text(row['text'], 'english')
        unknown = [word for word in text.split() if word not in lexicon]
        if unknown:
            raise ValueError('Unknown preview words: ' + repr(unknown))
        phones = [p for word in text.split() for p in lexicon[word]]
        sequence = text_to_sequence('{' + ' '.join(phones) + '}', [])
        assert len(sequence) == len(phones)
        speaker = row['speaker_id']
        chunks = plan_utterance(row['text']) if pacing == 'punctuation' else [Chunk(row['text'], text, '', 0)]
        waves, predictions, chunk_details = [], [], []
        for chunk in chunks:
            chunk_phones = [p for word in chunk.normalized_text.split() for p in lexicon[word]]
            chunk_sequence = text_to_sequence('{'+' '.join(chunk_phones)+'}', [])
            assert len(chunk_sequence) == len(chunk_phones)
            inputs = (torch.tensor([speakers[speaker]]), torch.tensor([chunk_sequence]),
                      torch.tensor([len(chunk_sequence)]), len(chunk_sequence))
            with torch.inference_mode():
                prediction = model(*inputs)
                raw_frames = int(prediction[9][0])
                if pacing == 'punctuation' or duration_control != 1:
                    durations = scaled_durations(prediction[4], duration_control)
                    target_frames = int(durations.sum())
                    if not 0 < target_frames <= 3000:
                        raise ValueError('Unusable paced duration')
                    prediction = model(*inputs, d_targets=durations,
                                       mel_lens=torch.tensor([target_frames]), max_mel_len=target_frames)
                mel = prediction[1]
                frames = int(prediction[9][0])
                assert 0 < frames <= 3000
                wave = vocoder(mel[:, :frames].transpose(1, 2)).squeeze().numpy()
            assert np.isfinite(wave).all() and len(wave) == frames * 256
            waves.append(wave); predictions.append(prediction)
            chunk_details.append({'source_text':chunk.source_text, 'normalized_text':chunk.normalized_text,
                                  'phones':chunk_phones, 'punctuation':chunk.punctuation,
                                  'pause_after_seconds':chunk.pause_after_seconds,
                                  'unscaled_predicted_frames':raw_frames, 'mel_frames':frames})
        if pacing == 'punctuation':
            waveform, gap_samples = assemble_audio(waves, chunks, 22050)
        else:
            waveform, gap_samples = waves[0], [0]
        frames = sum(entry['mel_frames'] for entry in chunk_details)
        assert np.isfinite(waveform).all()
        directory = out / row['sample_id']; directory.mkdir(exist_ok=True)
        audio_path = directory / 'preview.wav'
        sf.write(audio_path, waveform, 22050, subtype='PCM_16')
        offset = 0
        for index, (prediction, chunk_info, wave, gap) in enumerate(zip(predictions, chunk_details, waves, gap_samples)):
            name = 'predictions.npz' if pacing == 'legacy' else f'chunk-{index+1:02d}-predictions.npz'
            np.savez(directory / name, mel=prediction[1][0].numpy(), duration=prediction[5][0].numpy(),
                     normalized_pitch=prediction[2][0].numpy(), normalized_energy=prediction[3][0].numpy())
            chunk_info.update(start_seconds=offset/22050, speech_end_seconds=(offset+len(wave))/22050,
                              pause_samples=gap, predictions_path=str((directory/name).relative_to(ROOT)))
            offset += len(wave) + gap
        assert len(waveform) == offset
        info = {'text': row['text'], 'normalized_text': text, 'phones': phones,
                'run_id': run, 'initialization': initialization,
                'speaker_label': speaker, 'speaker_label_estimated': True,
                'checkpoint_step': metadata['step'], 'checkpoint_path': str(path.relative_to(ROOT)),
                'checkpoint_sha256': metadata['sha256'], 'feature_path': pre['path']['preprocessed_path'],
                'source_audio_path': row['prepared_audio_path'], 'source_split': None if args.text is not None else 'train',
                'audio_path': str(audio_path.relative_to(ROOT)), 'sampling_rate': 22050,
                'seconds': len(waveform) / 22050, 'mel_frames': frames, 'denoising': False,
                'duration_control':duration_control, 'pacing':pacing, 'chunks':chunk_details,
                'waveform_silence_samples':sum(gap_samples),
                'boundary_fade_seconds':.005 if pacing == 'punctuation' else 0,
                'mel_frames_note':'Per-clause predicted frames; inserted waveform pauses are recorded separately',
                'quality_assessment': 'not_performed_user_will_listen'}
        (directory / 'preview.json').write_text(json.dumps(info, ensure_ascii=False, indent=2) + '\n')
        report['samples'].append(info)
        (out / 'generation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps(info, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
