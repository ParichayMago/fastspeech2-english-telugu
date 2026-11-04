#!/usr/bin/env python3
"""Local preview synthesis from saved checkpoints; no cloud or quality scoring.

These short preview sentences use the existing lexicons. This is not the final
assignment test-sentence frontend; unknown words fail rather than being dropped.
"""
import hashlib
import argparse
import io
import json
import os
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
os.chdir(ROOT)
os.environ.setdefault('MPLBACKEND', 'Agg')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', help='Use an isolated corrected training run')
    parser.add_argument('--checkpoint', choices=['latest', 'best'], default='latest')
    args = parser.parse_args()
    if args.run_id and (args.run_id != Path(args.run_id).name or args.run_id in ('.', '..')):
        parser.error('run-id must be a directory name')
    if args.checkpoint == 'best' and not args.run_id:
        parser.error('--checkpoint best requires --run-id')
    import numpy as np
    import soundfile as sf
    import torch
    import hifigan
    from model import FastSpeech2
    from text import text_to_sequence
    from setup.prepare_assignment_corpus import normalize_text

    torch.set_num_threads(4)
    torch.manual_seed(20261103)
    device = torch.device('cpu')
    vocoder_path = ROOT/'src/hifigan/generator_universal.pth.tar'
    if not vocoder_path.exists():
        with zipfile.ZipFile(str(vocoder_path)+'.zip') as archive:
            with archive.open(vocoder_path.name) as source, vocoder_path.open('wb') as target:
                import shutil
                shutil.copyfileobj(source, target)
    vocoder = hifigan.Generator(hifigan.AttrDict(json.loads((ROOT/'src/hifigan/config.json').read_text())))
    vocoder.load_state_dict(torch.load(vocoder_path, map_location=device, weights_only=False)['generator'])
    vocoder.eval()
    vocoder.remove_weight_norm()
    report = {'scope':'initial_preview_only_not_required_assignment_test_outputs',
              'device':'cpu', 'quality_assessment':'not_performed_user_will_listen',
              'vocoder':'pretrained universal HiFi-GAN',
              'vocoder_sha256':hashlib.sha256(vocoder_path.read_bytes()).hexdigest(),
              'run_id':args.run_id, 'checkpoint_selection':args.checkpoint,
              'samples':{}}
    for language, sentence in [('english','The meeting will start on time.'),
                                ('telugu','అతని పట్టు చాలా బలంగా ఉంది.')]:
        if args.run_id:
            checkpoint_path = ROOT/'output/runs'/args.run_id/language/(args.checkpoint+'.pth.tar')
            metadata_path = checkpoint_path.parent/('best.json' if args.checkpoint == 'best' else 'checkpoint.json')
            metadata = json.loads(metadata_path.read_text())
            checkpoint_bytes = checkpoint_path.read_bytes()
            checkpoint_sha256 = hashlib.sha256(checkpoint_bytes).hexdigest()
            if checkpoint_sha256 != metadata['sha256'] or len(checkpoint_bytes) != metadata['bytes']:
                raise ValueError('Checkpoint backup is changing or hash verification failed; retry after backup completes')
            checkpoint = torch.load(io.BytesIO(checkpoint_bytes), map_location=device, weights_only=False)
            if checkpoint['step'] != metadata['step']:
                raise ValueError('Checkpoint step differs from verified metadata')
            del checkpoint_bytes
        else:
            checkpoint_path = ROOT/'output'/language/'bounded/latest.pth.tar'
            checkpoint_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        pre, model_config, _ = checkpoint['configs']
        lexicon = {line.split()[0]:line.split()[1:] for line in
                   (ROOT/'data/preprocessing'/language/'lexicon.txt').read_text().splitlines()}
        normalized = normalize_text(sentence, language)
        words = normalized.split()
        unknown = [word for word in words if word not in lexicon]
        if unknown:
            raise ValueError('Preview requires G2P for unknown words: '+repr(unknown))
        phones = [phone for word in words for phone in lexicon[word]]
        sequence = text_to_sequence('{'+' '.join(phones)+'}', [])
        if len(sequence) != len(phones):
            raise ValueError('A phone was silently dropped')
        speaker_map = json.loads((ROOT/pre['path']['preprocessed_path']/'speakers.json').read_text())
        from collections import Counter
        counts = Counter(line.split('|')[1] for line in
                         (ROOT/pre['path']['preprocessed_path']/'train.txt').read_text().splitlines())
        speaker = sorted(counts, key=lambda s:(-counts[s],s))[0]
        model = FastSpeech2(pre, model_config).to(device).eval()
        model.load_state_dict(checkpoint['model'])
        n = len(sequence)
        with torch.inference_mode():
            prediction = model(torch.tensor([speaker_map[speaker]]),
                               torch.tensor([sequence], dtype=torch.long),
                               torch.tensor([n]), n)
            mel = prediction[1]
            frames = int(prediction[9][0])
            if not 0 < frames <= 3000:
                raise ValueError('Unusable predicted length: '+str(frames))
            waveform = vocoder(mel[:, :frames].transpose(1,2)).squeeze().cpu().numpy()
        if not np.isfinite(waveform).all():
            raise ValueError('Non-finite audio samples')
        directory = (ROOT/'Results/Training'/args.run_id/'speech_samples'/language/str(checkpoint['step'])
                     if args.run_id else ROOT/'Results'/language.title()/'initial_generated_audios')
        directory.mkdir(parents=True, exist_ok=True)
        path = directory/'preview.wav'
        sf.write(path, waveform, 22050, subtype='PCM_16')
        np.savez(directory/'predictions.npz', mel=mel[0,:frames].cpu().numpy(),
                 duration=prediction[5][0].cpu().numpy(),
                 normalized_pitch=prediction[2][0].cpu().numpy(),
                 normalized_energy=prediction[3][0].cpu().numpy())
        info = {'text':sentence,'normalized_text':normalized,'phones':phones,
                'checkpoint_step':checkpoint['step'],'speaker_label':speaker,
                'checkpoint_path':str(checkpoint_path.relative_to(ROOT)),
                'checkpoint_sha256':checkpoint_sha256,
                'feature_path':pre['path']['preprocessed_path'],
                'speaker_label_estimated':language=='english',
                'audio_path':str(path.relative_to(ROOT)), 'sampling_rate':22050,
                'seconds':len(waveform)/22050, 'mel_frames':frames,
                'quality_assessment':'not_performed_user_will_listen'}
        (directory/'preview.json').write_text(json.dumps(info, ensure_ascii=False, indent=2)+'\n')
        report['samples'][language] = info
        report_path = (ROOT/'Results/Training'/args.run_id/'speech_samples/generation.json'
                       if args.run_id else ROOT/'Results/Training/initial-speech-generation.json')
        report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps(info,ensure_ascii=False), flush=True)


if __name__=='__main__':
    main()
