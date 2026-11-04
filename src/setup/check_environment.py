#!/usr/bin/env python3
"""Exercise the real upstream model and feature extraction; no trained weights needed."""
import argparse
import importlib
import json
import os
from pathlib import Path
import platform
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--require-cuda', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    repo = args.repo.resolve()
    if not (repo / 'src/model/fastspeech2.py').is_file():
        parser.error('Expected the upstream FastSpeech2 checkout')
    os.chdir(repo)
    sys.path.insert(0, str(repo/'src'))
    modules = ['numpy', 'scipy', 'librosa', 'pyworld', 'soundfile', 'tgt',
               'yaml', 'datasets', 'matplotlib', 'torch', 'torchaudio',
               'torch.utils.tensorboard']
    versions = {}
    for name in modules:
        module = importlib.import_module(name)
        versions[name] = getattr(module, '__version__', 'imported')
    import numpy as np
    import torch
    import yaml
    import pyworld
    from g2p_en import G2p
    from model import FastSpeech2
    from audio.stft import TacotronSTFT

    cuda = torch.cuda.is_available()
    if args.require_cuda and not cuda:
        raise RuntimeError('PyTorch cannot use CUDA; do not start a paid training run')
    device = torch.device('cuda' if cuda else 'cpu')
    torch.manual_seed(42)
    assert len(G2p()('This is a speech test.')) > 0
    with (repo / 'src/config/LJSpeech/preprocess.yaml').open() as f:
        preprocess = yaml.safe_load(f)
    with (repo / 'src/config/LJSpeech/model.yaml').open() as f:
        model_config = yaml.safe_load(f)
    # Temporary synthetic statistics are only for this software smoke check.
    # They must NEVER be used as dataset statistics or evaluation evidence.
    with tempfile.TemporaryDirectory(prefix='fs2-smoke-') as directory:
        preprocess['path']['preprocessed_path'] = directory
        preprocess['preprocessing']['pitch']['feature'] = 'phoneme_level'
        preprocess['preprocessing']['energy']['feature'] = 'phoneme_level'
        Path(directory, 'stats.json').write_text(json.dumps({
            'pitch': [-3, 3, 0, 1], 'energy': [-3, 3, 0, 1]}))
        model = FastSpeech2(preprocess, model_config).to(device).eval()
        speakers = torch.zeros(1, dtype=torch.long, device=device)
        texts = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long, device=device)
        lengths = torch.tensor([5], dtype=torch.long, device=device)
        mel_lengths = torch.tensor([10], dtype=torch.long, device=device)
        durations = torch.full((1, 5), 2, dtype=torch.long, device=device)
        outputs = model(speakers, texts, lengths, 5, mel_lens=mel_lengths,
                        max_mel_len=10, p_targets=torch.zeros(1, 5, device=device),
                        e_targets=torch.zeros(1, 5, device=device), d_targets=durations)
        assert tuple(outputs[0].shape) == (1, 10, 80)
        assert all(torch.isfinite(t).all() for t in outputs[:5])
        loss = outputs[0].square().mean() + sum(t.square().mean() for t in outputs[2:5])
        loss.backward()
        assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        # The upstream STFT explicitly calls CUDA, so test it only on a GPU.
        if cuda:
            stft = TacotronSTFT(1024, 256, 1024, 80, 22050, 0, 8000)
            t = torch.arange(22050, dtype=torch.float32) / 22050
            waveform = (0.1 * torch.sin(2 * torch.pi * 220 * t)).unsqueeze(0)
            mel, energy = stft.mel_spectrogram(waveform)
            assert mel.shape[1] == 80 and torch.isfinite(mel).all()
            assert torch.isfinite(energy).all()
    signal = 0.1 * np.sin(2 * np.pi * 220 * np.arange(22050) / 22050)
    f0, _ = pyworld.dio(signal.astype(np.float64), 22050)
    assert len(f0) > 0 and np.isfinite(f0).all()
    report = {'status': 'passed', 'python': platform.python_version(),
              'cuda_available': cuda, 'device': torch.cuda.get_device_name(0) if cuda else 'cpu',
              'torch_cuda': torch.version.cuda, 'packages': versions,
              'checks': ['English G2P', 'model forward/backward', 'WORLD pitch'] +
                        (['CUDA STFT/mel/energy'] if cuda else []),
              'not_validated': ['Telugu frontend and alignment', 'dataset preprocessing',
                                'trained checkpoint quality', 'vocoder inference']}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
