#!/usr/bin/env python3
"""Synthetic CUDA training smoke check, never assignment training evidence."""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    os.chdir(repo)
    sys.path.insert(0, str(repo/'Telugu'))

    import torch
    import yaml
    from model import FastSpeech2, FastSpeech2Loss
    from model.optimizer import ScheduledOptim

    assert torch.cuda.is_available(), 'CUDA is required for the paid GPU check'
    torch.manual_seed(42)
    configs = [yaml.safe_load((repo / 'Telugu/config/LJSpeech' / name).read_text())
               for name in ('preprocess.yaml', 'model.yaml', 'train.yaml')]
    preprocess, model_config, train_config = configs
    preprocess['preprocessing']['pitch']['feature'] = 'phoneme_level'
    preprocess['preprocessing']['energy']['feature'] = 'phoneme_level'
    losses, timings = [], []
    with tempfile.TemporaryDirectory(prefix='fs2-synthetic-training-') as directory:
        # Artificial ranges exist only in this temporary software check.
        preprocess['path']['preprocessed_path'] = directory
        Path(directory, 'stats.json').write_text(json.dumps({
            'pitch': [-3, 3, 0, 1], 'energy': [-3, 3, 0, 1]}))
        device = torch.device('cuda')
        model = FastSpeech2(preprocess, model_config).to(device).train()
        criterion = FastSpeech2Loss(preprocess, model_config).to(device)
        optimizer = ScheduledOptim(model, train_config, model_config, 0)
        batch_size, tokens, frames = 2, 80, 800
        batch = [
            ['synthetic'] * batch_size, ['synthetic'] * batch_size,
            torch.zeros(batch_size, dtype=torch.long, device=device),
            torch.randint(1, 20, (batch_size, tokens), device=device),
            torch.full((batch_size,), tokens, dtype=torch.long, device=device),
            tokens,
            torch.randn(batch_size, frames, 80, device=device),
            torch.full((batch_size,), frames, dtype=torch.long, device=device),
            frames,
            torch.zeros(batch_size, tokens, device=device),
            torch.zeros(batch_size, tokens, device=device),
            torch.full((batch_size, tokens), 10, dtype=torch.long, device=device),
        ]
        initial_weights = model.mel_linear.weight.detach().clone()
        torch.cuda.reset_peak_memory_stats()
        for _ in range(3):
            torch.cuda.synchronize()
            started = time.perf_counter()
            optimizer.zero_grad()
            predictions = model(*batch[2:])
            loss = criterion(batch, predictions)[0]
            assert torch.isfinite(loss), 'Non-finite training loss'
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            assert torch.isfinite(norm), 'Non-finite gradient norm'
            optimizer.step_and_update_lr()
            torch.cuda.synchronize()
            losses.append(loss.item())
            timings.append(time.perf_counter() - started)
        changed = not torch.equal(initial_weights, model.mel_linear.weight.detach())
        assert changed, 'Optimizer did not update model weights'
        result = {
            'status': 'passed', 'data_kind': 'synthetic_software_smoke_only',
            'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__,
            'torch_cuda': torch.version.cuda,
            'batch_size': batch_size, 'phonemes_per_sample': tokens,
            'mel_frames_per_sample': frames, 'optimizer_steps': len(losses),
            'losses': losses, 'seconds_per_step': timings,
            'peak_allocated_memory_gb': torch.cuda.max_memory_allocated() / 1024**3,
            'weights_updated': changed,
            'not_validated': ['real dataset training speed', 'English/Telugu alignment',
                              'speech quality', 'full-run cost or convergence'],
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
