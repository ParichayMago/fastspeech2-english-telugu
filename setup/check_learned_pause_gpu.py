#!/usr/bin/env python3
"""Discarded CUDA optimizer smoke check before the bounded learned-pause pilot."""
import hashlib
import json
import os
from pathlib import Path
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
RUN = 'english-learned-pauses-pilot-20261104'
PARENT_SHA = 'b4955131de0b8915ede03eb2232fe4da9702136c1086d67bb31f5c0943f24aad'


def main():
    os.chdir(ROOT)
    import numpy as np
    import torch
    import yaml
    from setup.learned_pause_data import LearnedPauseDataset
    from setup.train_assignment import FixedLearningRateOptim
    from model.learned_pause import LearnedPauseFastSpeech2, LearnedPauseLoss
    from utils.tools import to_device
    assert torch.cuda.is_available()
    torch.set_num_threads(2)
    torch.manual_seed(20261104)
    configs = ROOT/'data/rebuild-config/learned-pauses-v1/English'
    pre, mod, cfg = [yaml.safe_load((configs/n).read_text()) for n in
                     ('preprocess.yaml', 'model.yaml', 'train.yaml')]
    cfg['optimizer']['batch_size'] = 8
    parent = ROOT/'output/initialization/english-learned-pause-parent.pth.tar'
    assert hashlib.sha256(parent.read_bytes()).hexdigest() == PARENT_SHA
    initial = torch.load(parent, map_location='cpu', weights_only=False)
    for name in ('stats.json', 'speakers.json'):
        assert (ROOT/pre['path']['preprocessed_path']/name).read_bytes() == (
            ROOT/initial['configs'][0]['path']['preprocessed_path']/name).read_bytes()
    net = LearnedPauseFastSpeech2(pre, mod)
    missing = net.load_parent(initial['model'])
    assert all(torch.equal(net.state_dict()[key], value) for key, value in initial['model'].items())
    net = net.cuda()
    loss = LearnedPauseLoss(mod).cuda()
    opt = FixedLearningRateOptim(net, cfg, 2e-5, 2e-4)
    checks = []
    torch.cuda.reset_peak_memory_stats()
    for split in ('train.txt', 'val.txt'):
        dataset = LearnedPauseDataset(split, pre, cfg)
        feature_root = ROOT/pre['path']['preprocessed_path']
        # mmap reads only the array header when ranking, without loading all mels.
        indices = sorted(range(len(dataset)), key=lambda i: np.load(
            feature_root/'mel'/f'{dataset.speaker[i]}-mel-{dataset.basename[i]}.npy',
            mmap_mode='r').shape[0], reverse=True)[:8]
        raw = dataset.reprocess([dataset[i] for i in indices], list(range(8)))
        batch = to_device(raw[:12], torch.device('cuda'))
        boundaries = torch.from_numpy(raw[12]).long().cuda()
        net.train(split == 'train.txt')
        opt.zero_grad()
        with torch.set_grad_enabled(split == 'train.txt'):
            pred, aux = net(*batch[2:], boundary_types=boundaries, return_aux=True)
            values = loss(batch, pred, boundaries, aux)
            assert pred[0].shape == batch[6].shape
            assert all(torch.isfinite(x).item() for x in values)
            gradients = {}
            if split == 'train.txt':
                before = net.mel_linear.weight.detach().clone()
                values[0].backward()
                for name, layer in (('presence', net.pause_presence.linear_layer),
                                    ('pause_duration', net.pause_duration.linear_layer),
                                    ('boundary_embedding', net.boundary_embedding)):
                    gradients[name] = float(layer.weight.grad.abs().sum())
                    assert gradients[name] > 0 and np.isfinite(gradients[name])
                norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.)
                assert torch.isfinite(norm).item()
                opt.step_and_update_lr()
                assert not torch.equal(before, net.mel_linear.weight.detach())
            checks.append({'split':split, 'examples':len(dataset), 'batch_size':8,
                           'mel_frames':batch[7].tolist(), 'full_targets':True,
                           'losses':[float(x.detach()) for x in values],
                           'new_module_gradient_sums':gradients})
    report = {'status':'passed', 'utc':datetime.now(timezone.utc).isoformat(),
              'gpu':torch.cuda.get_device_name(0), 'torch':torch.__version__,
              'peak_gpu_gib':torch.cuda.max_memory_allocated()/1024**3,
              'checks':checks, 'parent_sha256':PARENT_SHA,
              'parent_step':initial['step'], 'new_parameters':missing,
              'learning_rates':[g['lr'] for g in opt._optimizer.param_groups],
              'optimizer_updates':1, 'smoke_weights_discarded':True,
              'parent_normalization_and_speakers_identical':True,
              'longest_eight_per_split':True, 'quality_validated':False}
    out = ROOT/'Results/Training'/RUN/'checks/gpu-readiness.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
