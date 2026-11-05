#!/usr/bin/env python3
"""Bounded, resumable real-data training of separate English/Telugu models.

One process alternates independent optimizer updates. No weights are shared.
Training duration is a compute limit, not a convergence criterion.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import sys
import time
from datetime import datetime, timezone

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'Telugu'))
from setup.training_recovery import recover_arguments, restore_selection, select_best, run_directory, equivalent_preprocessing


def source_fingerprints():
    """Hash the language-local source used by future training/resume records."""
    patterns=('Telugu/model/*.py','Telugu/transformer/*.py','Telugu/text/*.py','Telugu/utils/*.py',
                     'Telugu/dataset.py','Telugu/setup/train_assignment.py','Telugu/setup/training_recovery.py','Telugu/setup/learned_pause_*.py')
    return {str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
            for pattern in patterns for path in sorted(ROOT.glob(pattern))}


def utc():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


class FixedLearningRateOptim:
    """New Adam state for an explicitly initialized fine-tuning experiment."""
    def __init__(self, model, config, learning_rate, new_head_learning_rate=None):
        import torch
        cfg = config['optimizer']
        parameters = model.parameters()
        if new_head_learning_rate is not None:
            prefixes = ('boundary_embedding.', 'pause_presence.', 'pause_duration.')
            parameters = [dict(params=[p for n,p in model.named_parameters() if not n.startswith(prefixes)],lr=learning_rate),
                          dict(params=[p for n,p in model.named_parameters() if n.startswith(prefixes)],lr=new_head_learning_rate)]
        self._optimizer = torch.optim.Adam(parameters, lr=learning_rate,
            betas=cfg['betas'], eps=cfg['eps'], weight_decay=cfg['weight_decay'])

    def zero_grad(self):
        self._optimizer.zero_grad()

    def step_and_update_lr(self):
        self._optimizer.step()

    def load_state_dict(self, state):
        self._optimizer.load_state_dict(state)


def main(argv=None, language=None):
    if language not in (None, 'english', 'telugu'):
        raise ValueError('Unknown training language')
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--minutes', type=float, default=40)
    p.add_argument('--max-steps', type=int, default=10000)
    p.add_argument('--batch-size', type=int)
    p.add_argument('--checkpoint-every', type=int, default=100)
    p.add_argument('--validation-every', type=int, default=500)
    p.add_argument('--run-id',default=None,required=language is not None,
                   help='Isolated experiment name under output/runs/; required by language entrypoints')
    mode=p.add_mutually_exclusive_group()
    mode.add_argument('--fresh',action='store_true')
    mode.add_argument('--resume',action='store_true',help='Restore the saved Telugu/model/optimizer policy and selection state')
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda',help='CUDA by default; CPU supports local smoke tests')
    p.add_argument('--language', choices=[language] if language else ['both','english','telugu'],
                   default=language or 'both')
    p.add_argument('--initialize-from', help='Model-only checkpoint for a separate English fine-tuning run')
    p.add_argument('--initialize-sha256')
    p.add_argument('--learning-rate', type=float)
    p.add_argument('--early-stop-patience', type=int,
                   help='Fine-tuning only: stop after this many full evaluations without improvement')
    p.add_argument('--learned-pauses', action='store_true',default=None)
    p.add_argument('--config-root')
    p.add_argument('--new-head-learning-rate', type=float)
    args = p.parse_args(argv)
    if not 0 < args.minutes <= 180 or (args.batch_size is not None and args.batch_size < 1):
        p.error('Session must be positive and at most 180 minutes')
    if args.run_id and not args.run_id.replace('-','').replace('_','').isalnum():
        p.error('Run ID must be a simple directory name')
    if args.fresh and not args.run_id:
        p.error('Fresh training requires an isolated run ID')
    try:resume_metadata=recover_arguments(args,ROOT)
    except ValueError as exc:p.error(str(exc))
    if args.initialize_from and (not args.fresh or args.language != 'english' or not args.initialize_sha256):
        p.error('Initialization requires --fresh, --language english and --initialize-sha256')
    if args.initialize_from and (args.learning_rate is None or not 0 < args.learning_rate <= .001):
        p.error('Fine-tuning requires an explicit positive learning rate <= 0.001')
    if not args.initialize_from and not args.resuming and (args.learning_rate is not None or args.initialize_sha256):
        p.error('Fine-tuning options require --initialize-from')
    if args.early_stop_patience < 0 or (args.early_stop_patience and not args.initialize_from and not args.resuming):
        p.error('Early stopping requires fine-tuning initialization and nonnegative patience')
    if args.learned_pauses and (args.language != 'english' or not (args.initialize_from or args.resuming)):
        p.error('Learned pauses require an initialized English-only experiment')
    run_root=ROOT/'output/runs'/args.run_id if args.run_id else None
    out = run_root/'session' if run_root else ROOT/'output/training-session'
    if (out/'STOP').exists():
        p.error('Retained STOP marker blocks training: '+str(out/'STOP')+
                '. After confirming that you intend to continue, remove this marker explicitly '
                'and rerun --resume. No checkpoints or session files have been changed.')
    os.chdir(ROOT)
    import numpy as np
    import torch
    import yaml
    from torch.utils.data import DataLoader
    from torch.utils.tensorboard import SummaryWriter
    from dataset import Dataset
    from model import FastSpeech2, FastSpeech2Loss, ScheduledOptim
    from utils.tools import to_device
    if args.learned_pauses:
        from model.learned_pause import LearnedPauseFastSpeech2, LearnedPauseLoss, pause_metrics
        from setup.learned_pause_data import LearnedPauseDataset
    torch.set_num_threads(2)
    random.seed(20261103)
    np.random.seed(20261103)
    torch.manual_seed(20261103)
    if args.device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is required by default; choose --device cpu explicitly for local tests')
    device = torch.device(args.device)
    if args.fresh and (out/'status.json').exists():
        raise ValueError('Fresh run already exists; use a new run ID')
    out.mkdir(parents=True, exist_ok=True)
    lock = out / 'trainer.lock'
    # flock is released by the OS on crashes; the file itself is harmless.
    import fcntl
    lock_handle = lock.open('a')
    fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    stop = [False]
    early_stop = [False]
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.__setitem__(0, True))
    started = time.monotonic()
    deadline = started + args.minutes * 60
    state = {'status': 'initializing', 'started_utc': utc(),
             'run_id':args.run_id,'fresh_initialization':args.fresh and not args.initialize_from,
             'new_run':args.fresh,
             'pid': os.getpid(), 'settings': vars(args),
             'gpu': torch.cuda.get_device_name(0) if device.type=='cuda' else None,
             'device':str(device),
             'scope': 'bounded_initial_training_not_convergence_or_quality_validation',
             'english_speaker_labels': 'estimated_acoustic_clusters_not_verified_identities',
             'languages': {}}
    initialization = None
    if args.resuming:
        initialization=next(iter(resume_metadata.values())).get('initialization')
        if args.learning_rate is not None and not initialization:
            raise ValueError('Fine-tuning initialization metadata is missing; restore run-config.json')
    if args.initialize_from:
        initial_path = (ROOT / args.initialize_from).resolve()
        if not initial_path.is_relative_to(ROOT / 'output'):
            raise ValueError('Initialization must use an assignment checkpoint under output/')
        digest = hashlib.sha256(initial_path.read_bytes()).hexdigest()
        if digest != args.initialize_sha256:
            raise ValueError('Initialization checkpoint SHA256 mismatch')
        initial = torch.load(initial_path, map_location='cpu', weights_only=False)
        if initial['language'] != 'english':
            raise ValueError('English-only initialization expected')
        initialization = {'path': str(initial_path.relative_to(ROOT)), 'sha256': digest,
                          'parent_step': initial['step'], 'new_optimizer': True,
                          'learning_rate': args.learning_rate}
        state['initialization'] = initialization
    trainers = []

    def stopping():
        return stop[0] or early_stop[0] or (out / 'STOP').exists() or time.monotonic() >= deadline

    def publish():
        state['updated_utc'] = utc()
        state['elapsed_seconds'] = round(time.monotonic() - started, 2)
        state['peak_gpu_gib'] = torch.cuda.max_memory_allocated() / 1024**3 if device.type=='cuda' else 0
        atomic_json(out / 'status.json', state)

    def record(t, phase, losses, **extra):
        row = {'utc': utc(), 'language': t['language'], 'step': t['step'],
               'phase': phase, 'losses': [float(x) for x in losses], **extra}
        with (t['directory'] / 'metrics.jsonl').open('a') as f:
            f.write(json.dumps(row) + '\n')
        for key, value in zip(('total', 'mel', 'mel_postnet', 'pitch', 'energy', 'duration', 'pause_presence', 'pause_duration'), row['losses']):
            t['writer'].add_scalar(phase + '/' + key, value, t['step'])
        state['languages'][t['language']][phase] = row
        print(json.dumps(row), flush=True)

    def prepare(raw):
        if args.learned_pauses:
            return to_device(raw[:12], device), torch.from_numpy(raw[12]).long().to(device)
        return to_device(raw, device), None

    def predict_and_loss(t, batch, boundaries):
        if args.learned_pauses:
            prediction, aux = t['model'](*batch[2:], boundary_types=boundaries, return_aux=True)
            return prediction,t['loss'](batch,prediction,boundaries,aux),aux
        prediction = t['model'](*batch[2:])
        return prediction,t['loss'](batch,prediction),None

    def save(t):
        d = t['directory']
        path = d / 'latest.pth.tar'
        tmp = d / 'latest.pth.tar.tmp'
        torch.save({'model': t['model'].state_dict(),
                    'model_type':'learned_pause_v1' if args.learned_pauses else 'fastspeech2',
                    'initialization': initialization,
                    'optimizer': t['optimizer']._optimizer.state_dict(),
                    'step': t['step'], 'epoch': t['epoch'],
                    'language': t['language'], 'configs': t['configs'],
                    'estimated_speaker_labels': t['language'] == 'english',
                    'training_policy':{'optimizer_type':'fixed' if initialization else 'scheduled',
                        'learning_rate':args.learning_rate,'new_head_learning_rate':args.new_head_learning_rate,
                        'early_stop_patience':args.early_stop_patience,'learned_pauses':args.learned_pauses},
                    'best_validation':t['best_validation'],'best_checkpoint':t.get('best_checkpoint'),
                    'stale_validations':t.get('stale_validations',0),
                    'last_validation_step':t.get('last_validation_step',-1),
                    'torch_rng_state': torch.get_rng_state(),
                    'python_rng_state':random.getstate(),'numpy_rng_state':np.random.get_state(),
                    'cuda_rng_state': torch.cuda.get_rng_state() if device.type=='cuda' else None}, tmp)
        tmp.replace(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        entry = {'step': t['step'], 'path': str(path.relative_to(ROOT)),
                 'model_type':'learned_pause_v1' if args.learned_pauses else 'fastspeech2',
                 'sha256': digest, 'bytes': path.stat().st_size, 'utc': utc()}
        atomic_json(d / 'checkpoint.json', entry)
        state['languages'][t['language']]['checkpoint'] = entry
        t['writer'].flush()
        publish()

    def validate(t, final=False):
        t['model'].eval()
        sums = np.zeros(8 if args.learned_pauses else 6)
        prosody = {}
        count = 0
        complete = True
        with torch.inference_mode():
            for groups in t['valid_loader']:
                for raw in groups:
                    if stopping() and not final:
                        complete = False
                        break
                    batch, boundaries = prepare(raw)
                    prediction, losses, aux = predict_and_loss(t,batch,boundaries)
                    if args.learned_pauses:
                        for key,value in pause_metrics(batch[11],aux).items():
                            prosody[key] = prosody.get(key,0)+value
                    values = np.array([x.item() for x in losses])
                    if not np.isfinite(values).all():
                        raise RuntimeError('Non-finite validation loss')
                    n = len(raw[0])
                    sums += values * n
                    count += n
                if not complete:
                    break
        if count:
            if prosody:
                tp,fp,fn = prosody['tp'],prosody['fp'],prosody['fn']
                prosody.update(precision=tp/max(1,tp+fp),recall=tp/max(1,tp+fn),
                    f1=2*tp/max(1,2*tp+fp+fn),positive_pause_mae_seconds=
                    prosody['positive_duration_abs_frame_error']/max(1,prosody['positive_boundaries'])*256/22050)
            record(t, 'validation', sums / count, examples=count,
                   complete=count == len(t['valid']), final=final,
                   **({'pause_metrics':prosody} if prosody else {}))
            improved = count == len(t['valid']) and select_best(
                t,float(sums[0]/count),initialization,args.learned_pauses,ROOT,atomic_json)
            if improved:
                state['languages'][t['language']]['best_checkpoint']=t['best_checkpoint']
                publish()
            if (initialization and args.early_stop_patience and count == len(t['valid'])
                    and not final and t.get('last_validation_step')!=t['step']):
                t['stale_validations'] = 0 if improved else t.get('stale_validations', 0) + 1
                state['languages'][t['language']]['validations_without_improvement'] = t['stale_validations']
                if t['stale_validations'] >= args.early_stop_patience:
                    early_stop[0] = True
            if count==len(t['valid']):t['last_validation_step']=t['step']
        t['model'].train()

    try:
        languages = [('english', 'English'), ('telugu', 'Telugu')]
        if args.language != 'both':
            languages = [entry for entry in languages if entry[0] == args.language]
        for language, folder in languages:
            d=run_directory(ROOT,args.run_id,language)
            checkpoint=None
            if args.resuming:
                checkpoint=torch.load(d/'latest.pth.tar',map_location='cpu',weights_only=False)
                if checkpoint['language']!=language:raise ValueError('Checkpoint language differs')
                expected_type='learned_pause_v1' if args.learned_pauses else 'fastspeech2'
                if checkpoint.get('model_type','fastspeech2')!=expected_type:
                    raise ValueError('Saved model type differs from the run policy')
                expected_policy={'optimizer_type':'fixed' if initialization else 'scheduled',
                    'learning_rate':args.learning_rate,'new_head_learning_rate':args.new_head_learning_rate,
                    'early_stop_patience':args.early_stop_patience,'learned_pauses':args.learned_pauses}
                if checkpoint.get('training_policy') and checkpoint['training_policy']!=expected_policy:
                    raise ValueError('Saved optimizer policy differs from run-config.json')
                selection_state=restore_selection(d,checkpoint)
                configs=copy.deepcopy(checkpoint['configs'])
                for filename,digest in resume_metadata[language].get('data_sha256',{}).items():
                    if hashlib.sha256((ROOT/filename).read_bytes()).hexdigest()!=digest:
                        raise ValueError('Training data identity changed on resume: '+filename)
            else:
                configs = [yaml.safe_load((ROOT / args.config_root / folder / name).read_text())
                           for name in ('preprocess.yaml', 'model.yaml', 'train.yaml')]
            pre, mod, cfg = configs
            cfg['optimizer']['batch_size'] = args.batch_size
            cfg['optimizer']['grad_acc_step'] = 1
            cfg['step'].update(total_step=args.max_steps,log_step=20,
                               save_step=args.checkpoint_every,val_step=args.validation_every)
            if args.fresh and d.exists():raise ValueError('Fresh output exists: '+str(d))
            d.mkdir(parents=True, exist_ok=True)
            dataset_class = LearnedPauseDataset if args.learned_pauses else Dataset
            train = dataset_class('train.txt', pre, cfg, sort=True, drop_last=False)
            valid = dataset_class('val.txt', pre, cfg)
            model_class = LearnedPauseFastSpeech2 if args.learned_pauses else FastSpeech2
            model = model_class(pre, mod).to(device)
            if initialization and not args.resuming:
                if args.learned_pauses:
                    old_root = ROOT/initial['configs'][0]['path']['preprocessed_path']
                    new_root = ROOT/pre['path']['preprocessed_path']
                    for name in ('stats.json','speakers.json'):
                        if (old_root/name).read_bytes() != (new_root/name).read_bytes():
                            raise ValueError('Learned-pause transfer must preserve normalization and speaker map')
                    initialization['new_parameters'] = model.load_parent(initial['model'])
                    initialization['model_type'] = 'learned_pause_v1'
                else:
                    if not equivalent_preprocessing(initial['configs'][0],pre) or initial['configs'][1] != mod:
                        raise ValueError('Fine-tuning preprocessing/model configuration differs from checkpoint')
                    model.load_state_dict(initial['model'], strict=True)
            step, epoch = 0, 1
            if checkpoint:
                model.load_state_dict(checkpoint['model'])
                step, epoch = checkpoint['step'], checkpoint['epoch'] + 1
            optim = (FixedLearningRateOptim(model, cfg, args.learning_rate,
                        args.new_head_learning_rate if args.learned_pauses else None) if initialization
                     else ScheduledOptim(model, cfg, mod, step))
            if checkpoint:
                optim.load_state_dict(checkpoint['optimizer'])
                if checkpoint.get('torch_rng_state') is not None:torch.set_rng_state(checkpoint['torch_rng_state'].cpu())
                if checkpoint.get('python_rng_state') is not None:random.setstate(checkpoint['python_rng_state'])
                if checkpoint.get('numpy_rng_state') is not None:np.random.set_state(checkpoint['numpy_rng_state'])
                if device.type=='cuda' and checkpoint.get('cuda_rng_state') is not None:
                    torch.cuda.set_rng_state(checkpoint['cuda_rng_state'].cpu())
            loader = DataLoader(train, batch_size=args.batch_size * 4, shuffle=True,
                                collate_fn=train.collate_fn, num_workers=0)
            vl = DataLoader(valid, batch_size=args.batch_size, shuffle=False,
                            collate_fn=valid.collate_fn, num_workers=0)
            t = {'language': language, 'directory': d, 'configs': configs,
                 'best_validation':float('inf'),
                 'train': train, 'valid': valid, 'loader': loader, 'valid_loader': vl,
                 'iterator': iter(loader), 'pending': [], 'model': model,
                 'optimizer': optim, 'loss': (LearnedPauseLoss(mod) if args.learned_pauses else FastSpeech2Loss(pre,mod)).to(device),
                 'step': step, 'epoch': epoch,
                 'writer': SummaryWriter(str(d / 'tensorboard'))}
            if checkpoint:t.update(selection_state)
            trainers.append(t)
            state['languages'][language] = {'step': step, 'train_examples': len(train),
                                           'feature_version':pre.get('feature_version','legacy'),
                                           'initialized_from_checkpoint':checkpoint is not None,
                                           'validation_examples': len(valid)}
            if t.get('best_checkpoint'):
                state['languages'][language]['best_checkpoint']=t['best_checkpoint']
            if initialization and args.early_stop_patience and t.get('stale_validations',0)>=args.early_stop_patience:
                early_stop[0]=True
            if initialization:
                state['languages'][language].update(initialized_from_checkpoint=True,
                    initialization=initialization, step_semantics='additional fine-tuning updates; saved counter retained on resume')
            # Snapshot exact feature/config identity with the checkpoints.
            files = [ROOT / pre['path']['preprocessed_path'] / n for n in
                     ('train.txt', 'val.txt', 'stats.json', 'speakers.json')]
            if args.learned_pauses:
                files.append(ROOT/pre['path']['preprocessed_path']/'boundary-schema.json')
            run_record={'configs': configs, 'session': vars(args),
                'initialization': initialization,
                'data_sha256': {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
                'source_sha256':source_fingerprints(),
                'resume_semantics': 'weights, Adam state, scheduler step; reshuffle on resume, not bit-exact mid-epoch',
                'source_hours_note': 'source selections approximately 5h/1h; aligned feature hours are smaller; see Preprocessing_Methods.txt'}
            if args.resuming:
                with (d/'resume-history.jsonl').open('a') as stream:
                    stream.write(json.dumps({'utc':utc(),**run_record})+'\n')
            else:atomic_json(d/'run-config.json',run_record)
            model.train()
            if args.run_id:save(t)
            if initialization and not args.resuming:
                # Establish a full baseline before any updates. The parent model
                # remains this run's best until a complete evaluation improves.
                validate(t)
        state['status'] = 'training'
        publish()
        while not stopping() and any(t['step'] < args.max_steps for t in trainers):
            for t in trainers:
                if stopping():
                    break
                if t['step'] >= args.max_steps:
                    continue
                if not t['pending']:
                    try:
                        t['pending'] = list(next(t['iterator']))
                    except StopIteration:
                        t['epoch'] += 1
                        t['iterator'] = iter(t['loader'])
                        t['pending'] = list(next(t['iterator']))
                batch, boundaries = prepare(t['pending'].pop(0))
                t['optimizer'].zero_grad()
                prediction, losses, aux = predict_and_loss(t,batch,boundaries)
                if prediction[0].shape[1]!=batch[6].shape[1]:
                    raise ValueError('Decoder truncated full training mel targets')
                if not all(torch.isfinite(x).item() for x in losses):
                    raise RuntimeError('Non-finite training loss: ' + t['language'])
                losses[0].backward()
                grad = torch.nn.utils.clip_grad_norm_(t['model'].parameters(), 1.0)
                if not torch.isfinite(grad).item():
                    raise RuntimeError('Non-finite gradient: ' + t['language'])
                t['optimizer'].step_and_update_lr()
                t['step'] += 1
                state['languages'][t['language']]['step'] = t['step']
                if t['step'] == 1 or t['step'] % 20 == 0:
                    record(t, 'train', [x.detach().item() for x in losses],
                           learning_rate=t['optimizer']._optimizer.param_groups[0]['lr'],
                           learning_rates=[g['lr'] for g in t['optimizer']._optimizer.param_groups],
                           batch_size=len(batch[0]), gradient_norm=grad.item())
                    publish()
                if t['step'] == 1 or t['step'] % args.checkpoint_every == 0:
                    save(t)
                if t['step'] % args.validation_every == 0 and not stopping():
                    validate(t)
        state['status'] = 'saving_final_checkpoints'
        publish()
        for t in trainers:
            save(t)
        # Full held-out validation after the compute window; supervisor reserves time.
        if not stop[0] and not (out / 'STOP').exists():
            for t in trainers:
                validate(t, final=True)
        for t in trainers:save(t)
        state['status'] = 'completed_bounded_session'
        state['stop_reason'] = ('validation_plateau' if early_stop[0] else
            'stop_requested' if stop[0] or (out / 'STOP').exists() else 'wall_clock_or_step_limit')
    except BaseException as exc:
        state['status'] = 'failed'
        state['error'] = type(exc).__name__ + ': ' + str(exc)
        # Preserve last known-good atomic checkpoints; never save a corrupt update.
        publish()
        raise
    finally:
        for t in trainers:
            t['writer'].close()
        publish()
        lock_handle.close()


if __name__ == '__main__':
    main()
