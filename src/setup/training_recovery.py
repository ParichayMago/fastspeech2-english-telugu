"""Restore experiment policy and checkpoint selection without restarting a run."""
import copy
import hashlib
import json
import math
from pathlib import Path


def run_directory(root, run_id, language):
    return root/'output/runs'/run_id/language if run_id else root/'output'/language/'bounded'


def equivalent_preprocessing(left,right):
    """Allow only the declared lexicon relocation when transferring old weights."""
    left=copy.deepcopy(left);right=copy.deepcopy(right)
    for config in (left,right):
        path=config.get('path',{}).get('lexicon_path','')
        if path.startswith('lexicon/'):
            config['path']['lexicon_path']='src/'+path
    return left==right


def recover_arguments(args, root):
    """Use the original run policy; only runtime bounds may change on resume."""
    languages=['english','telugu'] if args.language=='both' else [args.language]
    directories=[run_directory(root,args.run_id,language) for language in languages]
    args.resuming=not args.fresh and any((d/'latest.pth.tar').exists() for d in directories)
    if args.resume and not args.resuming:
        raise ValueError('No latest checkpoint exists for the requested resume')
    if args.resuming:
        if args.initialize_from or args.initialize_sha256:
            raise ValueError('Resume restores its saved parent metadata; do not specify initialization flags')
        metadata=[]
        for directory in directories:
            if not (directory/'latest.pth.tar').exists() or not (directory/'run-config.json').exists():
                raise ValueError('Resume requires latest.pth.tar and run-config.json for every selected language')
            metadata.append(json.loads((directory/'run-config.json').read_text()))
        for key,default in [('learning_rate',None),('new_head_learning_rate',.0002),
                            ('early_stop_patience',0),('learned_pauses',False),('config_root','src/config')]:
            values=[m['session'].get(key,default) for m in metadata]
            if any(v!=values[0] for v in values):
                raise ValueError('Selected languages have different training policies')
            requested=getattr(args,key)
            if requested is not None and requested!=values[0]:
                raise ValueError('Resume cannot change saved '+key)
            setattr(args,key,values[0])
        saved_batch=metadata[0]['session'].get('batch_size',8)
        if args.batch_size is not None and args.batch_size!=saved_batch:
            raise ValueError('Resume cannot change the saved batch size')
        args.batch_size=saved_batch
        return {language:m for language,m in zip(languages,metadata)}
    if not args.fresh:
        raise ValueError('No existing run to resume; use --fresh and a new --run-id')
    for key,value in [('batch_size',8),('early_stop_patience',0),('learned_pauses',False),
                      ('config_root','src/config'),('new_head_learning_rate',.0002)]:
        if getattr(args,key) is None:setattr(args,key,value)
    return {}


def restore_selection(directory, checkpoint):
    """Reconcile atomic best files and logs, including legacy checkpoints."""
    import torch
    directory=Path(directory);best_path=directory/'best.pth.tar';sidecar=directory/'best.json'
    entries={}
    if (directory/'metrics.jsonl').exists():
        for line in (directory/'metrics.jsonl').read_text().splitlines():
            try:row=json.loads(line)
            except json.JSONDecodeError:continue  # A crash may leave one partial line.
            if row.get('phase')=='validation' and row.get('complete'):
                old=entries.get(row['step'])
                if old is None or old.get('final',False) or not row.get('final',False):
                    entries[row['step']]=row
    recorded=min((r['losses'][0] for r in entries.values()),default=float('inf'))
    best=copy.deepcopy(checkpoint.get('best_checkpoint'))
    if best_path.exists():
        if sidecar.exists():
            best=json.loads(sidecar.read_text())
            if hashlib.sha256(best_path.read_bytes()).hexdigest()!=best['sha256']:
                raise ValueError('Best checkpoint hash differs from best.json; restore a verified backup')
        else:
            saved=torch.load(best_path,map_location='cpu',weights_only=False)
            best={'step':saved['step'],'validation_total_loss':saved['validation_total_loss'],
                  'sha256':hashlib.sha256(best_path.read_bytes()).hexdigest()}
        loss=float(best['validation_total_loss'])
        if not math.isfinite(loss):raise ValueError('Nonfinite historical best loss')
        if recorded < loss-1e-7:
            raise ValueError('Historical log minimum is better than the available best file; restore its backup')
    else:
        loss=float(checkpoint.get('best_validation',float('inf')))
        if best or math.isfinite(loss) or math.isfinite(recorded):
            raise ValueError('Historical best checkpoint is missing; restore its backup before resuming')
    stale=int(checkpoint.get('stale_validations',0))
    if entries and best:
        stale=sum(r['step']>best['step'] and not r.get('final',False) for r in entries.values())
    return {'best_validation':loss,'best_checkpoint':best,'stale_validations':stale,
            'last_validation_step':max(entries,default=checkpoint.get('last_validation_step',-1))}


def select_best(trainer, loss, initialization, learned, root, atomic_json):
    """Replace a best file only after a strictly improved complete validation."""
    import torch
    if (trainer['directory']/'best.pth.tar').exists() and not trainer.get('best_checkpoint'):
        trainer.update(restore_selection(trainer['directory'],{'best_validation':trainer['best_validation']}))
    if not loss < trainer['best_validation']:return False
    path=trainer['directory']/'best.pth.tar';temporary=path.with_suffix('.tmp')
    torch.save({'model':trainer['model'].state_dict(),'step':trainer['step'],
                'model_type':'learned_pause_v1' if learned else 'fastspeech2',
                'initialization':initialization,'language':trainer['language'],
                'configs':trainer['configs'],'validation_total_loss':loss},temporary)
    temporary.replace(path)
    entry={'step':trainer['step'],'path':str(path.relative_to(root)),
           'model_type':'learned_pause_v1' if learned else 'fastspeech2',
           'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size,
           'validation_total_loss':loss}
    atomic_json(trainer['directory']/'best.json',entry)
    trainer.update(best_validation=loss,best_checkpoint=entry)
    return True
