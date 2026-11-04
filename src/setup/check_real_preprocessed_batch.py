#!/usr/bin/env python3
"""Check real preprocessed batches; never claim this is a trained TTS model."""
import argparse
import json
import os
from pathlib import Path
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--language',choices=['english','telugu'],required=True)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    parser.add_argument('--config-root',default='src/config')
    parser.add_argument('--report-version')
    parser.add_argument('--longest',action='store_true')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[2];os.chdir(root);sys.path.insert(0,str(root/'src'))
    import torch,yaml
    from dataset import Dataset
    from model import FastSpeech2,FastSpeech2Loss
    from utils.tools import to_device
    torch.set_num_threads(2);torch.manual_seed(42)
    directory=root/args.config_root/('English' if args.language=='english' else 'Telugu')
    preprocess,model_config,train_config=[yaml.safe_load((directory/name).read_text())
        for name in ['preprocess.yaml','model.yaml','train.yaml']]
    train_config['optimizer']['batch_size']=2
    device=torch.device(args.device)
    if args.device=='cuda':assert torch.cuda.is_available()
    train=Dataset('train.txt',preprocess,train_config)
    valid=Dataset('val.txt',preprocess,train_config)
    def batch(dataset):
        if len(dataset)<2:raise ValueError('At least two real examples required')
        if args.longest:
            audit=Path(preprocess['path']['preprocessed_path'])/'feature-audit.jsonl'
            frames={r['sample_id']:r['frames'] for r in
                    (json.loads(line) for line in audit.read_text().splitlines()) if r['status']=='passed'}
            indices=sorted(range(len(dataset)),key=lambda i:frames[dataset.basename[i]],reverse=True)[:2]
            candidates=[dataset[i] for i in indices]
        else:
            candidates=[dataset[i] for i in range(min(16,len(dataset)))]
            candidates.sort(key=lambda row:abs(len(row['mel'])-600))
        rows=candidates[:2]
        return to_device(dataset.reprocess(rows,[0,1]),device)
    train_batch,valid_batch=batch(train),batch(valid)
    model=FastSpeech2(preprocess,model_config).to(device)
    loss_fn=FastSpeech2Loss(preprocess,model_config).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-5)
    if args.device=='cuda':torch.cuda.reset_peak_memory_stats()
    model.train(); before=model.mel_linear.weight.detach().clone()
    prediction=model(*train_batch[2:])
    assert prediction[0].shape[1]==train_batch[6].shape[1],'Training decoder truncated mel targets'
    losses=loss_fn(train_batch,prediction)
    assert all(torch.isfinite(x) for x in losses)
    losses[0].backward()
    grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
    assert torch.isfinite(grad)
    optimizer.step()
    assert not torch.equal(before,model.mel_linear.weight.detach())
    model.eval()
    with torch.inference_mode():validation=loss_fn(valid_batch,model(*valid_batch[2:]))
    assert all(torch.isfinite(x) for x in validation)
    report={'status':'passed','scope':'real_audio_preprocessed_software_smoke_only',
        'language':args.language,'train_dataset_rows':len(train),'validation_dataset_rows':len(valid),
        'batch_size':2,'optimizer_steps':1,'weights_updated':True,
        'train_loss':float(losses[0]),'validation_loss':float(validation[0]),
        'estimated_english_speakers':args.language=='english','multi_speaker':model_config['multi_speaker'],
        'device':args.device,
        'feature_version':preprocess.get('feature_version','legacy'),
        'longest_examples_checked':args.longest,
        'training_mel_frames':train_batch[7].tolist(),
        'validation_mel_frames':valid_batch[7].tolist(),
        'full_training_mel_targets_covered':True,
        'peak_allocated_gpu_gib':torch.cuda.max_memory_allocated()/1024**3 if args.device=='cuda' else None,
        'not_validated':['Full training convergence','Generated speech quality','Human-verified speaker identities']}
    out=root/'src/artifacts/Preprocessing'
    if args.report_version:out=out/args.report_version
    out.mkdir(parents=True,exist_ok=True)
    (out/(args.language+'-real-batch-check.json')).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
