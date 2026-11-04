"""Portable regression tests for checkpoint recovery, configs and model helpers."""
import contextlib
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import yaml

import train_assignment as runner
from setup.training_recovery import restore_selection,select_best
from setup.learned_pause_config import write_configs
from model.modules import LengthRegulator,VarianceAdaptor
from utils.tools import get_mask_from_lengths

REPO=Path(__file__).resolve().parents[2]


def fixture(root):
    """Small synthetic features; no private datasets or trained weights needed."""
    pre,mod,cfg=[yaml.safe_load((REPO/'English/config/English'/name).read_text()) for name in ('preprocess.yaml','model.yaml','train.yaml')]
    pre['path']['preprocessed_path']='data/features/gap-fixed-v2/english'
    pre['feature_version']='gap-fixed-v2'
    mod['transformer'].update(encoder_layer=1,decoder_layer=1,encoder_hidden=8,decoder_hidden=8,
                              conv_filter_size=16,encoder_dropout=0.,decoder_dropout=0.)
    mod['variance_predictor'].update(filter_size=8,dropout=0.)
    mod['variance_embedding']['n_bins']=8;mod['max_seq_len']=32
    cfg['optimizer']['batch_size']=2
    base=root/'English/config/English';base.mkdir(parents=True)
    for name,value in zip(('preprocess.yaml','model.yaml','train.yaml'),(pre,mod,cfg)):
        (base/name).write_text(yaml.safe_dump(value))
    schema=dict(version='learned-pauses-v1',train_pause_prior=.5,train_pause_pos_weight=1.,
                train_mean_positive_log_duration=float(np.log(3)),train_max_pause_frames=2,
                boundary_types={'none':1,'period':3},fixed_pause_lengths=False)
    rng=np.random.default_rng(2);mel=rng.normal(size=(8,80)).astype(np.float32)
    for version in ('gap-fixed-v2','learned-pauses-v1'):
        out=root/'data/features'/version/'english'
        for kind in ('mel','pitch','energy','duration','boundary'):(out/kind).mkdir(parents=True)
        (out/'stats.json').write_text(json.dumps({'pitch':[-2.,3.,0.,1.],'energy':[-2.,3.,0.,1.]}))
        (out/'speakers.json').write_text(json.dumps({'speaker':0}))
        (out/'boundary-schema.json').write_text(json.dumps(schema))
        pause=version=='learned-pauses-v1'
        phones='HH AH0 sp W ER1 sp L OW0 sp' if pause else 'HH AH0 sp W ER1 L OW0'
        duration=np.array([1,1,2,1,1,0,1,1,0] if pause else [1,1,2,1,1,1,1],dtype=np.int64)
        lines=[]
        for sid in ('sample0','sample1'):
            lines.append(f'{sid}|speaker|{{{phones}}}|hello world low.')
            for kind,array in [('mel',mel),('duration',duration),('pitch',np.ones(len(duration),dtype=np.float32)),
                               ('energy',np.ones(len(duration),dtype=np.float32)),
                               ('boundary',np.array([0,0,1,0,0,1,0,0,3]))]:
                np.save(out/kind/f'speaker-{kind}-{sid}.npy',array)
        for filename in ('train.txt','val.txt'):(out/filename).write_text('\n'.join(lines)+'\n')
    write_configs(root)
    return pre,mod,cfg


class HelperTests(unittest.TestCase):
    def test_cpu_helpers_ignore_cuda_availability_and_follow_tensor_device(self):
        with patch('torch.cuda.is_available',return_value=True):
            mask=get_mask_from_lengths(torch.tensor([2,3]))
            expanded,lengths=LengthRegulator()(torch.ones(1,2,4),torch.tensor([[1,2]]),None)
        self.assertEqual(mask.device.type,'cpu');self.assertEqual(lengths.device.type,'cpu')
        self.assertEqual(mask.tolist(),[[False,False,True],[False,False,False]])
        self.assertEqual(expanded.shape,(1,3,4))
        meta=get_mask_from_lengths(torch.empty(2,device='meta',dtype=torch.long),3)
        self.assertEqual(meta.device.type,'meta')

    def test_energy_control_reaches_both_feature_level_branches(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve();pre,mod,_=fixture(root)
            pre['path']['preprocessed_path']=str(root/pre['path']['preprocessed_path'])
            for level in ('phoneme_level','frame_level'):
                pre['preprocessing']['energy']['feature']=level
                net=VarianceAdaptor(pre,mod).eval();x=torch.randn(1,3,8);mask=torch.zeros(1,3,dtype=torch.bool)
                with patch.object(net,'get_energy_embedding',wraps=net.get_energy_embedding) as energy:
                    with torch.inference_mode():
                        one=net(x,mask,duration_target=torch.ones(1,3,dtype=torch.long),p_control=.5,e_control=1.)
                        two=net(x,mask,duration_target=torch.ones(1,3,dtype=torch.long),p_control=.5,e_control=2.)
                    self.assertEqual(energy.call_args.args[-1],2.)
                    torch.testing.assert_close(two[2],one[2]*2)
                    torch.testing.assert_close(two[1],one[1])

    def test_configs_reconstruct_from_features_without_existing_config_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve();fixture(root)
            destination=root/'data/rebuild-config/learned-pauses-v1/English'
            before={p.name:p.read_bytes() for p in destination.glob('*.yaml')}
            for p in destination.glob('*.yaml'):p.unlink()
            destination.rmdir();write_configs(root)
            self.assertEqual(before,{p.name:p.read_bytes() for p in destination.glob('*.yaml')})
            model=yaml.safe_load((destination/'model.yaml').read_text())
            self.assertEqual(model['learned_pause']['train_pause_pos_weight'],1.)

    def test_worse_validation_preserves_best_and_missing_best_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve();directory=root/'english';directory.mkdir()
            t=dict(directory=directory,model=torch.nn.Linear(2,2),step=1,language='english',configs=[],best_validation=float('inf'))
            self.assertTrue(select_best(t,1.,None,False,root,runner.atomic_json))
            before=(directory/'best.pth.tar').read_bytes()
            t.update(step=2,best_validation=float('inf'),best_checkpoint=None)
            self.assertFalse(select_best(t,5.,None,False,root,runner.atomic_json))
            self.assertEqual(before,(directory/'best.pth.tar').read_bytes())
            (directory/'metrics.jsonl').write_text('\n'.join(json.dumps(dict(phase='validation',step=i,losses=[loss],complete=True)) for i,loss in [(1,1.),(2,5.),(3,4.)]))
            state=restore_selection(directory,{})
            self.assertEqual(state['best_validation'],1.);self.assertEqual(state['stale_validations'],2)
            (directory/'best.pth.tar').unlink()
            with self.assertRaisesRegex(ValueError,'missing'):restore_selection(directory,dict(best_validation=1.))

    def test_fixed_adam_restores_moments_and_next_update(self):
        torch.manual_seed(1);a=torch.nn.Linear(2,1);cfg={'optimizer':{'betas':[.9,.98],'eps':1e-9,'weight_decay':0}}
        oa=runner.FixedLearningRateOptim(a,cfg,2e-5)
        def update(net,opt):opt.zero_grad();net(torch.ones(1,2)).square().sum().backward();opt.step_and_update_lr()
        update(a,oa);b=copy.deepcopy(a);ob=runner.FixedLearningRateOptim(b,cfg,2e-5)
        ob.load_state_dict(copy.deepcopy(oa._optimizer.state_dict()));update(a,oa);update(b,ob)
        for x,y in zip(a.parameters(),b.parameters()):torch.testing.assert_close(x,y,rtol=0,atol=0)
        self.assertEqual(ob._optimizer.param_groups[0]['lr'],2e-5)


class RunnerRecoveryTests(unittest.TestCase):
    def test_retained_stop_rejects_explicit_and_implicit_resume_without_writes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve();fixture(root)
            self.run_training(root,['--run-id','stopped','--fresh','--max-steps','2'])
            folder=root/'output/runs/stopped';marker=folder/'session/STOP';marker.touch()
            before={str(p.relative_to(folder)):hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in folder.rglob('*') if p.is_file()}
            for mode in (['--resume'],[]):
                with contextlib.redirect_stderr(io.StringIO()) as errors:
                    with self.assertRaises(SystemExit) as failure:
                        self.run_training(root,['--run-id','stopped',*mode,'--max-steps','4'])
                self.assertEqual(failure.exception.code,2)
                self.assertIn('Retained STOP marker',errors.getvalue())
                self.assertIn('remove this marker explicitly',errors.getvalue())
                self.assertEqual(before,{str(p.relative_to(folder)):hashlib.sha256(p.read_bytes()).hexdigest()
                                         for p in folder.rglob('*') if p.is_file()})
            marker.unlink()  # Explicitly release only this synthetic test's stop request.
            self.run_training(root,['--run-id','stopped','--resume','--max-steps','4'])
            self.assertEqual(torch.load(folder/'english/latest.pth.tar',map_location='cpu',weights_only=False)['step'],4)

    def test_new_stop_during_training_still_saves_gracefully(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve();fixture(root);original=runner.atomic_json
            def request_stop(path,value):
                original(path,value)
                progress=value.get('languages',{}).get('english',{}).get('train',{})
                if progress.get('step')==1:
                    (path.parent/'STOP').touch()
            with patch.object(runner,'atomic_json',side_effect=request_stop):
                self.run_training(root,['--run-id','stop-during-run','--fresh','--max-steps','4'])
            folder=root/'output/runs/stop-during-run'
            self.assertEqual(torch.load(folder/'english/latest.pth.tar',map_location='cpu',weights_only=False)['step'],1)
            self.assertEqual(json.loads((folder/'session/status.json').read_text())['stop_reason'],'stop_requested')

    def run_training(self,root,args):
        cwd=Path.cwd();handlers={s:signal.getsignal(s) for s in (signal.SIGTERM,signal.SIGINT)}
        try:
            with patch.object(runner,'ROOT',root),contextlib.redirect_stdout(io.StringIO()):
                runner.main(['--language','english','--device','cpu','--batch-size','2',
                             '--checkpoint-every','1','--validation-every','1','--minutes','1',*args])
        finally:
            os.chdir(cwd)
            for s,h in handlers.items():signal.signal(s,h)

    def check_mode(self,mode,legacy=False):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve();pre,mod,cfg=fixture(root);initial=[]
            if mode!='base':
                from model import FastSpeech2
                adjusted=copy.deepcopy(pre);adjusted['path']['preprocessed_path']=str(root/pre['path']['preprocessed_path'])
                parent=FastSpeech2(adjusted,mod);folder=root/'output/initialization';folder.mkdir(parents=True)
                path=folder/'parent.pth.tar'
                historical_pre=copy.deepcopy(pre)
                if legacy:
                    historical_pre['path']['lexicon_path']=historical_pre['path']['lexicon_path'].removeprefix('English/')
                torch.save(dict(model=parent.state_dict(),step=12,language='english',configs=[historical_pre,mod,cfg]),path)
                initial=['--initialize-from','output/initialization/parent.pth.tar','--initialize-sha256',hashlib.sha256(path.read_bytes()).hexdigest(),
                         '--learning-rate','2e-5','--early-stop-patience','20']
                if mode=='pause':initial+=['--learned-pauses','--config-root','data/rebuild-config/learned-pauses-v1']
            self.run_training(root,['--run-id',mode,'--fresh','--max-steps','2',*initial])
            folder=root/'output/runs'/mode/'english'
            if legacy:
                path=folder/'latest.pth.tar';saved=torch.load(path,map_location='cpu',weights_only=False)
                for name in ('training_policy','best_validation','best_checkpoint','stale_validations','last_validation_step'):
                    saved.pop(name,None)
                torch.save(saved,path)
            original=(folder/'run-config.json').read_bytes()
            # Resume must not require the original parent checkpoint or generated YAMLs.
            if mode!='base':(root/'output/initialization/parent.pth.tar').unlink()
            if mode=='pause':
                for path in (root/'data/rebuild-config/learned-pauses-v1/English').glob('*.yaml'):path.unlink()
            self.run_training(root,['--run-id',mode,'--resume','--max-steps','4'])
            latest=torch.load(folder/'latest.pth.tar',map_location='cpu',weights_only=False)
            self.assertEqual(latest['step'],4);self.assertEqual(original,(folder/'run-config.json').read_bytes())
            self.assertTrue(all(int(s['step'])==4 for s in latest['optimizer']['state'].values()))
            groups=latest['optimizer']['param_groups']
            if mode!='base':
                self.assertEqual(groups[0]['lr'],2e-5)
                if mode=='pause':self.assertEqual(groups[1]['lr'],2e-4)
            else:
                self.assertEqual(latest['training_policy']['optimizer_type'],'scheduled')
                self.assertAlmostEqual(groups[0]['lr'],8**-.5*4*4000**-1.5)
            self.assertEqual(latest['model_type'],'learned_pause_v1' if mode=='pause' else 'fastspeech2')
            self.assertEqual(len((folder/'resume-history.jsonl').read_text().splitlines()),1)
            rows=[json.loads(s) for s in (folder/'metrics.jsonl').read_text().splitlines()]
            self.assertEqual(sum(r['phase']=='validation' and r['step']==0 for r in rows),int(mode!='base'))
            self.assertEqual(latest['best_validation'],min(r['losses'][0] for r in rows if r['phase']=='validation' and r.get('complete')))
            stale_steps={r['step'] for r in rows if r['phase']=='validation' and r.get('complete')
                         and not r.get('final',False) and r['step']>latest['best_checkpoint']['step']}
            if mode!='base':self.assertEqual(latest['stale_validations'],len(stale_steps))

    def test_base_interruption(self):self.check_mode('base')
    def test_fixed_rate_interruption(self):self.check_mode('fixed')
    def test_pause_interruption(self):self.check_mode('pause')
    def test_legacy_fixed_rate_interruption(self):self.check_mode('fixed',legacy=True)
    def test_legacy_pause_interruption(self):self.check_mode('pause',legacy=True)


if __name__=='__main__':unittest.main()
