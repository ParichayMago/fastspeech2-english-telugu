import json
from pathlib import Path
import unittest

import numpy as np
import torch
import yaml

from setup.learned_pause_frontend import words_and_types, encode_words
from setup.learned_pause_data import LearnedPauseDataset
from model.learned_pause import LearnedPauseFastSpeech2, LearnedPauseLoss
from utils.tools import to_device

ROOT = Path(__file__).resolve().parent.parent


class LearnedPauseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        configs = ROOT/'data/rebuild-config/learned-pauses-v1/English'
        cls.pre,cls.mod,cls.cfg = [yaml.safe_load((configs/n).read_text()) for n in ['preprocess.yaml','model.yaml','train.yaml']]
        cls.cfg['optimizer']['batch_size']=2
        cls.dataset = LearnedPauseDataset('train.txt',cls.pre,cls.cfg)
        cls.parent = torch.load(ROOT/'output/runs/english-finetune-pilot-20261104/english/best.pth.tar',map_location='cpu',weights_only=False)

    def model(self):
        net=LearnedPauseFastSpeech2(self.pre,self.mod)
        missing=net.load_parent(self.parent['model'])
        self.assertTrue(missing)
        for key in self.parent['model']:
            self.assertTrue(torch.equal(net.state_dict()[key],self.parent['model'][key]),key)
        return net

    def test_boundary_inputs_are_text_derived_not_pause_labels(self):
        lexicon={'hello':['HH','EH1','L','OW0'],'world':['W','ER1','L','D']}
        words,types=words_and_types('Hello, world?')
        phones,b=encode_words(words,types,lexicon)
        self.assertEqual([x for x in b if x],[2,4])
        self.assertEqual(phones.count('sp'),2)
        self.assertEqual(words_and_types('Hello world')[1],[1,1])

    def test_all_saved_targets_preserve_acoustics_and_no_pause_examples(self):
        old=ROOT/'data/features/gap-fixed-v2/english';new=ROOT/self.pre['path']['preprocessed_path']
        positive=zero=0
        for filename in ('train.txt','val.txt'):
            source={r.split('|')[0]:r.split('|') for r in (old/filename).read_text().splitlines()}
            rows=[r.split('|') for r in (new/filename).read_text().splitlines()]
            self.assertEqual(set(source),{r[0] for r in rows})
            for sid,sp,ph,raw in rows:
                d=np.load(new/'duration'/f'{sp}-duration-{sid}.npy');b=np.load(new/'boundary'/f'{sp}-boundary-{sid}.npy')
                original=np.load(old/'duration'/f'{sp}-duration-{sid}.npy')
                self.assertEqual(int(d.sum()),int(original.sum()))
                self.assertEqual(len(d),len(ph.strip('{}').split()))
                words,types=words_and_types(raw)
                self.assertEqual(b[b>0].tolist(),types)
                mask=b>0;mask[-1]=False;positive+=int(((d>0)&mask).sum());zero+=int(((d==0)&mask).sum())
                self.assertTrue(np.isfinite(d).all())
        self.assertEqual(positive,8994+1751);self.assertEqual(zero,26255+5597)

    def test_real_batch_gradients_reach_both_new_heads_and_conditioning(self):
        net=self.model().train();samples=[self.dataset[i] for i in (1,2)]
        raw=self.dataset.reprocess(samples,[0,1]);batch=to_device(raw[:12],torch.device('cpu'));b=torch.from_numpy(raw[12]).long()
        pred,aux=net(*batch[2:],boundary_types=b,return_aux=True)
        self.assertEqual(pred[0].shape,batch[6].shape)
        losses=LearnedPauseLoss(self.mod)(batch,pred,b,aux)
        self.assertTrue(all(torch.isfinite(x).item() for x in losses));losses[0].backward()
        for layer in (net.pause_presence.linear_layer,net.pause_duration.linear_layer,net.boundary_embedding):
            self.assertIsNotNone(layer.weight.grad);self.assertGreater(float(layer.weight.grad.abs().sum()),0)

    def test_inference_duration_is_controlled_by_learned_pause_heads(self):
        net=self.model().eval();sample=self.dataset[0]
        # Trim to a short real word sequence with the same text-derived slots.
        seq=torch.from_numpy(sample['text'])[None].long();b=torch.from_numpy(sample['boundary'])[None].long()
        inputs=(torch.tensor([sample['speaker']]),seq,torch.tensor([seq.shape[1]]),seq.shape[1])
        with torch.no_grad():
            net.pause_presence.linear_layer.bias.fill_(-20)
            no_pause=net(*inputs,boundary_types=b)
            net.pause_presence.linear_layer.bias.fill_(20)
            net.pause_duration.linear_layer.bias.fill_(float(np.log1p(5)))
            yes_pause,aux=net(*inputs,boundary_types=b,return_aux=True)
        mask=aux['pause_mask']
        self.assertTrue(torch.all(no_pause[5][mask]==0))
        self.assertTrue(torch.all(yes_pause[5][mask]==5))
        self.assertEqual(int(yes_pause[5][0,-1]),0)
        self.assertEqual(int(yes_pause[9][0]-no_pause[9][0]),5*int(mask.sum()))
        self.assertEqual(yes_pause[1].shape[1],int(yes_pause[5].sum()))


if __name__=='__main__':
    unittest.main()
