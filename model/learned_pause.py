"""FastSpeech 2 with text-derived boundary conditioning and learned pauses."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .fastspeech2 import FastSpeech2
from .modules import VariancePredictor
from utils.tools import get_mask_from_lengths


class LearnedPauseFastSpeech2(FastSpeech2):
    def __init__(self, pre, config):
        super().__init__(pre, config)
        spec = config['learned_pause']
        self.max_pause_frames = int(spec['train_max_pause_frames'])
        self.boundary_embedding = nn.Embedding(7, config['transformer']['encoder_hidden'], padding_idx=0)
        nn.init.zeros_(self.boundary_embedding.weight)
        self.pause_presence = VariancePredictor(config)
        self.pause_duration = VariancePredictor(config)
        nn.init.zeros_(self.pause_presence.linear_layer.weight)
        prior = spec['train_pause_prior']
        nn.init.constant_(self.pause_presence.linear_layer.bias, math.log(prior/(1-prior)))
        nn.init.zeros_(self.pause_duration.linear_layer.weight)
        nn.init.constant_(self.pause_duration.linear_layer.bias, spec['train_mean_positive_log_duration'])

    def load_parent(self, state):
        incompatible = self.load_state_dict(state, strict=False)
        allowed = ('boundary_embedding.', 'pause_presence.', 'pause_duration.')
        if incompatible.unexpected_keys or any(not k.startswith(allowed) for k in incompatible.missing_keys):
            raise ValueError('Parent checkpoint differs outside the new boundary/pause modules')
        return list(incompatible.missing_keys)

    def forward(self, speakers, texts, src_lens, max_src_len, mels=None,
                mel_lens=None, max_mel_len=None, p_targets=None, e_targets=None,
                d_targets=None, p_control=1., e_control=1., d_control=1.,
                boundary_types=None, return_aux=False):
        if boundary_types is None or boundary_types.shape != texts.shape:
            raise ValueError('Learned-pause model requires one boundary type per input token')
        if d_control != 1. or p_control != 1. or e_control != 1.:
            raise ValueError('Learned-pause evaluation uses unmodified model predictions')
        src_mask = get_mask_from_lengths(src_lens, max_src_len)
        boundary_mask = (boundary_types > 0) & ~src_mask
        positions = torch.arange(texts.shape[1],device=texts.device)[None]
        pause_mask = boundary_mask & (positions < src_lens[:,None]-1)
        x = self.encoder(texts, src_mask, extra_embeddings=self.boundary_embedding(boundary_types))
        if self.speaker_emb is not None:
            x = x + self.speaker_emb(speakers)[:,None,:]
        logits = self.pause_presence(x, src_mask)
        pause_log_duration = self.pause_duration(x, src_mask)
        used_duration = d_targets
        if used_duration is None:
            speech_log_duration = self.variance_adaptor.duration_predictor(x, src_mask)
            speech_durations = torch.clamp(torch.round(torch.expm1(speech_log_duration)),min=1).long()
            predicted_pause = torch.clamp(torch.round(torch.expm1(torch.clamp(
                pause_log_duration, max=math.log1p(self.max_pause_frames)))), min=1).long()
            predicted_pause = predicted_pause * (torch.sigmoid(logits) >= .5) * pause_mask
            used_duration = torch.where(boundary_mask,predicted_pause,speech_durations).masked_fill(src_mask,0)
            mel_lens = used_duration.sum(1)
            max_mel_len = int(mel_lens.max())
            if not 0 < max_mel_len <= 6000:
                raise ValueError('Predicted utterance too long or empty')
        mel_mask = get_mask_from_lengths(mel_lens,max_mel_len) if mel_lens is not None else None
        variance_mask = src_mask | (boundary_mask & (used_duration == 0))
        x, pitch, energy, log_duration, durations, predicted_lens, mel_mask = self.variance_adaptor(
            x,variance_mask,mel_mask,max_mel_len,p_targets,e_targets,used_duration,1.,1.,1.)
        x,mel_mask = self.decoder(x,mel_mask)
        mel = self.mel_linear(x); postnet = mel+self.postnet(mel)
        result = (mel,postnet,pitch,energy,log_duration,durations,src_mask,mel_mask,src_lens,predicted_lens)
        aux = {'pause_logits':logits,'pause_log_duration':pause_log_duration,
               'pause_mask':pause_mask,'boundary_mask':boundary_mask}
        return (result,aux) if return_aux else result


class LearnedPauseLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.register_buffer('pos_weight',torch.tensor(config['learned_pause']['train_pause_pos_weight']))

    def forward(self, batch, pred, boundary_types, aux):
        target_mel, _, _, target_pitch, target_energy, target_duration = batch[6:]
        valid = ~pred[6]; mel_valid = ~pred[7]
        if pred[0].shape != target_mel.shape:
            raise ValueError('Full acoustic targets must be preserved')
        acoustic = valid & (target_duration > 0)
        speech = valid & (boundary_types == 0)
        pause = aux['pause_mask']; positive = pause & (target_duration > 0)
        mel = F.l1_loss(pred[0][mel_valid],target_mel[mel_valid])
        post = F.l1_loss(pred[1][mel_valid],target_mel[mel_valid])
        pitch = F.mse_loss(pred[2][acoustic],target_pitch[acoustic])
        energy = F.mse_loss(pred[3][acoustic],target_energy[acoustic])
        duration = F.mse_loss(pred[4][speech],torch.log1p(target_duration.float())[speech])
        presence = (F.binary_cross_entropy_with_logits(aux['pause_logits'][pause],
                    (target_duration[pause]>0).float(),pos_weight=self.pos_weight) if pause.any()
                    else aux['pause_logits'].sum()*0)
        pause_duration = (F.mse_loss(aux['pause_log_duration'][positive],torch.log1p(target_duration.float())[positive])
                          if positive.any() else aux['pause_log_duration'].sum()*0)
        total = mel+post+pitch+energy+duration+.25*presence+.25*pause_duration
        return total,mel,post,pitch,energy,duration,presence,pause_duration


def pause_metrics(target_duration, aux):
    mask = aux['pause_mask']; actual = target_duration > 0
    predicted = torch.sigmoid(aux['pause_logits']) >= .5
    positive = actual & mask
    frame_error = (torch.expm1(aux['pause_log_duration'][positive])-target_duration[positive]).abs()
    return {'tp':int((predicted & actual & mask).sum()),'fp':int((predicted & ~actual & mask).sum()),
            'fn':int((~predicted & actual & mask).sum()),'positive_boundaries':int(positive.sum()),
            'positive_duration_abs_frame_error':float(frame_error.sum())}
