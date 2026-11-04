"""Punctuation-aware English preview planning; no training-data changes."""
from dataclasses import dataclass
import re

from setup.prepare_assignment_corpus import normalize_text

PAUSES = {',': .14, ':': .24, ';': .24, '.': .40, '!': .45, '?': .45}
ABBREVIATIONS = {'mr', 'mrs', 'ms', 'dr', 'prof', 'sr', 'jr', 'vs', 'st'}


@dataclass(frozen=True)
class Chunk:
    source_text: str
    normalized_text: str
    punctuation: str
    pause_after_seconds: float


def plan_utterance(text, max_words=30):
    """Keep word order; put deterministic pauses between punctuation clauses.

    Punctuation marks are not new untrained phoneme IDs. Each clause uses the
    existing English/model/frontend, followed by explicitly recorded waveform silence.
    Numeric separators, contractions and word-internal hyphens stay intact.
    """
    if max_words < 1:
        raise ValueError('max_words must be positive')
    spans, start = [], 0
    for match in re.finditer(r'[.!?,;:]+(?=\s|$)', text):
        punctuation = match.group()
        before = text[:match.end()]
        last_word = re.search(r'([A-Za-z]+)\.$', before)
        if punctuation == '.' and (last_word and last_word[1].lower() in ABBREVIATIONS
                                    or re.search(r'(?:[A-Za-z]\.){2,}$', before)):
            continue
        spans.append((text[start:match.end()].strip(), punctuation))
        start = match.end()
    if text[start:].strip():
        spans.append((text[start:].strip(), ''))
    chunks = []
    for source, punctuation in spans:
        normalized = normalize_text(source, 'english')
        words = normalized.split()
        for first in range(0, len(words), max_words):
            segment = words[first:first + max_words]
            last = first + max_words >= len(words)
            pause = max((PAUSES.get(p, 0) for p in punctuation), default=0) if last else .16
            chunks.append(Chunk(source if len(words) <= max_words else ' '.join(segment),
                                ' '.join(segment), punctuation if last else '', pause))
    if not chunks:
        raise ValueError('No pronounceable English text')
    final = chunks[-1]
    chunks[-1] = Chunk(final.source_text, final.normalized_text, final.punctuation, 0)
    if ' '.join(c.normalized_text for c in chunks) != normalize_text(text, 'english'):
        raise ValueError('Chunking changed normalized word content')
    return chunks


def scaled_durations(log_duration, scale):
    """Scale before rounding; retain every spoken phone with at least one frame."""
    import torch
    if not .5 <= scale <= 2:
        raise ValueError('Duration scale must be between 0.5 and 2')
    if not torch.isfinite(log_duration).all():
        raise ValueError('Non-finite duration prediction')
    return torch.clamp(torch.round(torch.expm1(log_duration) * scale), min=1).long()


def assemble_audio(waves, chunks, sample_rate, fade_seconds=.005):
    """Add exact inter-clause silence and short click-preventing boundary fades.

    There is no filtering, noise estimation, denoising or loudness normalization.
    """
    import numpy as np
    if len(waves) != len(chunks) or not waves:
        raise ValueError('One waveform per chunk is required')
    pieces, gaps = [], []
    for wave, chunk in zip(waves, chunks):
        wave = np.asarray(wave, dtype=np.float32).copy()
        if wave.ndim != 1 or not len(wave) or not np.isfinite(wave).all():
            raise ValueError('Invalid generated audio')
        n = min(round(sample_rate * fade_seconds), len(wave)//2)
        if n:
            ramp = np.linspace(0, 1, n, dtype=np.float32)
            wave[:n] *= ramp; wave[-n:] *= ramp[::-1]
        gap = round(sample_rate * chunk.pause_after_seconds)
        pieces.append(wave)
        if gap:
            pieces.append(np.zeros(gap, dtype=np.float32))
        gaps.append(gap)
    return np.concatenate(pieces), gaps
