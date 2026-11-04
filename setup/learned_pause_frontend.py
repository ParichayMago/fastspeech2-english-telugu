"""Identical word-boundary inputs for learned-pause training and inference.

Only punctuation categories are encoded here. Pause presence/duration must be
predicted by the model; there are no pause lengths or waveform edits here.
"""
import re
from setup.prepare_assignment_corpus import normalize_text

BOUNDARY_TYPES = {'none': 1, 'comma': 2, 'period': 3, 'question': 4, 'exclamation': 5, 'clause': 6}
PUNCTUATION = {',': 2, '.': 3, '?': 4, '!': 5, ':': 6, ';': 6}
ABBREVIATIONS = {'mr', 'mrs', 'ms', 'dr', 'prof', 'sr', 'jr', 'vs', 'st'}


def words_and_types(text):
    words, types, start = [], [], 0
    for match in re.finditer(r'[.!?,;:]+(?=\s|$)', text):
        before = text[:match.end()]
        last = re.search(r'([A-Za-z]+)\.$', before)
        if match.group() == '.' and (last and last[1].lower() in ABBREVIATIONS
                                      or re.search(r'(?:[A-Za-z]\.){2,}$', before)):
            continue
        part = normalize_text(text[start:match.start()], 'english').split()
        if part:
            words.extend(part); types.extend([1] * len(part))
            types[-1] = next((PUNCTUATION[c] for c in '?!:;,.' if c in match.group()), 1)
        start = match.end()
    part = normalize_text(text[start:], 'english').split()
    words.extend(part); types.extend([1] * len(part))
    if not words or ' '.join(words) != normalize_text(text, 'english'):
        raise ValueError('Punctuation mapping changed normalized words')
    return words, types


def encode_words(words, types, lexicon):
    phones, boundaries = [], []
    for word, kind in zip(words, types):
        if word not in lexicon:
            raise ValueError('Unknown pronunciation: ' + word)
        phones.extend(lexicon[word]); boundaries.extend([0] * len(lexicon[word]))
        phones.append('sp'); boundaries.append(kind)
    return phones, boundaries
