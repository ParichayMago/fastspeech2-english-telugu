#!/usr/bin/env python3
"""Deterministic text normalization, lexicons and 22.05 kHz alignment corpora."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import unicodedata

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def normalize_text(text, language):
    text = unicodedata.normalize('NFC', text).replace('’', "'").replace('‘', "'")
    if language == 'english':
        from text.cleaners import english_cleaners
        return ' '.join(re.findall(r"[a-z]+(?:'[a-z]+)*", english_cleaners(text)))
    from num2words import num2words
    # Unicode decimal digits are converted to ASCII before the Telugu converter.
    text = ''.join(str(unicodedata.decimal(c)) if c.isdecimal() else c for c in text)
    text = re.sub(r'\d+(?:\.\d+)?', lambda m: ' ' + num2words(m[0],lang='te') + ' ', text)
    return ' '.join(''.join(c if unicodedata.category(c)[0] in 'LM' else ' '
                            for c in text).casefold().split())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--language', choices=['english','telugu'], required=True)
    parser.add_argument('--lexicons-only', action='store_true')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0,str(root)); os.chdir(root)
    language=args.language
    out=root/'data/preprocessing'/language
    out.mkdir(parents=True,exist_ok=True)
    manifests={}; vocabulary=set(); input_hash={}
    for split in ['train','validation']:
        path=root/'data/splits/v1'/language/(split+'-prepared.jsonl')
        input_hash[split]=hashlib.sha256(path.read_bytes()).hexdigest()
        manifests[split]=[]
        for line in path.read_text().splitlines():
            row=json.loads(line)
            normalized=normalize_text(row['text'],language)
            if not normalized:
                raise ValueError('Normalization removed entire transcript: '+row['sample_id'])
            vocabulary.update(normalized.split())
            manifests[split].append({**row,'normalized_text':normalized,'raw_text':row['text']})
    lexicon=out/'lexicon.txt'
    words=sorted(vocabulary)
    if language=='english':
        from g2p_en import G2p
        from text.cmudict import valid_symbols
        g2p=G2p(); valid=set(valid_symbols); entries={}
        for word in words:
            phones=[p for p in g2p(word) if p.strip() and p not in ",.!?;:'\"-"]
            if not phones or not set(phones)<=valid:
                raise ValueError('Invalid ARPAbet pronunciation for '+word+': '+repr(phones))
            entries[word]=phones
    else:
        from phonemizer import phonemize
        from phonemizer.separator import Separator
        pronunciations=phonemize(words,language='te',backend='espeak',
            separator=Separator(phone=' ',word=' | ',syllable=''),strip=True,
            preserve_punctuation=False,with_stress=False,language_switch='remove-flags',
            njobs=args.workers)
        if len(pronunciations)!=len(words):
            raise ValueError('Phonemizer dropped a vocabulary entry')
        entries={word:pron.replace('|',' ').split() for word,pron in zip(words,pronunciations)}
        if any(not phones for phones in entries.values()):
            raise ValueError('Empty Telugu pronunciation')
        # Use safe ASCII labels for MFA and the FastSpeech2 symbol vocabulary.
        ipa=sorted({p for phones in entries.values() for p in phones})
        # MFA interprets underscores/digit suffixes as phone annotations in some
        # model round trips. Pure lowercase letter IDs preserve phone identity.
        hex_to_letters=str.maketrans('0123456789abcdef','abcdefghijklmnop')
        phone_map={p:'telph'+'q'.join(f'{ord(c):x}'.translate(hex_to_letters)
                                    for c in p) for p in ipa}
        assert len(set(phone_map.values()))==len(phone_map)
        assert all(p.isascii() and p.isalpha() and p.islower() for p in phone_map.values())
        (out/'ipa-to-phone.json').write_text(json.dumps(phone_map,ensure_ascii=False,indent=2)+'\n')
        entries={word:[phone_map[p] for p in phones] for word,phones in entries.items()}
        (root/'text/assignment_phones.json').write_text(json.dumps(sorted(phone_map.values()),indent=2)+'\n')
    lexicon.write_text(''.join(word+'\t'+' '.join(entries[word])+'\n' for word in words))
    report={'language':language,'input_manifest_sha256':input_hash,
        'normalization':'English upstream english_cleaners + word tokenization; Telugu NFC, num2words 0.5.14, Unicode letter/mark word tokenization.',
        'pronunciations':'g2p-en 2.1.0 ARPAbet' if language=='english' else 'phonemizer 3.3.0 / eSpeak-NG Telugu; stress removed; safe ASCII phone IDs map back to IPA.',
        'vocabulary_size':len(words),'lexicon_sha256':hashlib.sha256(lexicon.read_bytes()).hexdigest(),
        'lexicon_validation_text_usage':'Deterministic G2P covers known validation words; no validation audio or learned statistics used to train a lexicon.',
        'target_sampling_rate':22050,'corpus_written':False,
        'limitations':['Automatic pronunciations and number expansions still need listening review.',
            'Original transcripts are retained; normalization is applied to a separate field.']}
    for split,rows in manifests.items():
        path=out/(split+'-normalized.jsonl')
        path.write_text(''.join(json.dumps(row,ensure_ascii=False,sort_keys=True)+'\n' for row in rows))
    if not args.lexicons_only:
        allrows=[(split,row) for split,rows in manifests.items() for row in rows]
        def export(item):
            split,row=item
            speaker=row.get('speaker_id')
            if not speaker:
                raise ValueError('Speaker label required: '+row['sample_id'])
            corpus=out/'corpus'/split/speaker
            corpus.mkdir(parents=True,exist_ok=True)
            wav,rate=sf.read(root/row['prepared_audio_path'],dtype='float32')
            ratio=Fraction(22050,rate)
            converted=resample_poly(wav,ratio.numerator,ratio.denominator).astype(np.float32)
            peak=float(np.abs(converted).max()); scaled=peak>.999
            if scaled:converted*=.999/peak
            wavpath=corpus/(row['sample_id']+'.wav')
            sf.write(wavpath,converted,22050,subtype='PCM_16')
            labpath=corpus/(row['sample_id']+'.lab')
            labpath.write_text(row['normalized_text']+'\n')
            return {'split':split,'sample_id':row['sample_id'],'speaker_id':speaker,
                    'samples':len(converted),'peak_scaled':scaled,
                    'wav':str(wavpath.relative_to(root)),
                    'sha256':hashlib.sha256(wavpath.read_bytes()).hexdigest()}
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            exported=list(pool.map(export,allrows))
        (out/'corpus-manifest.jsonl').write_text(''.join(json.dumps(row,sort_keys=True)+'\n' for row in exported))
        report.update(corpus_written=True,corpus_rows=len(exported),
            resampling='scipy.signal.resample_poly, no trimming; original WAV remains intact; scale only interpolation peaks exceeding .999.',
            interpolation_peak_scaled_rows=sum(row['peak_scaled'] for row in exported),
            split_hours={split:sum(row['samples'] for row in exported if row['split']==split)/22050/3600 for split in manifests})
    dest=root/'Results/Preprocessing';dest.mkdir(parents=True,exist_ok=True)
    (dest/(language+'-corpus.json')).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
