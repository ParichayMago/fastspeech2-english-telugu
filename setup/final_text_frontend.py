"""Auditable text-only normalization and G2P for the frozen submission models.

No waveform edits, injected silence, duration targets, or pitch/energy controls.
The exact source string remains separate from the normalized spoken form.
"""
import json
import os
from pathlib import Path
import re
import unicodedata

ROOT = Path(__file__).resolve().parent.parent
NLTK_DIR = ROOT/'data/inference/nltk_data'
os.environ.setdefault('NLTK_DATA', str(NLTK_DIR))
from setup.learned_pause_frontend import words_and_types


def parent_english_inputs(encoded):
    """Keep common lexical pronunciations but omit the child's new pause slots.

    This is a system comparison, not a controlled ablation of just the pause head:
    the parent was trained before text-derived boundary slots were introduced.
    """
    import copy
    from text import text_to_sequence
    result=copy.deepcopy(encoded)
    keep=[i for i,kind in enumerate(encoded['boundary_types']) if not kind]
    result['phones']=[encoded['phones'][i] for i in keep]
    result['boundary_types']=[0]*len(keep)
    for span in result['word_spans']:
        start,end=span['start'],span['end']
        span['start']=sum(i<start for i in keep);span['end']=sum(i<end for i in keep)
    result['sequence']=text_to_sequence('{'+' '.join(result['phones'])+'}',[])
    if len(result['sequence'])!=len(result['phones']):raise ValueError('Untrained parent phone')
    result['normalization_note']+='; parent representation omits learned-pause boundary slots'
    return result


def english_spoken_form(text):
    from num2words import num2words
    changes = []

    def clock(m):
        h, minute, period = int(m[1]), int(m[2]), m[3]
        if not 0 <= h <= 23 or not 0 <= minute <= 59:
            raise ValueError('Invalid clock time: '+m[0])
        spoken = num2words(h)
        if minute:
            spoken += ' '+('oh ' if minute < 10 else '')+num2words(minute)
        if period:
            spoken += ' '+' '.join(period.lower().replace('.',''))
        changes.append({'source':m[0], 'spoken':spoken, 'kind':'clock_time'})
        return spoken

    expanded = re.sub(r'\b(\d{1,2}):(\d{2})(?:\s*([AP]M))?\b',clock,text,flags=re.I)
    months = '(?:January|February|March|April|May|June|July|August|September|October|November|December)'
    def date(m):
        day = int(m[1])
        if not 1 <= day <= 31:
            raise ValueError('Invalid date day')
        spoken = num2words(day,to='ordinal')+' '+m[2]
        changes.append({'source':m[0],'spoken':spoken,'kind':'day_before_month'})
        return spoken
    expanded = re.sub(r'\b(\d{1,2})\s+('+months+r')\b',date,expanded,flags=re.I)
    return expanded, changes


def telugu_spoken_form(text):
    from num2words import num2words
    if re.search(r'[A-Za-z]',text):
        raise ValueError('Telugu inference expects Telugu script; no silent translation of Latin words')
    changes = []
    def cardinal(value):
        # num2words 0.5.14 uses the nonstandard "వేయిల" for this scale.
        return unicodedata.normalize('NFC',num2words(int(value),lang='te').replace('వేయిల','వేల'))
    def time(m):
        h,minute = int(m[1]),int(m[2])
        if not 0 <= h <= 23 or not 0 <= minute <= 59:
            raise ValueError('Invalid clock time: '+m[0])
        spoken = cardinal(h)+' గంటల '+cardinal(minute)+' నిమిషాలకు'
        changes.append({'source':m[0],'spoken':spoken,'kind':'clock_with_telugu_unit'})
        return spoken
    expanded = re.sub(r'\b(\d{1,2}):(\d{2})\s+గంటలకు',time,text)
    def ordinal(m):
        if int(m[1]) != 12:
            raise ValueError('Unsupported Telugu numeric ordinal; spell it out explicitly')
        # Correctly inflected date ordinal; library ordinal is పన్నెండువ.
        spoken = 'పన్నెండవ'
        changes.append({'source':m[0],'spoken':spoken,'kind':'reviewed_telugu_date_ordinal'})
        return spoken
    expanded = re.sub(r'(\d+)వ(?=\s|$|[.,])',ordinal,expanded)
    def number(m):
        spoken=cardinal(m[0]);changes.append({'source':m[0],'spoken':spoken,'kind':'cardinal_number'})
        return spoken
    expanded = re.sub(r'\d+',number,expanded)
    return expanded,changes


class TextFrontend:
    def __init__(self, language, resources):
        self.language=language;self.resources=Path(resources)
        self.lexicon={p[0]:p[1:] for p in [l.split() for l in
            (self.resources/'lexicon.tsv').read_text().splitlines()] if p}
        self.g2p=None;self.phonemizer=None;self.espeak_version=None

    def english(self,text):
        spoken,changes=english_spoken_form(text)
        words,types=words_and_types(spoken)
        import nltk
        nltk.data.path.insert(0,str(NLTK_DIR))
        try:
            tags=nltk.pos_tag(words)
            if self.g2p is None:
                from g2p_en import G2p
                self.g2p=G2p()
        except LookupError as exc:
            raise RuntimeError('Run python setup/prepare_final_inference.py to prepare project-scoped NLTK resources') from exc
        phones=[];boundaries=[];word_spans=[];pronunciations=[]
        for word,kind,(_,tag) in zip(words,types,tags):
            if word=='conduct':
                # CMU stress contrast selected by sentence POS, never by sample ID.
                pron=(['K','AH0','N','D','AH1','K','T'] if tag.startswith('VB')
                      else ['K','AA1','N','D','AH0','K','T'])
                origin='contextual_conduct_verb' if tag.startswith('VB') else 'contextual_conduct_noun'
            elif word=='andaroo':
                pron=['AH0','N','D','AH0','R','UW1']
                origin='reviewed_code_mixed_Telugu_andaroo_ARPAbet_approximation'
            elif word in self.g2p.homograph2features:
                a,b,pos=self.g2p.homograph2features[word];pron=a if tag.startswith(pos) else b
                origin='g2p_en_contextual_homograph'
            elif word in self.lexicon:
                pron=self.lexicon[word];origin='frozen_lexicon'
            elif word in self.g2p.cmu:
                pron=self.g2p.cmu[word][0];origin='CMU_dictionary'
            else:
                pron=self.g2p.predict(word);origin='g2p_en_neural_OOV'
                if len(pron)==20:
                    raise ValueError('G2P may have reached its 20-phone limit for '+word)
            if not pron:
                raise ValueError('Empty pronunciation for '+word)
            start=len(phones);phones.extend(pron);boundaries.extend([0]*len(pron))
            word_spans.append({'word':word,'start':start,'end':len(phones)})
            phones.append('sp');boundaries.append(kind)
            pronunciations.append({'word':word,'POS':tag,'phones':pron,'source':origin})
        return {'source_text':text,'spoken_form_before_generic_normalization':spoken,
                'normalized_words':words,'phones':phones,'boundary_types':boundaries,
                'word_spans':word_spans,'normalization_changes':changes,'pronunciations':pronunciations,
                'normalization_note':'Literal split words retained; clock/date verbalization recorded separately'}

    def telugu(self,text):
        spoken,changes=telugu_spoken_form(text)
        normalized=' '.join(''.join(c if unicodedata.category(c)[0] in 'LM' else ' '
                                   for c in unicodedata.normalize('NFC',spoken)).split())
        words=normalized.split()
        missing=sorted(set(words)-self.lexicon.keys());generated={}
        if missing:
            import espeakng_loader
            os.environ['ESPEAK_DATA_PATH']=espeakng_loader.get_data_path()
            from phonemizer.backend.espeak.wrapper import EspeakWrapper
            EspeakWrapper.set_library(espeakng_loader.get_library_path())
            from phonemizer import phonemize
            from phonemizer.separator import Separator
            ipa_map=json.loads((self.resources/'ipa-to-phone.json').read_text())
            raw=phonemize(missing,language='te',backend='espeak',separator=Separator(
                phone=' ',word=' | ',syllable=''),strip=True,preserve_punctuation=False,
                with_stress=False,language_switch='remove-flags',njobs=1)
            if len(raw)!=len(missing):raise ValueError('Telugu G2P changed word count')
            for word,pron in zip(missing,raw):
                ipa=pron.replace('|',' ').split()
                unknown=set(ipa)-ipa_map.keys()
                if unknown:raise ValueError('Untrained Telugu IPA phones: '+repr(unknown))
                generated[word]=([ipa_map[p] for p in ipa],ipa)
            self.espeak_version='.'.join(map(str,EspeakWrapper().version))
        phones=[];spans=[];pronunciations=[]
        for word in words:
            pron=self.lexicon.get(word,generated.get(word,([],[]))[0])
            if not pron:raise ValueError('Empty Telugu pronunciation for '+word)
            start=len(phones);phones.extend(pron);spans.append({'word':word,'start':start,'end':len(phones)})
            pronunciations.append({'word':word,'phones':pron,
                'source':'frozen_training_lexicon' if word in self.lexicon else 'espeak_Telugu_OOV',
                **({'IPA':generated[word][1]} if word in generated else {})})
        return {'source_text':text,'spoken_form_before_generic_normalization':spoken,
                'normalized_words':words,'phones':phones,'boundary_types':[0]*len(phones),
                'word_spans':spans,'normalization_changes':changes,'pronunciations':pronunciations,
                'espeak_version_for_OOV':self.espeak_version,
                'normalization_note':'Visible split words retained; no spelling correction or waveform pacing'}

    def encode(self,text):
        if not text.strip():raise ValueError('Empty text')
        result=self.english(text) if self.language=='english' else self.telugu(text)
        from text import text_to_sequence,sequence_to_text
        sequence=text_to_sequence('{'+' '.join(result['phones'])+'}',[])
        if len(sequence)!=len(result['phones']):raise ValueError('A phoneme is outside the trained inventory')
        result['sequence']=sequence
        return result
