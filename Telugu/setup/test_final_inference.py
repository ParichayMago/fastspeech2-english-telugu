"""Assignment-specific frontend contracts; these do not rate speech quality."""
import hashlib
import json
from pathlib import Path
import unittest
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from final_text_frontend import TextFrontend,english_spoken_form,telugu_spoken_form,parent_english_inputs


class FinalFrontendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest=json.loads((ROOT/'Telugu/setup/assignment_test_sentences.json').read_text())
        cls.cases={s['id']:s for s in cls.manifest['samples']}
        cls.en=TextFrontend('english',ROOT/'English/resources')
        cls.te=TextFrontend('telugu',ROOT/'Telugu/resources')

    def test_exact_assignment_ids_and_intentionally_split_source_words(self):
        self.assertEqual(set(self.cases),{'E1','E2a','E2b','E3','E4','T1','T2a','T2b','T3','T4'})
        self.assertIn('rhyth m',self.cases['E3']['text'])
        self.assertIn('అడ విలో',self.cases['T3']['text'])
        source=ROOT/self.manifest['source']
        # The emailed instructions are not redistributed with the public code.
        # Their transcription and fingerprint are retained in the manifest.
        if source.exists():
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(),self.manifest['source_sha256'])

    def test_conduct_stress_depends_on_sentence_context(self):
        a=self.en.encode(self.cases['E2a']['text']);b=self.en.encode(self.cases['E2b']['text'])
        noun=next(p for p in a['pronunciations'] if p['word']=='conduct')
        verb=next(p for p in b['pronunciations'] if p['word']=='conduct')
        self.assertEqual(noun['phones'],['K','AA1','N','D','AH0','K','T'])
        self.assertEqual(verb['phones'],['K','AH0','N','D','AH1','K','T'])
        self.assertEqual(a['source_text'],self.cases['E2a']['text'])

    def test_parent_projection_preserves_lexical_inputs_and_does_not_mutate_child(self):
        for key in ('E1','E2a','E2b','E3','E4'):
            child=self.en.encode(self.cases[key]['text']);before=json.dumps(child,sort_keys=True)
            parent=parent_english_inputs(child)
            self.assertEqual(json.dumps(child,sort_keys=True),before)
            self.assertEqual(parent['pronunciations'],child['pronunciations'])
            self.assertEqual(parent['normalized_words'],child['normalized_words'])
            self.assertEqual(parent['phones'],[p for p,k in zip(child['phones'],child['boundary_types']) if not k])
            self.assertFalse(any(parent['boundary_types']))
            self.assertEqual(len(parent['sequence']),len(parent['phones']))
            for span,pron in zip(parent['word_spans'],parent['pronunciations']):
                self.assertEqual(parent['phones'][span['start']:span['end']],pron['phones'])

    def test_long_word_is_not_cut_at_neural_g2p_limit(self):
        encoded=self.en.encode(self.cases['E1']['text'])
        pron=next(p for p in encoded['pronunciations'] if p['word']=='antidisestablishmentarianism')
        self.assertGreater(len(pron['phones']),20)
        self.assertEqual(pron['source'],'CMU_dictionary')
        split=self.en.encode(self.cases['E3']['text'])
        self.assertEqual(split['normalized_words'][1:3],['rhyth','m'])

    def test_number_normalization_preserves_original_text_and_phone_inventory(self):
        a=self.en.encode(self.cases['E4']['text']);b=self.te.encode(self.cases['T4']['text'])
        for case,encoded in ((self.cases['E4'],a),(self.cases['T4'],b)):
            self.assertEqual(encoded['source_text'],case['text'])
            self.assertEqual(len(encoded['sequence']),len(encoded['phones']))
            self.assertFalse(any(any(c.isdigit() for c in w) for w in encoded['normalized_words']))
        self.assertIn('twelfth',a['normalized_words']);self.assertIn('andaroo',a['normalized_words'])
        self.assertIn('పన్నెండవ',b['normalized_words']);self.assertIn('వేల',b['normalized_words'])
        self.assertNotIn('వేయిల',b['normalized_words'])
        self.assertTrue(set(b['phones']).issubset(set(json.loads((ROOT/'Telugu/text/assignment_phones.json').read_text()))))
        split=self.te.encode(self.cases['T3']['text']);self.assertEqual(split['normalized_words'][1:3],['అడ','విలో'])


    def test_telugu_decimal_preserves_point_digits_zeros_and_sign(self):
        cases={
            'ధర 3.14':'ధర మూడు బిందువు ఒకటి నాలుగు',
            'ధర 3.140':'ధర మూడు బిందువు ఒకటి నాలుగు సున్న',
            'ధర 0.05':'ధర సున్న బిందువు సున్న అయిదు',
            'ధర .5':'ధర సున్న బిందువు అయిదు',
            'ధర -3.14':'ధర మైనస్ మూడు బిందువు ఒకటి నాలుగు',
            'ధర +3.14':'ధర ప్లస్ మూడు బిందువు ఒకటి నాలుగు',
            'ధర −.05':'ధర మైనస్ సున్న బిందువు సున్న అయిదు',
            'ధర ౩.౧౪':'ధర మూడు బిందువు ఒకటి నాలుగు',
        }
        for text,expected in cases.items():
            with self.subTest(text=text):
                spoken,changes=telugu_spoken_form(text)
                self.assertEqual(spoken,expected)
                decimal=next(c for c in changes if c['kind']=='decimal_number')
                self.assertEqual(decimal['fraction_digits'],text.split('.')[-1])
        spoken,_=telugu_spoken_form('ధర 3.14.')
        self.assertIn('బిందువు ఒకటి నాలుగు',spoken)

    def test_telugu_unsupported_decimal_formats_fail_before_integer_fallback(self):
        for text in ('ధర 3.14.15','ధర 3..14','ధర 1,234.56','ధర 3.14,15','ధర 3.14వ'):
            with self.subTest(text=text),self.assertRaises(ValueError):telugu_spoken_form(text)

    def test_telugu_decimal_frontend_preserves_source_and_uses_trained_phones(self):
        text='ధర 3.14'
        encoded=self.te.encode(text)
        self.assertEqual(encoded['source_text'],text)
        self.assertEqual(encoded['normalized_words'],['ధర','మూడు','బిందువు','ఒకటి','నాలుగు'])
        self.assertEqual(len(encoded['sequence']),len(encoded['phones']))
        self.assertTrue(all(span['end']>span['start'] for span in encoded['word_spans']))


if __name__=='__main__':unittest.main()
