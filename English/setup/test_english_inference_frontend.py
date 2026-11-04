import unittest
import numpy as np
import torch

from english_inference_frontend import plan_utterance, scaled_durations, assemble_audio

PASSAGE = "i've reproduced the results from the MIT blog post today on weight-space programming. but one additional result i have to add: this is a new, step-change emerging capability that's appeared in just the latest set of models."


class FrontendTests(unittest.TestCase):
    def test_passage_words_and_punctuation_are_preserved(self):
        chunks = plan_utterance(PASSAGE)
        self.assertEqual(len(chunks), 4)
        self.assertEqual([c.punctuation for c in chunks], ['.', ':', ',', '.'])
        self.assertEqual([c.pause_after_seconds for c in chunks], [.4, .24, .14, 0])
        text = ' '.join(c.normalized_text for c in chunks)
        self.assertEqual(len(text.split()), 39)
        self.assertIn("i've", text); self.assertIn("that's", text)
        self.assertIn('weight space', text); self.assertIn('step change', text)

    def test_numbers_and_abbreviations_do_not_create_false_pauses(self):
        chunks = plan_utterance('Dr. Jones uses the U.S. clock at 10:45 and pays 1,000.50 dollars. Then leaves.')
        self.assertEqual(len(chunks), 2)
        self.assertTrue(chunks[0].source_text.startswith('Dr. Jones'))

    def test_long_unpunctuated_text_keeps_every_word(self):
        text = ' '.join(['hello'] * 75)
        chunks = plan_utterance(text)
        self.assertEqual([len(c.normalized_text.split()) for c in chunks], [30, 30, 15])
        self.assertEqual(' '.join(c.normalized_text for c in chunks), text)
        self.assertEqual(chunks[-1].pause_after_seconds, 0)

    def test_duration_scaling_occurs_before_rounding(self):
        raw = torch.log1p(torch.tensor([[1.3, 2.0, .1]]))
        self.assertEqual(scaled_durations(raw, 1.2).tolist(), [[2, 2, 1]])
        with self.assertRaises(ValueError):
            scaled_durations(torch.tensor([[float('nan')]]), 1.2)

    def test_waveform_pause_lengths_and_source_arrays(self):
        chunks = plan_utterance('Hello. World.')
        a = np.ones(2205, dtype=np.float32); b = a.copy()
        combined, gaps = assemble_audio([a, b], chunks, 22050)
        self.assertEqual(gaps, [8820, 0])
        self.assertEqual(len(combined), len(a) + len(b) + 8820)
        self.assertTrue(np.all(combined[2205:2205+8820] == 0))
        self.assertTrue(np.all(a == 1)); self.assertTrue(np.all(b == 1))
        self.assertEqual(combined[0], 0); self.assertEqual(combined[-1], 0)


if __name__ == '__main__':
    unittest.main()
