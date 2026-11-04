"""Checks for leakage, duration targeting, and preserving Telugu text distinctions."""
import unittest
from prepare_splits import canonical_text, choose_groups, flatten, make_groups, verify_disjoint


def example(identity, text, audio, recording, seconds=2):
    return {'sample_id': identity, 'text': text, 'audio_sha256': audio,
            'audio_path': recording + '_chunk_' + identity + '.wav',
            'duration_seconds': seconds}


class SplitTests(unittest.TestCase):
    def test_transitive_leakage_connections_stay_together(self):
        rows = [example('a', 'hello', 'audio1', 'recording1'),
                example('b', 'different', 'audio1', 'recording2'),
                example('c', 'Other phrase!', 'audio3', 'recording2'),
                example('d', 'other PHRASE', 'audio4', 'recording4'),
                example('e', 'separate', 'audio5', 'recording5')]
        groups = make_groups(rows, 'english')
        self.assertEqual(sorted(len(g['rows']) for g in groups), [1, 4])
        self.assertTrue(any({r['sample_id'] for r in g['rows']} == {'a','b','c','d'} for g in groups))

    def test_telugu_vowel_marks_are_not_discarded(self):
        self.assertNotEqual(canonical_text('కల'), canonical_text('కాల'))
        self.assertEqual(canonical_text('  Hello, WORLD! '), 'hello world')

    def test_overlap_is_rejected_even_when_sample_ids_differ(self):
        train = [example('a', 'A shared sentence.', 'audio1', 'recording1')]
        validation = [example('b', 'a shared sentence', 'audio2', 'recording2')]
        with self.assertRaisesRegex(ValueError, 'text'):
            verify_disjoint(train, validation, 'english')

    def test_duration_selection_is_deterministic_and_atomic(self):
        rows = [example(str(i), str(i), str(i), str(i), i + 1) for i in range(20)]
        groups = make_groups(rows, 'english')
        picked, rest = choose_groups(groups, 60, 42)
        again, _ = choose_groups(groups, 60, 42)
        self.assertEqual([g['id'] for g in picked], [g['id'] for g in again])
        self.assertLessEqual(abs(sum(g['seconds'] for g in picked)-60), 1)
        self.assertFalse({g['id'] for g in picked} & {g['id'] for g in rest})
        self.assertEqual(len(flatten(picked)) + len(flatten(rest)), len(rows))


if __name__ == '__main__':
    unittest.main()
