"""Regression tests for omitted alignment silence that cut off speech tails."""
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace as Item
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from preprocessor.alignment import alignment_to_phones


def tier(rows):
    return Item(_objects=[Item(start_time=a,end_time=b,text=p) for a,b,p in rows])


class AlignmentGapTests(unittest.TestCase):
    def test_430ms_gap_preserves_tail_and_timeline(self):
        p,d,s,e,info=alignment_to_phones(tier([(0,.31,''),(.31,1.71,'a'),
                                              (2.14,2.85,'b'),(2.85,3.5,'')]),22050,256)
        self.assertEqual(p,['a','sp','b'])
        self.assertEqual(sum(d),219)
        self.assertAlmostEqual(info['preserved_gap_seconds'],.43)
        self.assertEqual((s,e),(.31,2.85))

    def test_explicit_blank_and_hidden_gap_agree(self):
        a=alignment_to_phones(tier([(.31,1.71,'a'),(2.14,2.85,'b')]),22050,256)
        b=alignment_to_phones(tier([(.31,1.71,'a'),(1.71,2.14,''),(2.14,2.85,'b')]),22050,256)
        self.assertEqual(a[:4],b[:4])

    def test_overlap_is_rejected(self):
        with self.assertRaises(ValueError):
            alignment_to_phones(tier([(0,1,'a'),(.9,2,'b')]),22050,256)

    def test_no_speech_does_not_fabricate_targets(self):
        self.assertEqual(alignment_to_phones(tier([(0,1,'')]),22050,256)[:4],([],[],0,0))


if __name__=='__main__':unittest.main()
