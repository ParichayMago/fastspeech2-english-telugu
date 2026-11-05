"""Cross-file regressions for the independent submission review."""
import hashlib
from html.parser import HTMLParser
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.parse import urlsplit,unquote
import train_assignment as runner

ROOT=Path(__file__).resolve().parents[2]
LANGUAGE=Path(__file__).resolve().parents[1].name

class ReviewRegressions(unittest.TestCase):
    def test_review_page_links_resolve_and_twenty_sidecar_links_use_current_locations(self):
        page=ROOT/'English/artifacts/Assignment/index.html'
        class Links(HTMLParser):
            def __init__(self):super().__init__();self.urls=[]
            def handle_starttag(self,tag,attrs):self.urls.extend(v for k,v in attrs if k in ('href','src'))
        parser=Links();parser.feed(page.read_text())
        sidecars=[]
        for href in parser.urls:
            parts=urlsplit(href)
            if parts.scheme or parts.netloc or not parts.path:continue
            target=(page.parent/unquote(parts.path)).resolve()
            self.assertTrue(target.is_relative_to(ROOT),href)
            self.assertTrue(target.is_file(),href)
            if target.suffix in ('.txt','.json'):
                self.assertEqual(target.parent,ROOT/target.relative_to(ROOT).parts[0]/'artifacts/test_generated_audios')
                sidecars.append(target)
        self.assertEqual(len(sidecars),20)
        self.assertEqual(len(set(sidecars)),20)

    def test_training_fingerprint_includes_language_dataset_and_changes_with_its_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);dataset=root/LANGUAGE/'dataset.py';dataset.parent.mkdir()
            dataset.write_text('version_one')
            with patch.object(runner,'ROOT',root):first=runner.source_fingerprints()
            key=LANGUAGE+'/dataset.py'
            self.assertEqual(first[key],hashlib.sha256(dataset.read_bytes()).hexdigest())
            dataset.write_text('version_two')
            with patch.object(runner,'ROOT',root):second=runner.source_fingerprints()
            self.assertNotEqual(first[key],second[key])
            self.assertNotIn('dataset.py',second)

if __name__=='__main__':unittest.main()
