"""Check saved reviews, export boundaries, local access and seekable playback."""
import csv
import http.client
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer

from server import ROOT, ReviewStore, allowed_path, build_catalog, handler_for, range_bounds


class ReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = build_catalog()
        cls.item = next(i for i in cls.catalog.values() if i['kind'] == 'dataset' and i['language'] == 'english' and i['split'] == 'train')

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / 'reviews.json'
        self.store = ReviewStore(self.path)
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), handler_for({self.item['id']: self.item}, self.store, 'test-token'))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True); self.thread.start()

    def tearDown(self):
        self.server.shutdown(); self.server.server_close(); self.thread.join(); self.temp.cleanup()

    def request(self, method, path, data=None, headers=None):
        client = http.client.HTTPConnection('127.0.0.1', self.server.server_port)
        body = json.dumps(data) if data is not None else None
        client.request(method, path, body=body, headers=headers or {})
        response = client.getresponse(); content = response.read(); status = response.status
        client.close(); return status, content

    def payload(self):
        return dict(id=self.item['id'], sha256=self.item['sha256'], status='good', issues=['hiss'], notes='Keep after listening')

    def test_review_survives_reload_and_export_preserves_split(self):
        status, _ = self.request('POST', '/api/review', self.payload(), {'X-Review-Token': 'test-token'})
        self.assertEqual(status, 200)
        reloaded = ReviewStore(self.path).snapshot()[self.item['id']]
        self.assertEqual(reloaded['notes'], 'Keep after listening')
        self.assertEqual(reloaded['audio_sha256'], self.item['sha256'])
        status, body = self.request('GET', '/api/export?kind=dataset&language=english&split=train')
        rows = list(csv.DictReader(io.StringIO(body.decode('utf-8-sig'))))
        self.assertEqual(status, 200); self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['split'], 'train')
        _, body = self.request('GET', '/api/export?split=validation&format=json')
        self.assertEqual(json.loads(body), [])

    def test_invalid_writes_and_host_are_rejected(self):
        self.assertEqual(self.request('POST', '/api/review', self.payload())[0], 403)
        self.assertEqual(self.request('GET', '/api/catalog', headers={'Host': 'external.example'})[0], 403)
        self.assertEqual(self.request('POST', '/api/review', [], {'X-Review-Token': 'test-token'})[0], 400)
        payload = self.payload(); payload['sha256'] = 'stale'
        self.assertEqual(self.request('POST', '/api/review', payload, {'X-Review-Token': 'test-token'})[0], 400)
        self.assertFalse(self.path.exists())

    def test_audio_range_and_no_arbitrary_files(self):
        status, body = self.request('GET', '/audio/' + self.item['id'], headers={'Range': 'bytes=0-43'})
        self.assertEqual(status, 206); self.assertEqual(len(body), 44); self.assertEqual(body[:4], b'RIFF')
        self.assertEqual(self.request('GET', '/audio/' + self.item['id'], headers={'Range': 'bytes=999999999-'} )[0], 416)
        self.assertEqual(self.request('GET', '/.git/config')[0], 404)
        self.assertEqual(self.request('GET', '/audio/../../.git/config')[0], 404)
        with self.assertRaises(ValueError):
            allowed_path(ROOT / '.git/config')
        self.assertEqual(range_bounds('bytes=-10', 100), (90, 99))

    def test_failed_write_does_not_report_saved(self):
        blocked = Path(self.temp.name) / 'file'; blocked.write_text('not a folder')
        self.store.path = blocked / 'review.json'
        status, _ = self.request('POST', '/api/review', self.payload(), {'X-Review-Token': 'test-token'})
        self.assertEqual(status, 500); self.assertEqual(self.store.snapshot(), {})


if __name__ == '__main__':
    unittest.main()
