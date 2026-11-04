"""Check saved reviews, export boundaries, local access and seekable playback."""
import csv
import hashlib
import http.client
import io
import json
import shutil
from pathlib import Path
import tempfile
import threading
import unittest
import wave
from http.server import ThreadingHTTPServer

from server import ROOT, ReviewStore, PairReviewStore, allowed_path, build_catalog, handler_for, range_bounds, comparison_pairs


class ReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixtures=tempfile.TemporaryDirectory()
        cls.root=Path(cls.fixtures.name)
        directory=cls.root/'data/splits/v1/english';directory.mkdir(parents=True)
        wav=directory/'fixture.wav'
        with wave.open(str(wav),'wb') as stream:
            stream.setnchannels(1);stream.setsampwidth(2);stream.setframerate(22050);stream.writeframes(b'\0\0'*1000)
        row=dict(prepared_audio_path=str(wav.relative_to(cls.root)),sample_id='synthetic-english',
                 speaker_id='fixture',text='Synthetic HTTP fixture',flags=[],audio_sha256=hashlib.sha256(wav.read_bytes()).hexdigest())
        (directory/'train-prepared.jsonl').write_text(json.dumps(row)+'\n')
        generated=cls.root/'src/artifacts/Training/english/fixture';generated.mkdir(parents=True)
        (generated/'preview.wav').write_bytes(wav.read_bytes())
        (generated/'preview.json').write_text(json.dumps(dict(source_text='Source-text metadata fixture',speaker_label='en_fixture',checkpoint_step=1)))
        comparison=cls.root/'Results/Listening/english-parent-vs-learned-pauses';comparison.mkdir(parents=True)
        clips={}
        for variant in ('A','B'):
            path=comparison/(variant+'.wav');path.write_bytes(wav.read_bytes())
            clips[variant]=dict(audio_path=str(path.relative_to(cls.root)),audio_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        (comparison/'comparison.json').write_text(json.dumps(dict(speaker='en_fixture',
            pairs=[dict(id='E1',text='Synthetic comparison sentence.',clips=clips)],system_key={'E1':{'A':'parent','B':'learned'}})))
        cls.catalog = build_catalog(cls.root)
        cls.item = next(i for i in cls.catalog.values() if i['kind'] == 'dataset' and i['language'] == 'english' and i['split'] == 'train')

    @classmethod
    def tearDownClass(cls):cls.fixtures.cleanup()

    def test_catalog_accepts_source_text_metadata(self):
        item=next(i for i in self.catalog.values() if i['kind']=='generated')
        self.assertEqual(item['text'],'Source-text metadata fixture')

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / 'reviews.json'
        self.store = ReviewStore(self.path)
        self.pair_store=PairReviewStore(Path(self.temp.name)/'pair-reviews.json')
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), handler_for(self.catalog, self.store, 'test-token',self.root,self.pair_store))
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

    def pair_payload(self):
        pair=comparison_pairs(self.catalog)['E1']
        return dict(id='E1',audio_sha256=pair['audio_sha256'],listened_both=True,reviewer='Synthetic test only',
                    preferences=dict(clarity='A',pacing='same',intonation='unsure',overall='B'),
                    notes=dict(A='Synthetic A note',B='Synthetic B note',comparison='Synthetic comparison'))

    def test_final_only_snapshot_discovers_all_ten_and_uses_actual_audio_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);audio=(self.root/self.item['path']).read_bytes()
            for language,ids in [('English',['E1','E2a','E2b','E3','E4']),('Telugu',['T1','T2a','T2b','T3','T4'])]:
                folder=root/'Results'/language/'test_generated_audios';folder.mkdir(parents=True)
                for sid in ids:
                    (folder/(sid+'.wav')).write_bytes(audio)
                    (folder/(sid+'.json')).write_text(json.dumps(dict(source_text='Exact '+sid,speaker='fixture',audio_sha256='outdated')))
            catalog=build_catalog(root)
            self.assertEqual(len(catalog),10)
            self.assertEqual({i['language'] for i in catalog.values()},{'english','telugu'})
            for item in catalog.values():
                self.assertEqual(item['kind'],'final');self.assertEqual(item['text'],'Exact '+item['sample_id'])
                self.assertEqual(item['sha256'],hashlib.sha256(audio).hexdigest());self.assertTrue(item['flagged'])

    def test_comparison_review_persists_exports_and_does_not_expose_mapping(self):
        _,body=self.request('GET','/api/catalog');catalog=json.loads(body)
        self.assertEqual(len(catalog['comparison_pairs']),1)
        self.assertNotIn('system_key',body.decode());self.assertNotIn('"A": "parent"',body.decode())
        self.assertNotIn('checkpoint_sha256',body.decode())
        self.assertEqual(catalog['pair_reviews'],{})
        status,_=self.request('POST','/api/comparison-review',self.pair_payload(),{'X-Review-Token':'test-token'})
        self.assertEqual(status,200)
        self.assertEqual(PairReviewStore(self.pair_store.path).snapshot()['E1']['preferences']['overall'],'B')
        _,body=self.request('GET','/api/comparison-export');export=json.loads(body)
        self.assertEqual(len(export),1);self.assertEqual(export[0]['notes']['A'],'Synthetic A note')
        # Stale judgments must never appear as ratings for changed audio.
        entry=self.pair_store.snapshot()['E1'];entry['audio_sha256']={'A':'stale','B':'stale'}
        self.pair_store.commit('E1',entry)
        _,body=self.request('GET','/api/comparison-export');self.assertEqual(json.loads(body),[])

    def test_comparison_rejects_stale_incomplete_or_unlistened_judgments(self):
        payload=self.pair_payload()
        self.assertEqual(self.request('POST','/api/comparison-review',payload)[0],403)
        invalid=[dict(audio_sha256={'A':'old','B':'old'}),dict(listened_both=False),dict(preferences={}),dict(preferences=[]),
                 dict(notes=[]),dict(notes=dict(A='',B='A note',comparison=''))]
        for patch in invalid:
            self.assertEqual(self.request('POST','/api/comparison-review',{**payload,**patch},{'X-Review-Token':'test-token'})[0],400)
        self.assertEqual(self.pair_store.snapshot(),{})

    def test_comparison_rejects_missing_pair_and_changed_audio(self):
        catalog={k:v for k,v in self.catalog.items() if not(v.get('pair_id')=='E1' and v.get('variant')=='B')}
        with self.assertRaisesRegex(ValueError,'matched A and B'):comparison_pairs(catalog)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=self.root/'Results/Listening';shutil.copytree(source,root/'Results/Listening')
            path=root/'Results/Listening/english-parent-vs-learned-pauses/A.wav';path.write_bytes(path.read_bytes()+b'changed')
            with self.assertRaisesRegex(ValueError,'audio changed'):build_catalog(root)

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
        for target,expected in [('/','text/html'),('/app.js','text/javascript'),('/style.css','text/css'),
                                ('/audio/'+self.item['id'],'audio/wav')]:
            client=http.client.HTTPConnection('127.0.0.1',self.server.server_port)
            client.request('GET',target);response=client.getresponse()
            self.assertEqual(response.status,200)
            self.assertEqual(response.getheader('Content-Type').split(';')[0],expected)
            response.read();client.close()
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
