#!/usr/bin/env python3
"""Private, loopback-only audio review. Uses only Python's standard library."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import re
import secrets
import threading
from urllib.parse import parse_qs, urlparse
import wave

ROOT = Path(__file__).resolve().parent.parent
STATIC = Path(__file__).resolve().parent / 'static'
STATUSES = {'unreviewed', 'good', 'needs_work', 'reject'}
ISSUES = {'hiss', 'distortion', 'pronunciation', 'transcript', 'speaker', 'timing', 'quiet'}


def utc():
    return datetime.now(timezone.utc).isoformat()


def allowed_path(path, root=ROOT):
    path = path.resolve()
    if not any(path.is_relative_to((root / folder).resolve()) for folder in ('Results', 'data/splits/v1')):
        raise ValueError('Audio path is outside review directories')
    if not path.is_file() or path.suffix.lower() != '.wav':
        raise ValueError('Audio unavailable')
    return path


def build_catalog(root=ROOT):
    root=Path(root).resolve()
    items = {}
    def add(path, **metadata):
        try:
            path = allowed_path(path, root)
        except ValueError:
            return
        relative = str(path.relative_to(root))
        key = hashlib.sha256(relative.encode()).hexdigest()[:24]
        if key in items:
            return
        with wave.open(str(path), 'rb') as wav:
            seconds = wav.getnframes() / wav.getframerate()
        sha = metadata.pop('sha256', None) or hashlib.sha256(path.read_bytes()).hexdigest()
        items[key] = dict(id=key, path=relative, seconds=seconds, sha256=sha,
                          audio_url='/audio/' + key, **metadata)
    for lang in ('english', 'telugu'):
        for split in ('train', 'validation'):
            manifest = root / 'data/splits/v1' / lang / (split + '-prepared.jsonl')
            if not manifest.exists():
                continue
            for line in manifest.read_text().splitlines():
                row = json.loads(line)
                add(root / row['prepared_audio_path'], kind='dataset', language=lang,
                    split=split, speaker=row.get('speaker_id') or 'unknown',
                    estimated=bool(row.get('speaker_id_estimated')),
                    flagged=bool(row.get('speaker_estimate_needs_review') or row.get('flags')),
                    sample_id=row['sample_id'], title=row['sample_id'], text=row['text'],
                    version='Source recording', sha256=row.get('audio_sha256'),
                    flags=row.get('flags', []))
    # Read only explicit sample metadata; never expose supervisor/runtime files.
    for metadata_path in sorted((root / 'Results').rglob('preview.json')):
        info = json.loads(metadata_path.read_text())
        path = metadata_path.parent / 'preview.wav'
        lang = 'english' if info.get('speaker_label', '').startswith('en_') or '/english/' in str(path).lower() else 'telugu'
        corrected = '/gap-fixed-v2-' in str(path)
        add(path, kind='generated', language=lang, split='generated',
            speaker=info.get('speaker_label', ''), estimated=info.get('speaker_label_estimated', False),
            flagged=False, sample_id=path.parent.name, title=('Corrected' if corrected else 'Previous') + ' · step ' + str(info.get('checkpoint_step', '?')),
            text=info.get('text') or info.get('source_text') or info.get('raw_text',''),
            version='Corrected / untreated' if corrected else 'Previous run / untreated', flags=[])
    for language in ('English','Telugu'):
        directory=root/'Results'/language/'test_generated_audios'
        for metadata_path in sorted(directory.glob('*.json')):
            info=json.loads(metadata_path.read_text());path=metadata_path.with_suffix('.wav')
            if not path.is_file():continue
            sha=hashlib.sha256(path.read_bytes()).hexdigest()
            mismatch=bool(info.get('audio_sha256') and info['audio_sha256']!=sha)
            add(path,kind='final',language=language.lower(),split='final',
                speaker=info.get('speaker','unknown'),estimated=bool(info.get('speaker_labels_estimated')),
                flagged=mismatch,sample_id=metadata_path.stem,title=metadata_path.stem+' · final submission',
                text=info.get('source_text',''),version='Final / untreated',
                flags=['metadata_sha256_mismatch'] if mismatch else [],sha256=sha)
    comparison_path=root/'Results/Listening/english-parent-vs-learned-pauses/comparison.json'
    if comparison_path.exists():
        comparison=json.loads(comparison_path.read_text())
        for pair in comparison['pairs']:
            for variant in ('A','B'):
                clip=pair['clips'][variant];path=root/clip['audio_path']
                if not path.is_file():continue
                actual=hashlib.sha256(path.read_bytes()).hexdigest()
                if actual!=clip['audio_sha256']:
                    raise ValueError('Listening comparison audio changed; regenerate comparison: '+str(path))
                add(path,kind='comparison',language='english',split='comparison',speaker=comparison['speaker'],
                    estimated=True,flagged=False,sample_id=pair['id']+'-'+variant,
                    title=pair['id']+' · version '+variant,text=pair['text'],version='Untreated comparison',
                    pair_id=pair['id'],variant=variant,flags=[],sha256=actual)
    run = root / 'Results/Training/gap-fixed-v2-20261103'
    diagnosis_path = run / 'english_diagnostics/diagnosis.json'
    if diagnosis_path.exists():
        diagnosis = json.loads(diagnosis_path.read_text())
        for name, sample in diagnosis.get('samples', {}).items():
            add(Path(sample['path']), kind='generated', language='english', split='generated',
                speaker=diagnosis['speaker'], estimated=True, flagged=False, sample_id=name,
                title=name.replace('_', ' ').capitalize(), text=sample['text'],
                version='Diagnostic / ' + ('reference mel' if 'ground_truth' in name else 'untreated'), flags=[])
    for report_name in ('comparison.json', 'comparison_moderate.json'):
        report_path = run / 'english_hiss_comparison' / report_name
        if report_path.exists():
            report = json.loads(report_path.read_text())
            for name, sample in report['samples'].items():
                labels = {'original_universal': 'Original universal vocoder', 'gentle_denoise': 'Gentle denoising',
                          'moderate_denoise': 'Moderate denoising', 'alternate_vocoder': 'English-specific vocoder'}
                add(Path(sample['path']), kind='generated', language='english', split='generated',
                    speaker='en_est_0155', estimated=True, flagged=False, sample_id=name,
                    title=labels.get(name, name), text=sample['text'], version=labels.get(name, name), flags=[])
    return items


class ReviewStore:
    def __init__(self, path):
        self.path = path
        self.lock = threading.Lock()
        self.reviews = json.loads(path.read_text()).get('reviews', {}) if path.exists() else {}

    def snapshot(self):
        with self.lock:
            return json.loads(json.dumps(self.reviews))

    def save(self, item, data):
        status, notes, issues = data.get('status'), data.get('notes', ''), data.get('issues', [])
        if status not in STATUSES or not isinstance(notes, str) or len(notes) > 4000:
            raise ValueError('Invalid review')
        if not isinstance(issues, list) or any(issue not in ISSUES for issue in issues):
            raise ValueError('Invalid issue tags')
        entry = dict(status=status, notes=notes, issues=sorted(set(issues)), updated_utc=utc(),
                     audio_sha256=item['sha256'], path=item['path'], language=item['language'],
                     split=item['split'], speaker=item['speaker'], kind=item['kind'])
        return self.commit(item['id'],entry)

    def commit(self,key,entry):
        with self.lock:
            updated = {**self.reviews, key: entry}
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix('.tmp')
            temporary.write_text(json.dumps({'version': 1, 'reviews': updated}, ensure_ascii=False, indent=2) + '\n')
            temporary.replace(self.path)
            self.reviews = updated
        return entry


def comparison_pairs(catalog):
    groups={}
    for item in catalog.values():
        if item['kind']=='comparison':groups.setdefault(item['pair_id'],{})[item['variant']]=item
    pairs={}
    for key,clips in groups.items():
        if set(clips)!={'A','B'} or clips['A']['text']!=clips['B']['text']:
            raise ValueError('Listening comparison requires matched A and B audio: '+key)
        pairs[key]=dict(id=key,text=clips['A']['text'],clips=clips,
                        audio_sha256={v:i['sha256'] for v,i in clips.items()})
    return pairs


class PairReviewStore(ReviewStore):
    def save(self,pair,data):
        preferences=data.get('preferences',{})
        dimensions={'clarity','pacing','intonation','overall'}
        if not isinstance(preferences,dict) or set(preferences)!=dimensions or any(v not in {'A','B','same','neither','unsure'} for v in preferences.values()):
            raise ValueError('Choose a response for each comparison dimension')
        notes=data.get('notes',{});reviewer=data.get('reviewer','')
        if (not isinstance(notes,dict) or set(notes)!={'A','B','comparison'} or
                any(not isinstance(v,str) or len(v)>4000 for v in notes.values()) or
                any(not notes[v].strip() for v in ('A','B')) or
                not isinstance(reviewer,str) or len(reviewer)>100 or data.get('listened_both') is not True):
            raise ValueError('Listen to both versions and supply valid notes')
        return self.commit(pair['id'],dict(preferences=preferences,notes=notes,reviewer=reviewer,
            listened_both=True,audio_sha256=pair['audio_sha256'],updated_utc=utc()))


def range_bounds(header, size):
    match = re.fullmatch(r'bytes=(\d*)-(\d*)', header)
    if not match or not any(match.groups()):
        raise ValueError('Invalid byte range')
    first, last = match.groups()
    start = int(first) if first else max(0, size - int(last))
    end = min(int(last), size - 1) if first and last else size - 1
    if start >= size or end < start:
        raise ValueError('Unsatisfiable range')
    return start, end


def handler_for(catalog, store, token, root=ROOT, pair_store=None):
    pairs=comparison_pairs(catalog)
    pair_store=pair_store or PairReviewStore(store.path.with_name(store.path.stem+'-pairs.json'))
    def valid_pair_reviews():
        return {k:v for k,v in pair_store.snapshot().items()
                if k in pairs and v.get('audio_sha256')==pairs[k]['audio_sha256']}
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def valid_host(self):
            return self.headers.get('Host') in {f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}'}

        def send_bytes(self, body, content_type='application/json', status=200, headers=None):
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            for key, value in (headers or {}).items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(body)

        def send_json(self, value, status=200):
            self.send_bytes(json.dumps(value, ensure_ascii=False).encode(), status=status)

        def do_GET(self):
            if not self.valid_host():
                return self.send_json({'error': 'Local access only'}, 403)
            url = urlparse(self.path)
            if url.path == '/api/catalog':
                reviews = store.snapshot()
                # A changed generated file needs a fresh judgment, not an old rating.
                reviews = {key: value for key, value in reviews.items() if key in catalog and value.get('audio_sha256') == catalog[key]['sha256']}
                return self.send_json({'items': list(catalog.values()), 'reviews': reviews, 'token': token,
                                       'saved_to': str(store.path), 'catalog_utc': utc(),
                                       'comparison_pairs':list(pairs.values()),'pair_reviews':valid_pair_reviews(),
                                       'pair_saved_to':str(pair_store.path)})
            if url.path=='/api/comparison-export':
                rows=[dict(id=k,text=pairs[k]['text'],**v) for k,v in valid_pair_reviews().items()]
                return self.send_bytes(json.dumps(rows,ensure_ascii=False,indent=2).encode(),
                    headers={'Content-Disposition':'attachment; filename="listening-comparisons.json"'})
            if url.path == '/api/export':
                query = parse_qs(url.query)
                reviews = store.snapshot()
                rows = []
                for key, item in catalog.items():
                    review = reviews.get(key)
                    if not review or review.get('audio_sha256') != item['sha256']:
                        continue
                    if query.get('status', ['good'])[0] != 'all' and review['status'] != query.get('status', ['good'])[0]:
                        continue
                    if any(query.get(k, ['all'])[0] not in ('all', item[k]) for k in ('kind', 'language', 'split', 'speaker')):
                        continue
                    rows.append({**item, **review})
                if query.get('format', ['csv'])[0] == 'json':
                    return self.send_bytes(json.dumps(rows, ensure_ascii=False, indent=2).encode(), headers={'Content-Disposition': 'attachment; filename="audio-selections.json"'})
                buffer = io.StringIO(newline='')
                fields = ['id', 'kind', 'language', 'split', 'speaker', 'text', 'seconds', 'path', 'audio_sha256', 'status', 'issues', 'notes', 'updated_utc']
                writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction='ignore'); writer.writeheader()
                for row in rows:
                    row['issues'] = '; '.join(row['issues'])
                    # Spreadsheet consumers must treat review text as literal text.
                    row = {k: ('\'' + v if isinstance(v, str) and v.startswith(('=', '+', '-', '@')) else v) for k, v in row.items()}
                    writer.writerow(row)
                filename='audio-observations.csv' if query.get('status')==['all'] else 'good-audio.csv'
                return self.send_bytes(buffer.getvalue().encode('utf-8-sig'), 'text/csv; charset=utf-8', headers={'Content-Disposition': 'attachment; filename="'+filename+'"'})
            if url.path.startswith('/audio/'):
                item = catalog.get(url.path[len('/audio/'):])
                if not item:
                    return self.send_json({'error': 'Unknown audio'}, 404)
                size = 0
                try:
                    path = allowed_path(root / item['path'],root)
                    size = path.stat().st_size
                    partial = self.headers.get('Range')
                    start, end = range_bounds(partial, size) if partial else (0, size - 1)
                except ValueError:
                    return self.send_bytes(b'', status=416, headers={'Content-Range': f'bytes */{size}'})
                self.send_response(206 if partial else 200)
                self.send_header('Content-Type', 'audio/wav'); self.send_header('Accept-Ranges', 'bytes')
                self.send_header('Content-Length', str(end - start + 1)); self.send_header('X-Content-Type-Options', 'nosniff')
                if partial:
                    self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
                self.end_headers()
                try:
                    with path.open('rb') as f:
                        f.seek(start); remaining = end - start + 1
                        while remaining:
                            block = f.read(min(65536, remaining))
                            if not block:
                                break
                            self.wfile.write(block); remaining -= len(block)
                except (BrokenPipeError, ConnectionResetError):
                    pass
                return
            assets = {'/': ('index.html', 'text/html; charset=utf-8'), '/app.js': ('app.js', 'text/javascript; charset=utf-8'), '/style.css': ('style.css', 'text/css; charset=utf-8')}
            if url.path in assets:
                filename, content_type = assets[url.path]
                return self.send_bytes((STATIC / filename).read_bytes(), content_type)
            return self.send_json({'error': 'Not found'}, 404)

        def do_POST(self):
            if not self.valid_host() or self.headers.get('X-Review-Token') != token:
                return self.send_json({'error': 'Reload the local review page'}, 403)
            if self.path not in ('/api/review','/api/comparison-review'):
                return self.send_json({'error': 'Not found'}, 404)
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 20000:
                    raise ValueError('Invalid request size')
                data = json.loads(self.rfile.read(length))
                if not isinstance(data, dict):
                    raise ValueError('Expected a review object')
                if self.path=='/api/comparison-review':
                    pair=pairs.get(data.get('id'))
                    if not pair or data.get('audio_sha256')!=pair['audio_sha256']:
                        raise ValueError('Comparison audio changed; reload')
                    return self.send_json({'review':pair_store.save(pair,data)})
                item = catalog.get(data.get('id'))
                if not item or data.get('sha256') != item['sha256']:
                    raise ValueError('Audio changed; reload before reviewing')
                return self.send_json({'review': store.save(item, data)})
            except (ValueError, TypeError, KeyError):
                return self.send_json({'error': 'Could not save: invalid review data'}, 400)
            except OSError:
                return self.send_json({'error': 'Could not save to disk; your review has not been saved'}, 500)
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8766)
    parser.add_argument('--store',default='data/audio-review/ratings.json',help='Separate local listening session ratings')
    args = parser.parse_args()
    store_path=(ROOT/args.store).resolve()
    if not store_path.is_relative_to((ROOT/'data/audio-review').resolve()):
        parser.error('Reviews must stay under data/audio-review/')
    catalog = build_catalog()
    store = ReviewStore(store_path)
    server = ThreadingHTTPServer(('127.0.0.1', args.port), handler_for(catalog, store, secrets.token_urlsafe(32)))
    print(f'Audio review: http://127.0.0.1:{server.server_port} ({len(catalog):,} recordings)', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
