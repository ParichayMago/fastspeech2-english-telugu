#!/usr/bin/env python3
"""Inspect assignment Parquet audio without resampling or modifying source data.

Raw data and row manifests stay in the ignored data directory. Reports contain
aggregates only. File access uses normal Hugging Face authentication; this script
does not request access, accept conditions, or save credentials.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import datetime
import hashlib
import io
import json
import math
import os
from pathlib import Path
import unicodedata

os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')
os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '60')

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import HfApi, hf_hub_download, hf_hub_url, get_hf_file_metadata
from huggingface_hub.utils import HfHubHTTPError

DATASETS = {
    'telugu': ('SPRINGLab/IndicTTS_Telugu', 'train', 'audio'),
    'english': ('ai4bharat/Svarah', 'test', 'audio_filepath'),
}


def json_write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=DATASETS, required=True)
    parser.add_argument('--cache-dir', type=Path, default=Path('data/dataset-audit'))
    parser.add_argument('--output-dir', type=Path, default=Path('src/artifacts/DatasetAudit'))
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--max-shards', type=int)
    parser.add_argument('--revision', help='Pinned dataset commit to audit/download')
    args = parser.parse_args()
    repo, split, audio_column = DATASETS[args.dataset]
    # A public repository needs no credential. Gated English uses SDK auth only.
    token = False if args.dataset == 'telugu' else None
    api = HfApi(token=token)
    info = api.dataset_info(repo, revision=args.revision, files_metadata=True)
    card = info.card_data.to_dict() if info.card_data else {}
    files = [f for f in info.siblings if f.rfilename.endswith('.parquet')]
    files.sort(key=lambda f: f.rfilename)
    selected = files[:args.max_shards] if args.max_shards else files
    report = {
        'dataset': repo, 'revision': info.sha, 'split': split,
        'verified_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'gated': info.gated, 'card_dataset_info': card.get('dataset_info'),
        'download_bytes_all_shards': sum(f.size or 0 for f in files),
        'parquet_shards': len(files), 'selected_shards': len(selected),
        'audio_checked': False, 'access': 'unknown',
        'scope': 'Metadata only until download and decoding are explicitly enabled.',
        'audit_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    output = args.output_dir / (args.dataset + '-audit.json')
    try:
        get_hf_file_metadata(hf_hub_url(repo, files[0].rfilename,
            repo_type='dataset', revision=info.sha), token=token)
        report['access'] = 'verified_file_access'
    except HfHubHTTPError as e:
        report.update(access='blocked', access_http_status=e.response.status_code,
                      access_error=type(e).__name__)
        json_write(output, report)
        print(json.dumps({'dataset': repo, 'access': report['access'],
                          'http_status': report['access_http_status']}), flush=True)
        return
    if not args.download:
        json_write(output, report)
        print(json.dumps({'dataset': repo, 'access': report['access']}), flush=True)
        return

    args.cache_dir.mkdir(parents=True, exist_ok=True)

    def download(f):
        print('Downloading ' + f.rfilename, flush=True)
        p = Path(hf_hub_download(repo, f.rfilename, repo_type='dataset',
            revision=info.sha, token=token, cache_dir=str(args.cache_dir / 'hf-cache')))
        assert p.stat().st_size == f.size, 'Downloaded file size mismatch'
        if f.lfs:
            with p.open('rb') as handle:
                digest = hashlib.file_digest(handle, 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else None
                if digest is None:
                    h = hashlib.sha256()
                    for block in iter(lambda: handle.read(8 * 1024**2), b''):
                        h.update(block)
                    digest = h.hexdigest()
            assert digest == f.lfs['sha256'], 'Downloaded file SHA-256 mismatch'
        print('Verified ' + f.rfilename, flush=True)
        return f, p

    manifest = args.cache_dir / (args.dataset + '-manifest.jsonl')
    counters = Counter({key: 0 for key in ['rows', 'decoded_rows', 'candidate_rows',
        'empty_transcripts', 'audio_decode_errors', 'duplicate_audio_rows',
        'duplicate_audio_conflicting_text_rows', 'near_silent_rows',
        'rows_with_full_scale_samples', 'severe_clipping_rows']})
    rates, channels, encodings = Counter(), Counter(), Counter()
    groups = defaultdict(lambda: {'rows': 0, 'seconds': 0.0, 'candidate_seconds': 0.0,
        'candidate_rows': 0, 'sample_rates': Counter(), 'audio_path_examples': []})
    durations, rms_values, peak_values, text_lengths = [], [], [], []
    text_seen, audio_seen, candidate_seconds = Counter(), {}, 0.0
    audio_paths, column_names, file_reports = [], None, []
    with ThreadPoolExecutor(max_workers=2) as pool, manifest.open('w') as out:
        for source, path in pool.map(download, selected):
            parquet = pq.ParquetFile(path)
            column_names = parquet.schema_arrow.names
            file_reports.append({'file': source.rfilename, 'rows': parquet.metadata.num_rows,
                                 'bytes': source.size, 'sha256_verified': bool(source.lfs)})
            file_row = 0
            for batch in parquet.iter_batches(batch_size=16):
                for row in batch.to_pylist():
                    counters['rows'] += 1
                    text = unicodedata.normalize('NFC', row.get('text') or '').strip()
                    text_hash = hashlib.sha256(text.encode()).hexdigest()
                    text_seen[text_hash] += 1
                    text_lengths.append(len(text))
                    group = str(row.get('gender', 'unknown'))
                    if args.dataset == 'telugu':
                        group = {'0': 'female', '1': 'male'}.get(group, group)
                    speaker = row.get('speaker_id', row.get('speaker'))
                    audio = row.get(audio_column) or {}
                    raw = audio.get('bytes')
                    audio_path = audio.get('path')
                    if len(audio_paths) < 8 and audio_path:
                        audio_paths.append(audio_path)
                    record = {'shard': source.rfilename, 'row_in_shard': file_row,
                              'text': text, 'text_sha256': text_hash, 'gender': group,
                              'speaker_id': speaker, 'audio_path': audio_path}
                    file_row += 1
                    flags = []
                    if not text:
                        flags.append('empty_transcript')
                    wanted_script = any('\u0c00' <= c <= '\u0c7f' for c in text) if args.dataset == 'telugu' else any('a' <= c.lower() <= 'z' for c in text)
                    if not wanted_script:
                        flags.append('no_expected_language_letters')
                    counters['empty_transcripts'] += not bool(text)
                    counters['no_expected_language_letters'] += not wanted_script
                    if not raw:
                        flags.append('missing_embedded_audio')
                    else:
                        audio_hash = hashlib.sha256(raw).hexdigest()
                        record['audio_sha256'] = audio_hash
                        duplicate = audio_hash in audio_seen
                        if duplicate:
                            counters['duplicate_audio_rows'] += 1
                            counters['duplicate_audio_conflicting_text_rows'] += audio_seen[audio_hash] != text_hash
                        audio_seen.setdefault(audio_hash, text_hash)
                        try:
                            with sf.SoundFile(io.BytesIO(raw)) as sound:
                                sample_rate, frames, n_channels = sound.samplerate, sound.frames, sound.channels
                                encoding = sound.format + '/' + sound.subtype
                                pcm = sound.read(dtype='float32', always_2d=True)
                            seconds = frames / sample_rate
                            if pcm.shape != (frames, n_channels) or not math.isfinite(seconds) or seconds <= 0 or not np.isfinite(pcm).all():
                                raise ValueError('Invalid decoded waveform')
                            peak = float(np.abs(pcm).max())
                            rms = float(np.sqrt(np.mean(pcm.astype(np.float64)**2)))
                            clipped_fraction = float(np.mean(np.abs(pcm) >= .999))
                            record.update(duration_seconds=seconds, sample_rate=sample_rate,
                                          channels=n_channels, frames=frames, peak=peak, rms=rms,
                                          clipped_sample_fraction=clipped_fraction)
                            if 'duration' in row:
                                record['source_duration_seconds'] = row['duration']
                                counters['duration_metadata_mismatch_over_50ms'] += abs(seconds - row['duration']) > .05
                            rates[str(sample_rate)] += 1
                            channels[str(n_channels)] += 1
                            encodings[encoding] += 1
                            durations.append(seconds); rms_values.append(rms); peak_values.append(peak)
                            groups[group]['rows'] += 1
                            groups[group]['seconds'] += seconds
                            groups[group]['sample_rates'][str(sample_rate)] += 1
                            if audio_path and len(groups[group]['audio_path_examples']) < 3:
                                groups[group]['audio_path_examples'].append(audio_path)
                            counters['decoded_rows'] += 1
                            if rms <= 1e-5:
                                flags.append('near_silent')
                                counters['near_silent_rows'] += 1
                            if clipped_fraction > .01:
                                flags.append('over_one_percent_clipped_samples')
                                counters['severe_clipping_rows'] += 1
                            if n_channels != 1:
                                flags.append('non_mono')
                            if not 1 <= seconds <= 20:
                                flags.append('outside_1_to_20_seconds')
                            if duplicate:
                                flags.append('duplicate_audio')
                            if peak >= .999:
                                counters['rows_with_full_scale_samples'] += 1
                            if not flags:
                                candidate_seconds += seconds
                                groups[group]['candidate_seconds'] += seconds
                                groups[group]['candidate_rows'] += 1
                                counters['candidate_rows'] += 1
                        except (RuntimeError, ValueError, ZeroDivisionError) as e:
                            flags.append('audio_decode_error')
                            counters['audio_decode_errors'] += 1
                            record['decode_error'] = type(e).__name__
                    record['flags'] = flags
                    for flag in flags:
                        counters['flag_' + flag] += 1
                    out.write(json.dumps(record, ensure_ascii=False) + '\n')
            print(json.dumps({'scanned_shard': source.rfilename, 'rows_so_far': counters['rows'],
                'decoded_hours_so_far': round(sum(durations)/3600, 4),
                'candidate_hours_so_far': round(candidate_seconds/3600, 4)}), flush=True)

    complete = len(selected) == len(files)
    expected = sum(s['num_examples'] for s in card.get('dataset_info', {}).get('splits', []) if s['name'] == split)
    if complete and expected:
        assert counters['rows'] == expected, 'Source row count mismatch'
    report.update(audio_checked=True, complete=complete,
        completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        scope='All selected embedded waveforms decoded fully with libsndfile; no listening, ASR, or forced-alignment validation.',
        columns=column_names, counters=dict(counters), sample_rates=dict(rates),
        channel_counts=dict(channels), audio_encodings=dict(encodings),
        decoded_audio_hours=sum(durations)/3600, candidate_audio_hours=candidate_seconds/3600,
        candidate_filter='Nonempty expected-script transcript; fully decoded finite mono audio; RMS > 1e-5; at most 1% samples at absolute amplitude >= .999; 1 <= duration <= 20 seconds; first exact audio hash only. Provisional, not final alignment/TTS quality.',
        groups={k:{**v, 'hours':v['seconds']/3600, 'candidate_hours':v['candidate_seconds']/3600} for k,v in groups.items()},
        duplicate_transcript_rows=sum(max(0, x-1) for x in text_seen.values()),
        audio_path_examples=audio_paths, files=file_reports,
        row_manifest=str(manifest),
        duration_seconds_quantiles={str(q):float(np.quantile(durations,q)) for q in [0,.01,.25,.5,.75,.95,.99,1]} if durations else {},
        transcript_length_quantiles={str(q):float(np.quantile(text_lengths,q)) for q in [0,.5,.95,1]} if text_lengths else {},
        speaker_identity_note='Gender labels alone do not prove unique speaker identity. Inspect source paths/card before mapping gender to speaker IDs.',
        feasible_5h_train_1h_validation_by_candidate_duration=candidate_seconds >= 6*3600,
    )
    json_write(output, report)
    print(json.dumps({'report':str(output), 'rows':counters['rows'],
        'decoded_audio_hours':report['decoded_audio_hours'],
        'candidate_audio_hours':report['candidate_audio_hours'],
        'groups': report['groups']}), flush=True)


if __name__ == '__main__':
    main()
