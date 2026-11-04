#!/usr/bin/env python3
"""Export selected original WAV bytes and transcripts from cached Parquet shards."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq
import soundfile as sf


def checked_write(path, content):
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError('Existing output differs: ' + str(path))
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)


def main():
    root = Path(__file__).resolve().parents[2]
    report = json.loads((root / 'Results/DataSplits/summary.json').read_text())
    result = {'waveforms': 'Original bytes, no resampling or trimming', 'datasets': {}}
    for language, dataset in report['datasets'].items():
        selected = defaultdict(dict)
        partition_rows = {}
        for partition in ['train', 'validation']:
            source = root / dataset['partitions'][partition]['manifest']
            members = [json.loads(line) for line in source.read_text().splitlines()]
            partition_rows[partition] = members
            for row in members:
                if row['row_in_shard'] in selected[row['shard']]:
                    raise ValueError('Duplicate selected source row')
                selected[row['shard']][row['row_in_shard']] = (partition, row)
        cache = root / 'data/dataset-audit/hf-cache' / ('datasets--' + dataset['dataset'].replace('/', '--')) / 'snapshots' / dataset['revision']
        audio_column = 'audio_filepath' if language == 'english' else 'audio'
        found = set()
        exported = {'train': [], 'validation': []}
        for shard, requested in sorted(selected.items()):
            parquet = pq.ParquetFile(cache / shard)
            source_row = 0
            for batch in parquet.iter_batches(batch_size=16, columns=[audio_column]):
                for item in batch.to_pylist():
                    if source_row in requested:
                        partition, row = requested[source_row]
                        raw = item[audio_column]['bytes']
                        if hashlib.sha256(raw).hexdigest() != row['audio_sha256']:
                            raise ValueError('Selected audio does not match audited source')
                        directory = (root / dataset['partitions'][partition]['manifest']).parent
                        wav = directory / 'wav' / (row['sample_id'] + '.wav')
                        lab = directory / 'lab' / (row['sample_id'] + '.lab')
                        checked_write(wav, raw)
                        checked_write(lab, (row['text'] + '\n').encode())
                        info = sf.info(str(wav))
                        if info.frames != row['frames'] or info.samplerate != row['sample_rate']:
                            raise ValueError('Exported waveform metadata differs')
                        if hashlib.sha256(wav.read_bytes()).hexdigest() != row['audio_sha256']:
                            raise ValueError('Export verification failed')
                        exported[partition].append({**row,
                            'prepared_audio_path': str(wav.relative_to(root)),
                            'transcript_path': str(lab.relative_to(root))})
                        found.add(row['sample_id'])
                    source_row += 1
        expected = {row['sample_id'] for rows in partition_rows.values() for row in rows}
        if found != expected:
            raise ValueError('Some selected audio was not exported')
        result['datasets'][language] = {}
        for partition, rows in exported.items():
            rows.sort(key=lambda row: row['sample_id'])
            source = root / dataset['partitions'][partition]['manifest']
            output = source.with_name(partition + '-prepared.jsonl')
            output.write_text(''.join(json.dumps(row, sort_keys=True, ensure_ascii=False) + '\n' for row in rows))
            result['datasets'][language][partition] = {'rows': len(rows),
                'hours': sum(row['duration_seconds'] for row in rows)/3600,
                'all_audio_sha256_verified': True, 'all_waveform_headers_verified': True,
                'manifest': str(output.relative_to(root)),
                'manifest_sha256': hashlib.sha256(output.read_bytes()).hexdigest()}
        print(json.dumps({language: result['datasets'][language]}), flush=True)
    (root / 'Results/DataSplits/export.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
