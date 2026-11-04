#!/usr/bin/env python3
"""Verify completed dataset manifests and summarize available speaker metadata."""
from collections import Counter, defaultdict
import datetime
import hashlib
import json
import math
from pathlib import Path
import re


def main():
    root = Path(__file__).resolve().parents[2]
    output = root / 'English/artifacts/DatasetAudit'
    script_hash = hashlib.sha256((root / 'English/setup/audit_datasets.py').read_bytes()).hexdigest()
    summary = {'verified_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
               'status': 'passed', 'datasets': {}}
    for language in ('english', 'telugu'):
        report = json.loads((output / (language + '-audit.json')).read_text())
        assert report['audio_checked'] and report['complete'], language
        assert report['audit_script_sha256'] == script_hash, 'Executed audit source changed'
        assert all(f['sha256_verified'] for f in report['files'])
        rows, candidate_rows, decoded_rows = 0, 0, 0
        total_seconds, candidate_seconds = 0.0, 0.0
        texts, recordings = Counter(), Counter()
        ids, candidate_ids = set(), set()
        filename_parts = [set(), set()]
        by_group = defaultdict(lambda: {'rows': 0, 'seconds': 0.0, 'candidate_rows': 0,
                                        'candidate_seconds': 0.0})
        unparsed_recordings = 0
        telugu_path_gender_mismatch = 0
        with (root / report['row_manifest']).open() as stream:
            for line in stream:
                row = json.loads(line)
                rows += 1
                seconds = row.get('duration_seconds', 0.0)
                total_seconds += seconds
                decoded_rows += 'duration_seconds' in row
                texts[row['text_sha256']] += 1
                group = by_group[row['gender']]
                group['rows'] += 1
                group['seconds'] += seconds
                if row.get('speaker_id') is not None:
                    ids.add(str(row['speaker_id']))
                if not row['flags']:
                    candidate_rows += 1
                    candidate_seconds += seconds
                    group['candidate_rows'] += 1
                    group['candidate_seconds'] += seconds
                    if row.get('speaker_id') is not None:
                        candidate_ids.add(str(row['speaker_id']))
                name = Path(row.get('audio_path') or '').name
                if language == 'english':
                    match = re.fullmatch(r'(.+)_chunk_\d+\.wav', name)
                    if match:
                        recordings[match[1]] += 1
                        for i, part in enumerate(match[1].split('_')[:2]):
                            filename_parts[i].add(part)
                    else:
                        unparsed_recordings += 1
                else:
                    expected_prefix = 'train_telugu' + row['gender'] + '_'
                    telugu_path_gender_mismatch += not name.startswith(expected_prefix)
        assert rows == report['counters']['rows'] == decoded_rows == report['counters']['decoded_rows']
        assert candidate_rows == report['counters']['candidate_rows']
        assert math.isclose(total_seconds/3600, report['decoded_audio_hours'], abs_tol=1e-9)
        assert math.isclose(candidate_seconds/3600, report['candidate_audio_hours'], abs_tol=1e-9)
        assert sum(max(0, x-1) for x in texts.values()) == report['duplicate_transcript_rows']
        for gender, group in by_group.items():
            recorded = report['groups'][gender]
            assert group['rows'] == recorded['rows']
            assert group['candidate_rows'] == recorded['candidate_rows']
            assert math.isclose(group['seconds'], recorded['seconds'], abs_tol=1e-6)
            assert math.isclose(group['candidate_seconds'], recorded['candidate_seconds'], abs_tol=1e-6)
        item = {
            'dataset': report['dataset'], 'revision': report['revision'], 'access': report['access'],
            'rows': rows, 'decoded_rows': decoded_rows, 'candidate_rows': candidate_rows,
            'decoded_hours': report['decoded_audio_hours'],
            'candidate_hours': report['candidate_audio_hours'], 'sample_rates': report['sample_rates'],
            'groups': report['groups'], 'counters': report['counters'],
            'duplicate_transcript_rows': report['duplicate_transcript_rows'],
            'explicit_speaker_ids_found': len(ids),
            'speaker_identity_verified': False,
            'manifest_consistency': 'passed', 'audit_source_hash': script_hash,
            'all_parquet_sha256_verified': True,
            'sufficient_candidate_hours_for_5h_plus_1h': candidate_seconds >= 6*3600,
            'not_validated': ['Transcript/audio semantic agreement', 'Listening quality',
                              'Forced alignment', 'Final training/validation split duration'],
        }
        if language == 'english':
            item.update(provider_reported_speakers=117, recording_groups=len(recordings),
                maximum_clips_per_recording_group=max(recordings.values()),
                unparsed_recording_filenames=unparsed_recordings,
                filename_component_unique_values=[len(x) for x in filename_parts],
                speaker_mapping_status='unresolved',
                speaker_note='No speaker_id column. Filename components have far more than 117 values and must not be used as speaker IDs without source verification. Do not use demographics as identity.',
                split_grouping_note='Keep matching normalized transcripts and chunks from the same original recording together when constructing splits. Speaker IDs need separate recovery.',
                normalization_note='Numeric-only transcripts fail the conservative letter filter; they may be recoverable after verified number normalization.')
        else:
            assert telugu_path_gender_mismatch == 0
            item.update(provider_reported_speakers=2,
                filename_gender_mismatches=telugu_path_gender_mismatch,
                speaker_mapping_status='two_card_documented_gender_groups_with_consistent_filenames',
                speaker_note='Gender and filenames support two provisional source IDs, consistent with the card. They do not independently verify voice identity.',
                split_grouping_note='Keep matching normalized transcripts together across splits.')
        summary['datasets'][language] = item
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    estimated_path = root / 'English/artifacts/Preprocessing/english-estimated-speakers.json'
    if estimated_path.exists():
        estimated = json.loads(estimated_path.read_text())
        summary['datasets']['english']['estimated_training_labels'] = {
            'available': True, 'estimated': True,
            'original_ids_recovered': False,
            'estimated_clusters': estimated['estimated_cluster_count'],
            'report': str(estimated_path.relative_to(root))}
        (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (output / 'verification.json').write_text(json.dumps({
        'status': summary['status'], 'verified_utc': summary['verified_utc'],
        'audit_script_sha256': script_hash,
        'datasets': {k: {'rows': v['rows'], 'candidate_rows': v['candidate_rows'],
                         'manifest_consistency': v['manifest_consistency'],
                         'all_parquet_sha256_verified': v['all_parquet_sha256_verified']}
                     for k, v in summary['datasets'].items()}
    }, indent=2) + '\n')
    print(json.dumps({'status': summary['status'], 'datasets': {
        k: {field: v[field] for field in ['rows', 'decoded_hours', 'candidate_hours',
                                         'speaker_mapping_status']}
        for k, v in summary['datasets'].items()}}, indent=2))


if __name__ == '__main__':
    main()
