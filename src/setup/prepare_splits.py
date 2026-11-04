#!/usr/bin/env python3
"""Create deterministic, duration-targeted splits from the verified audio audits.

Repeated normalized text, exact audio, and English recording chunks are atomic
groups. Speaker identity is preserved when supplied; missing IDs remain null.
This selects source rows and does not alter waveforms or claim alignments exist.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import random
from pathlib import Path
import unicodedata


def canonical_text(text):
    text = unicodedata.normalize('NFKC', text).casefold()
    return ' '.join(''.join(c if unicodedata.category(c)[0] in 'LMN' else ' '
                            for c in text).split())


def stable_id(row, language, revision):
    key = f"{language}@{revision}:{row['shard']}:{row['row_in_shard']}"
    return language[:2] + '_' + hashlib.sha256(key.encode()).hexdigest()[:20]


def leakage_keys(row, language):
    keys = [('text', canonical_text(row['text'])), ('audio', row['audio_sha256'])]
    if language == 'english':
        keys.append(('recording', Path(row['audio_path']).stem.rsplit('_chunk_', 1)[0]))
    return keys


def make_groups(rows, language):
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    seen = {}
    for i, row in enumerate(rows):
        for key in leakage_keys(row, language):
            if key in seen:
                a, b = find(i), find(seen[key])
                if a != b:
                    parent[b] = a
            else:
                seen[key] = i
    components = defaultdict(list)
    for i, row in enumerate(rows):
        components[find(i)].append(row)
    groups = []
    for members in components.values():
        members.sort(key=lambda row: row['sample_id'])
        identity = '|'.join(row['sample_id'] for row in members)
        groups.append({'id': hashlib.sha256(identity.encode()).hexdigest(),
                       'seconds': sum(row['duration_seconds'] for row in members),
                       'rows': members})
    return sorted(groups, key=lambda group: group['id'])


def choose_groups(groups, seconds, seed):
    ordered = list(groups)
    random.Random(seed).shuffle(ordered)
    chosen, remaining, total = [], [], 0.0
    for group in ordered:
        if total + group['seconds'] <= seconds:
            chosen.append(group)
            total += group['seconds']
        else:
            remaining.append(group)
    if remaining:
        closest = min(remaining, key=lambda g: abs(total + g['seconds'] - seconds))
        if abs(total + closest['seconds'] - seconds) < abs(total - seconds):
            chosen.append(closest)
            remaining = [g for g in remaining if g['id'] != closest['id']]
    return chosen, remaining


def flatten(groups):
    return sorted(({**row, 'leakage_group_id': group['id']}
                   for group in groups for row in group['rows']),
                  key=lambda row: row['sample_id'])


def verify_disjoint(train, validation, language):
    for key_type in ['sample', 'text', 'audio'] + (['recording'] if language == 'english' else []):
        def keys(rows):
            if key_type == 'sample':
                return {row['sample_id'] for row in rows}
            return {value for row in rows for kind, value in leakage_keys(row, language)
                    if kind == key_type}
        if keys(train) & keys(validation):
            raise ValueError('Train/validation overlap: ' + key_type)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--seed', type=int, default=20261102)
    parser.add_argument('--train-hours', type=float, default=5)
    parser.add_argument('--validation-hours', type=float, default=1)
    parser.add_argument('--tolerance-seconds', type=float, default=2)
    parser.add_argument('--telugu-gender', choices=['male', 'female', 'both'], default='male')
    parser.add_argument('--english-speakers', type=Path)
    parser.add_argument('--output', type=Path, default=Path('data/splits/v1'))
    args = parser.parse_args()
    root = args.root.resolve()
    output = root / args.output
    if not all(x > 0 for x in [args.train_hours, args.validation_hours, args.tolerance_seconds]):
        parser.error('Durations and tolerance must be positive')
    mapping = {}
    if args.english_speakers:
        mapping = json.loads((root / args.english_speakers).read_text())
    summary = {'seed': args.seed, 'train_target_hours': args.train_hours,
        'validation_target_hours': args.validation_hours,
        'maximum_duration_error_seconds': args.tolerance_seconds,
        'algorithm': 'Seeded random ordering of atomic connected components; greedily approach each target without exceeding it, then add the closest remaining component only if it reduces duration error. Validation selected first, training second; remaining candidates unused.',
        'grouping': 'NFKC/casefold/letter-mark-number transcript keys, exact audio SHA-256, and English filename recording roots. No group crosses splits.',
        'speaker_split_policy': 'Utterance/recording held out; shared verified speakers across train/validation are allowed for TTS. This is not a speaker-disjoint evaluation.',
        'telugu_gender_selection': args.telugu_gender,
        'alignment_status': 'not_yet_aligned', 'datasets': {}}
    for language in ['english', 'telugu']:
        audit_path = root / 'Results/DatasetAudit' / (language + '-audit.json')
        audit = json.loads(audit_path.read_text())
        if not (audit['complete'] and audit['audio_checked']):
            raise ValueError('Completed audio audit required: ' + language)
        source = root / audit['row_manifest']
        rows = []
        with source.open() as stream:
            for line in stream:
                row = json.loads(line)
                if row['flags'] or (language == 'telugu' and args.telugu_gender != 'both'
                                    and row['gender'] != args.telugu_gender):
                    continue
                row['sample_id'] = stable_id(row, language, audit['revision'])
                row['dataset'] = audit['dataset']
                row['dataset_revision'] = audit['revision']
                row['source_speaker_id'] = row.get('speaker_id')
                if language == 'english' and mapping:
                    speaker = mapping.get(row['audio_path'])
                    row['speaker_id'] = speaker['speaker_id'] if speaker else None
                    row['speaker_id_provenance'] = speaker['provenance'] if speaker else 'Not estimated: unused candidate'
                    row['speaker_id_estimated'] = bool(speaker and speaker.get('estimated'))
                    if speaker:
                        row['speaker_estimate_needs_review'] = speaker.get('needs_review', True)
                        row['speaker_estimate_cosine'] = speaker.get('cosine_to_assigned_training_centroid')
                        row['speaker_estimate_margin'] = speaker.get('centroid_margin')
                elif language == 'telugu':
                    row['speaker_id'] = 'te_' + row['gender']
                    row['speaker_id_provenance'] = 'Card-documented source speaker group, verified filename/gender consistency; no independent voice identity check.'
                else:
                    row['speaker_id'] = None
                    row['speaker_id_provenance'] = 'unresolved'
                rows.append(row)
        if not rows:
            raise ValueError('No candidate audio: ' + language)
        groups = make_groups(rows, language)
        validation, rest = choose_groups(groups, args.validation_hours*3600, args.seed)
        train, unused = choose_groups(rest, args.train_hours*3600, args.seed + 1)
        partitions = {'train': flatten(train), 'validation': flatten(validation), 'unused': flatten(unused)}
        if language == 'english' and mapping and any(row['speaker_id'] is None
                for split in ['train','validation'] for row in partitions[split]):
            raise ValueError('Selected English data is missing speaker mappings')
        verify_disjoint(partitions['train'], partitions['validation'], language)
        record = {'dataset': audit['dataset'], 'revision': audit['revision'],
                  'source_manifest_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                  'candidate_rows': len(rows), 'candidate_groups': len(groups),
                  'candidate_hours': sum(row['duration_seconds'] for row in rows)/3600,
                  'overlap_checks': 'passed', 'partitions': {}}
        if language == 'english' and mapping:
            record['speaker_label_kind'] = 'estimated_acoustic'
            record['speaker_mapping_file'] = str(args.english_speakers)
            record['speaker_mapping_sha256'] = hashlib.sha256((root / args.english_speakers).read_bytes()).hexdigest()
        for partition, members in partitions.items():
            seconds = sum(row['duration_seconds'] for row in members)
            if partition != 'unused':
                target = args.train_hours if partition == 'train' else args.validation_hours
                if abs(seconds-target*3600) > args.tolerance_seconds:
                    raise ValueError(f'{language}/{partition} cannot meet duration tolerance: {seconds}')
            path = output / language / (partition + '.jsonl')
            path.parent.mkdir(parents=True, exist_ok=True)
            content = ''.join(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n' for row in members)
            path.write_text(content)
            record['partitions'][partition] = {'rows': len(members), 'seconds': seconds,
                'hours': seconds/3600, 'manifest': str(path.relative_to(root)),
                'sha256': hashlib.sha256(content.encode()).hexdigest(),
                'gender_rows': dict(Counter(row['gender'] for row in members)),
                'speakers': len({row['speaker_id'] for row in members if row['speaker_id'] is not None}),
                'unresolved_speaker_rows': sum(row['speaker_id'] is None for row in members)}
            record['partitions'][partition]['estimated_speaker_rows'] = sum(row.get('speaker_id_estimated',False) for row in members)
        assert sum(len(x) for x in partitions.values()) == len(rows)
        summary['datasets'][language] = record
    summary['script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report = root / 'Results/DataSplits/summary.json'
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({language: {p: {'rows': x['rows'], 'hours': x['hours'],
        'unresolved_speaker_rows': x['unresolved_speaker_rows']}
        for p,x in d['partitions'].items() if p != 'unused'}
        for language,d in summary['datasets'].items()}, indent=2))


if __name__ == '__main__':
    main()
