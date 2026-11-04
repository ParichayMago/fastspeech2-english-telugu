#!/usr/bin/env python3
"""Estimate acoustic training labels, never original/verified Svarah identities.

Fit clusters on training recordings only. Validation is assigned to frozen
training centroids. Gender/demographic fields are not clustering inputs.
"""
import argparse
from collections import defaultdict, Counter
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import soundfile as sf

MODEL = 'speechbrain/spkrec-ecapa-voxceleb'
MODEL_REVISION = '0f99f2d0ebe89ac095bcc5903c4dd8f72b367286'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--batch-size', type=int, default=12)
    parser.add_argument('--distance-threshold', type=float, default=.30)
    args = parser.parse_args()
    root = args.root.resolve()
    out = root / 'data/speaker-estimates'
    out.mkdir(parents=True, exist_ok=True)
    records, manifest_hashes = {}, {}
    for split in ['train', 'validation']:
        p = root / 'data/splits/v1/english' / (split + '-prepared.jsonl')
        manifest_hashes[split] = hashlib.sha256(p.read_bytes()).hexdigest()
        for line in p.read_text().splitlines():
            row = json.loads(line)
            recording = Path(row['audio_path']).stem.rsplit('_chunk_', 1)[0]
            if recording in records and records[recording]['split'] != split:
                raise ValueError('Recording crosses training/validation')
            rec = records.setdefault(recording, {'split': split, 'rows': []})
            rec['rows'].append(row)
    keys = sorted(records)
    for rec in records.values():
        rec['rows'].sort(key=lambda row: row['audio_path'])
    npz = out / 'recording-embeddings-v2.npz'
    # Audio input identity stays stable when speaker-label metadata is applied.
    audio_inputs = [(key, records[key]['split'], [(r['sample_id'],r['audio_sha256'])
                     for r in records[key]['rows']]) for key in keys]
    fingerprint = hashlib.sha256(json.dumps(audio_inputs, sort_keys=True).encode()).hexdigest()
    if npz.exists():
        cached = np.load(npz, allow_pickle=False)
        if str(cached['input_fingerprint']) != fingerprint:
            raise ValueError('Cached embeddings use different input manifests')
        vectors, durations = cached['vectors'], cached['durations']
        assert list(cached['recording_keys']) == keys
        model_revision = str(cached['model_revision'])
        assert model_revision == MODEL_REVISION, 'Cached speaker encoder revision differs'
    else:
        import torch
        from huggingface_hub import HfApi
        from speechbrain.inference.classifiers import EncoderClassifier
        torch.set_num_threads(4)
        model_revision = HfApi(token=False).model_info(MODEL,revision=MODEL_REVISION).sha
        encoder = EncoderClassifier.from_hparams(source=MODEL, revision=model_revision,
            savedir=str(out / 'ecapa-model'), run_opts={'device': 'cuda'})
        vectors, durations = [], []
        started = time.monotonic()
        for first in range(0, len(keys), args.batch_size):
            waves = []
            for key in keys[first:first+args.batch_size]:
                chunks, frames = [], 0
                for row in records[key]['rows']:
                    wav, rate = sf.read(root / row['prepared_audio_path'], dtype='float32')
                    if rate != 16000 or wav.ndim != 1:
                        raise ValueError('ECAPA expects mono 16 kHz')
                    remaining = 12 * 16000 - frames
                    if remaining <= 0:
                        break
                    chunks.append(wav[:remaining]); frames += min(len(wav), remaining)
                wav = np.concatenate(chunks)
                if not len(wav):
                    raise ValueError('Empty speaker waveform')
                waves.append(wav)
                durations.append(len(wav)/16000)
            longest = max(map(len, waves))
            padded = np.zeros((len(waves), longest), dtype=np.float32)
            lengths = []
            for i,wav in enumerate(waves):
                padded[i,:len(wav)] = wav; lengths.append(len(wav)/longest)
            with torch.inference_mode():
                embedding = encoder.encode_batch(torch.from_numpy(padded),
                    torch.tensor(lengths, dtype=torch.float32)).squeeze(1).cpu().numpy()
            vectors.extend(embedding)
            if first % (args.batch_size * 20) == 0:
                print(json.dumps({'estimated_embedding_records': min(first+len(waves),len(keys)),
                    'total_records':len(keys), 'elapsed_seconds':round(time.monotonic()-started,1)}), flush=True)
        vectors = np.asarray(vectors, dtype=np.float32)
        durations = np.asarray(durations, dtype=np.float32)
        np.savez_compressed(npz, vectors=vectors, durations=durations,
            recording_keys=np.asarray(keys), input_fingerprint=np.asarray(fingerprint),
            model_revision=np.asarray(model_revision))
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    if not np.isfinite(vectors).all():
        raise ValueError('Nonfinite speaker embeddings')
    train = np.array([records[key]['split'] == 'train' for key in keys])
    fit_indices = np.flatnonzero(train & (durations >= 3))
    if len(fit_indices) < 2:
        raise ValueError('Not enough training speech to estimate speakers')
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score
    fitted = AgglomerativeClustering(n_clusters=None, distance_threshold=args.distance_threshold,
        metric='cosine', linkage='average').fit_predict(vectors[fit_indices])
    groups = defaultdict(list)
    for index, label in zip(fit_indices, fitted):
        groups[int(label)].append(index)
    # Stable IDs from group membership, not the estimator's arbitrary label order.
    groups = sorted(groups.values(), key=lambda members: min(keys[i] for i in members))
    centroids = np.stack([vectors[members].mean(axis=0) for members in groups])
    centroids /= np.maximum(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-12)
    similarity = vectors @ centroids.T
    assigned = np.argmax(similarity, axis=1)
    # Keep fitted training memberships; only short training/validation are assigned.
    for label,members in enumerate(groups):
        assigned[members] = label
    mapping, qa = {}, defaultdict(Counter)
    for index,key in enumerate(keys):
        label = int(assigned[index])
        cosine = float(similarity[index,label])
        competitors = np.delete(similarity[index], label)
        margin = cosine-float(competitors.max()) if len(competitors) else 1.0
        review = bool(cosine < .50 or margin < .05 or durations[index] < 3)
        split = records[key]['split']
        qa[split]['recordings'] += 1
        qa[split]['review_flagged_recordings'] += review
        for row in records[key]['rows']:
            mapping[row['audio_path']] = {
                'speaker_id': f'en_est_{label:04d}', 'estimated': True,
                'provenance': 'ESTIMATED acoustic cluster from ECAPA embeddings; not publisher IDs or verified identities. Clustering fit on training only; validation uses frozen centroids.',
                'recording_group': key, 'embedding_duration_seconds':float(durations[index]),
                'cosine_to_assigned_training_centroid':cosine,
                'centroid_margin':margin,'needs_review':bool(review)}
    (out / 'english-mapping.json').write_text(json.dumps(mapping,indent=2)+'\n')
    np.save(out / 'training-centroids.npy', centroids)
    report = {'status':'estimated_labels_generated', 'labels_are_estimated':True,
        'original_speaker_ids_recovered':False,'biometric_identity_verified':False,
        'model':MODEL,'model_revision':model_revision,'model_license':'Apache-2.0',
        'method':'Average-linkage agglomerative cosine clustering of training recording embeddings with >=3 seconds speech. Short training and validation recordings assigned to frozen training centroids.',
        'distance_threshold':args.distance_threshold,
        'threshold_calibration':'Heuristic threshold; no ground-truth identity labels available for calibration.',
        'recording_waveform':'Concatenate selected chunks from same filename recording root, cap at 12 seconds.',
        'estimated_cluster_count':len(groups),'fit_training_recordings':len(fit_indices),
        'fitted_training_only':True,'validation_used_to_fit_clusters':False,
        'demographics_used_to_cluster':False,'quality_flags':dict(qa),
        'mapped_selected_clips':len(mapping),'input_manifest_sha256':manifest_hashes,
        'training_cluster_recording_sizes':sorted([len(m) for m in groups]),
        'limitations':['Clusters can merge different speakers or fragment one speaker.',
            'Short clips and negative/small centroid margins need listening review.',
            'Do not report these labels as original dataset speaker identities or a verified count.']}
    dest=root/'English/artifacts/Preprocessing'; dest.mkdir(parents=True,exist_ok=True)
    (dest/'english-estimated-speakers.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__':
    main()
