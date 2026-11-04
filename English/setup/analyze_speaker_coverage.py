#!/usr/bin/env python3
"""Quantify estimated English label coverage from distributed source membership."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[2]


def summarize(rows,label):
    clips=Counter();seconds=Counter()
    for row in rows:
        clips[row['speaker_id']]+=1
        seconds[row['speaker_id']]+=row['duration_seconds']
    ranked=sorted(clips,key=lambda key:(-seconds[key],key))
    return dict(clips=len(rows),labels=len(clips),source_hours=sum(seconds.values())/3600,
        labels_with_one_or_two_clips=sum(v<=2 for v in clips.values()),
        labels_with_one_or_two_clips_percent=100*sum(v<=2 for v in clips.values())/len(clips),
        median_clips_per_label=statistics.median(clips.values()),
        median_source_seconds_per_label=statistics.median(seconds.values()),
        final_label=dict(id=label,clips=clips[label],source_seconds=seconds[label],
                        source_minutes=seconds[label]/60,source_duration_rank=ranked.index(label)+1),
        per_label={key:dict(clips=clips[key],source_seconds=seconds[key]) for key in sorted(clips)})


def main():
    membership=ROOT/'English/artifacts/DataSplits/selection-membership.jsonl'
    feature=ROOT/'English/artifacts/Preprocessing/gap-fixed-v2/rebuild-summary.json'
    estimate=ROOT/'English/artifacts/Preprocessing/english-estimated-speakers.json'
    recovery=ROOT/'English/artifacts/DataSplits/english-speaker-recovery.json'
    rows=[json.loads(s) for s in membership.read_text().splitlines()]
    rows=[r for r in rows if r['language']=='english' and r['partition']=='train']
    rebuild=json.loads(feature.read_text())['languages']['english']['rebuild']
    rejected={r['sample_id'] for r in rebuild['failures'] if r['split']=='train'}
    accepted=[r for r in rows if r['sample_id'] not in rejected]
    assert len(accepted)==rebuild['split_rows']['train']
    label='en_est_0155';clustering=json.loads(estimate.read_text());publisher=json.loads(recovery.read_text())
    result=dict(unit='Estimated conditioning label; not a verified person',
        coverage_measure='Selected source WAV seconds, including edges. Not cropped feature duration or non-silent speech.',
        selected_training_source=summarize(rows,label),accepted_training_source=summarize(accepted,label),
        publisher_reported_speakers=publisher['reported_speakers'],estimated_cluster_count=clustering['estimated_cluster_count'],
        clustering=dict(method=clustering['method'],distance_threshold=clustering['distance_threshold'],
            threshold_calibration=clustering['threshold_calibration'],validation_used_to_fit_clusters=False),
        final_label_selection='Retained from earlier preview/diagnostic workflow for continuity; not selected through a systematic held-out perceptual ranking of all labels.',
        interpretation='Sparse, uncertain labels are a plausible modeling risk. The cluster count mismatch warrants an audit; it does not prove fragmentation counts or establish a cause of English speech defects.',
        input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (membership,feature,estimate,recovery)})
    out=ROOT/'English/artifacts/Assignment/analysis/english-speaker-coverage.json'
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({key:{k:v for k,v in result[key].items() if k!='per_label'}
                      for key in ('selected_training_source','accepted_training_source')},indent=2))


if __name__=='__main__':main()
