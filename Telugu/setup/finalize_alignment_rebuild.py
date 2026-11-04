#!/usr/bin/env python3
"""Verify preservation, fingerprint the rebuilt version, and select its configs."""
import hashlib
import json
from pathlib import Path
import shutil
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[2]
VERSION='gap-fixed-v2'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    import numpy as np
    import yaml
    out=ROOT/'English/artifacts/Preprocessing'/VERSION
    report={'feature_version':VERSION,'created_utc':datetime.now(timezone.utc).isoformat(),
            'execution_device':'local_cpu','cloud_training_started':False,
            'languages':{},'legacy_checkpoint_hashes':{}}
    # The original GPU manifest establishes that old feature bytes remain intact.
    old_manifest=json.loads((ROOT/'data/preprocessing/artifact-sha256.json').read_text())
    old_count=0
    for name,entry in old_manifest['files'].items():
        if name.startswith(('data/features/english/','data/features/telugu/')):
            assert sha(ROOT/name)==entry['sha256'], 'Legacy feature changed: '+name
            old_count+=1
    report['legacy_feature_files_hash_verified']=old_count
    supervisor=json.loads((ROOT/'English/artifacts/Training/supervisor.json').read_text())
    for lang,entry in supervisor['verified_local_checkpoints'].items():
        assert sha(ROOT/entry['path'])==entry['sha256'],'Legacy checkpoint changed'
        report['legacy_checkpoint_hashes'][lang]=entry['sha256']
    artifacts={}
    for lang,folder in [('english','English'),('telugu','Telugu')]:
        check=json.loads((out/(lang+'-feature-verification.json')).read_text())
        batch=json.loads((out/(lang+'-real-batch-check.json')).read_text())
        assert check['status']==batch['status']=='passed'
        assert batch['feature_version']==VERSION
        assert check.get('full_cropped_alignment_timeline_verified')
        assert check.get('all_internal_gaps_covered_by_silence_tokens')
        base=ROOT/'data/features'/VERSION/lang
        summary=json.loads((out/(lang+'-features.json')).read_text())
        changes={}
        for split,filename in [('train','train.txt'),('validation','val.txt')]:
            old={line.split('|')[0]:line.split('|') for line in
                 (ROOT/'data/features'/lang/filename).read_text().splitlines()}
            new={line.split('|')[0]:line.split('|') for line in (base/filename).read_text().splitlines()}
            assert old.keys()==new.keys(),'Accepted split membership changed'
            phone_changed=mel_extended=added_frames=0
            for sample,new_fields in new.items():
                speaker=new_fields[1]
                assert old[sample][1]==speaker
                a=np.load(ROOT/'data/features'/lang/'duration'/f'{speaker}-duration-{sample}.npy')
                b=np.load(base/'duration'/f'{speaker}-duration-{sample}.npy')
                phone_changed+=new_fields[2]!=old[sample][2]
                mel_extended+=int(b.sum())>int(a.sum())
                added_frames+=int(b.sum())-int(a.sum())
            changes[split]={'rows':len(new),'phone_sequences_changed':phone_changed,
                            'rows_with_longer_acoustic_targets':mel_extended,
                            'net_restored_frames':added_frames,
                            'net_restored_hours':added_frames*256/22050/3600,
                            'same_sample_and_speaker_membership':True}
        for p in sorted(base.rglob('*')):
            if p.is_file():
                assert p.resolve().is_relative_to(ROOT/'data')
                artifacts[str(p.relative_to(ROOT))]={'sha256':sha(p),'bytes':p.stat().st_size}
        # Save exact old configs before selecting the verified corrected features.
        archived=ROOT/'data/rebuild-config/legacy-before-gap-fixed-v2'/folder
        archived.mkdir(parents=True,exist_ok=True)
        for name in ['preprocess.yaml','model.yaml','train.yaml']:
            src=ROOT/'Telugu/config'/folder/name
            if not (archived/name).exists():shutil.copyfile(src,archived/name)
        production=ROOT/'Telugu/config'/folder/'preprocess.yaml'
        config=yaml.safe_load(production.read_text())
        config['path']['preprocessed_path']=str(base.relative_to(ROOT))
        config['feature_version']=VERSION
        config['checkpoint_compatibility']='Legacy checkpoints were trained with truncated targets; prepare separate fresh training.'
        production.write_text(yaml.safe_dump(config,sort_keys=False))
        artifacts[str(production.relative_to(ROOT))]={'sha256':sha(production),'bytes':production.stat().st_size}
        model_source=ROOT/'data/rebuild-config'/VERSION/folder/'model.yaml'
        model_target=ROOT/'Telugu/config'/folder/'model.yaml'
        model_config=yaml.safe_load(model_source.read_text())
        audit=[json.loads(line) for line in (base/'feature-audit.jsonl').read_text().splitlines()]
        max_frames=max(row['frames'] for row in audit if row['status']=='passed')
        assert model_config['max_seq_len']>=max_frames
        assert batch.get('longest_examples_checked') and batch.get('full_training_mel_targets_covered')
        shutil.copyfile(model_source,model_target)
        artifacts[str(model_target.relative_to(ROOT))]={'sha256':sha(model_target),'bytes':model_target.stat().st_size}
        report['languages'][lang]={'rebuild':summary,'comparison_with_legacy':changes,
                                  'feature_validation':'passed','real_batch_check':'longest_examples_passed_on_cpu',
                                  'max_mel_frames':max_frames,'model_frame_limit':model_config['max_seq_len']}
    for name in ['Telugu/preprocessor/alignment.py','Telugu/preprocessor/preprocessor.py','Telugu/audio/stft.py',
                 'Telugu/setup/extract_assignment_features.py','Telugu/setup/verify_preprocessed_features.py',
                 'Telugu/setup/finalize_alignment_rebuild.py','Telugu/setup/train_assignment.py']:
        p=ROOT/name;artifacts[name]={'sha256':sha(p),'bytes':p.stat().st_size}
    manifest=ROOT/'data/features'/VERSION/'artifact-sha256.json'
    manifest.write_text(json.dumps({'hash':'SHA-256','files':artifacts},indent=2,sort_keys=True)+'\n')
    # Independent disk reread verifies the complete version manifest.
    for name,entry in json.loads(manifest.read_text())['files'].items():
        assert sha(ROOT/name)==entry['sha256']
    report['new_artifacts']={'status':'passed','files':len(artifacts),
        'bytes':sum(entry['bytes'] for entry in artifacts.values()),
        'manifest_path':str(manifest.relative_to(ROOT)),'manifest_sha256':sha(manifest)}
    report['status']='rebuilt_verified_and_selected_in_production_configs'
    comparison=json.loads((ROOT/'Results/VocoderComparison'/VERSION/'comparison.json').read_text())
    for lang,d in comparison['languages'].items():
        assert d['saved_mel_frames']==d['fresh_full_mel_frames']
        assert d['saved_mel_missing_tail_seconds']==0
    report['vocoder_regression']={lang:{
        'saved_mel_frames':d['saved_mel_frames'], 'fresh_full_mel_frames':d['fresh_full_mel_frames'],
        'missing_tail_seconds':d['saved_mel_missing_tail_seconds'],
        'reconstructed_seconds':d['samples']['reconstructed']['duration_seconds']}
        for lang,d in comparison['languages'].items()}
    report['legacy_features_preserved']=True
    report['legacy_checkpoints_preserved']=True
    report['legacy_checkpoints_compatible_with_new_training']=False
    (out/'rebuild-summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='languages'},indent=2))


if __name__=='__main__':main()
