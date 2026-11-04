"""Materialize pause-training YAMLs from measured features and base configs."""
import copy
import json
import math
from pathlib import Path
import yaml


def write_configs(root, destination=None):
    root=Path(root).resolve()
    features=root/'data/features/learned-pauses-v1/english'
    for name in ('stats.json','speakers.json','train.txt','val.txt','boundary-schema.json'):
        if not (features/name).is_file():raise ValueError('Missing pause feature resource: '+name)
    schema=json.loads((features/'boundary-schema.json').read_text())
    if schema['version']!='learned-pauses-v1' or not 0<schema['train_pause_prior']<1:
        raise ValueError('Invalid measured pause schema')
    if not all(math.isfinite(schema[k]) for k in
               ('train_pause_pos_weight','train_mean_positive_log_duration','train_max_pause_frames')):
        raise ValueError('Nonfinite pause statistics')
    if schema['train_pause_pos_weight']<=0 or schema['train_max_pause_frames']<1:
        raise ValueError('Invalid pause duration/occurrence statistics')
    base=root/'English/config/English'
    configs=[yaml.safe_load((base/name).read_text()) for name in ('preprocess.yaml','model.yaml','train.yaml')]
    pre,model,train=copy.deepcopy(configs)
    pre['path']['preprocessed_path']=str(features.relative_to(root))
    pre['feature_version']='learned-pauses-v1'
    pre['checkpoint_compatibility']='Explicit learned-pause transfer only; preserve original checkpoints'
    model['learned_pause']=schema
    destination=Path(destination) if destination else root/'data/rebuild-config/learned-pauses-v1/English'
    destination.mkdir(parents=True,exist_ok=True)
    for name,value in zip(('preprocess.yaml','model.yaml','train.yaml'),(pre,model,train)):
        path=destination/name
        if path.exists() and yaml.safe_load(path.read_text())!=value:
            raise ValueError('Existing config differs; refusing overwrite: '+str(path))
        path.write_text(yaml.safe_dump(value,sort_keys=False,allow_unicode=True))
        if yaml.safe_load(path.read_text())!=value:raise ValueError('Config round-trip failed')
    return destination
