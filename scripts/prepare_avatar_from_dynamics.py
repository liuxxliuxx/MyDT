"""Explicit new-run initialization: existing Avatar backbone plus newly trained dynamics."""
import argparse
import copy
import json
from pathlib import Path
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def prepare(avatar_path,dynamics_path,output):
    from emotion_ssm.utils.checkpoint_v3 import read_checkpoint,build_avatar,save_checkpoint
    from emotion_ssm.models.visual_affect_teacher import observer_digest
    old=read_checkpoint(avatar_path);new=read_checkpoint(dynamics_path)
    if old['kind']!='streaming_avatar_v3' or not {'observer','state'}<=set(new['models']):
        raise ValueError('Provide a complete Avatar and a complete upstream dynamics checkpoint')
    shared=set(old['provenance'])&set(new['provenance'])
    if any(old['provenance'][key]!=new['provenance'][key] for key in shared):
        raise ValueError('Shared source/token provenance differs; rebuild features before adaptation')
    if old['provenance'] and new['provenance'] and not shared:
        raise ValueError('No verified common feature source; align provenance before replacement')
    a=old['construction'].get('adapter_source_binding');b=new['construction'].get('adapter_source_binding')
    if a!=b:raise ValueError('Adapter/extractor source binding changed; recalibrate the matching extractor first')
    config=copy.deepcopy(old['config']);construction=copy.deepcopy(old['construction'])
    construction.update(observer=new['construction']['observer'],state=new['construction']['state'])
    construction.pop('visual_teacher',None);construction.pop('condition_routing',None)
    config['observer']=copy.deepcopy(construction['observer']);config['state']=copy.deepcopy(construction['state'])
    config['generation'].update(variant='dyadic',condition_routing={'mode':'full_state'},train_observer=False,train_state=False)
    config['generation'].pop('visual_teacher',None);config['generation_losses']={'boundary_weight':0.}
    config['generation_selection']={'metric':'generation_total'};config['long_history']={'enabled':False}
    config['paths'].update(resume='',avatar_initialization='',dynamics_checkpoint=str(dynamics_path))
    model=build_avatar(config,construction=construction,initialize=False)
    weights=old['models']['system'];prefix='generator.baseline.'
    model.generator.baseline.load_state_dict({k[len(prefix):]:v for k,v in weights.items() if k.startswith(prefix)},strict=True)
    model.observer.load_state_dict(new['models']['observer'],strict=True)
    model.teacher.load_state_dict(new['models'].get('teacher',new['models']['observer']),strict=True)
    model.state_model.load_state_dict(new['models']['state'],strict=True)
    if model.features is not None:
        prefix='features.';model.features.load_state_dict({k[len(prefix):]:v for k,v in weights.items() if k.startswith(prefix)},strict=True)
    # State distribution changed. Start the new readout at identity in every control.
    torch.nn.init.zeros_(model.generator.film.affine.weight);torch.nn.init.zeros_(model.generator.film.affine.bias)
    config['experiment']=dict(protocol='explicit-upstream-replacement-new-run-v1',avatar_source=str(avatar_path),
        dynamics_source=str(dynamics_path),avatar_source_step=old['global_step'],dynamics_source_step=new['global_step'],
        observer_sha256=observer_digest(model.observer),state_sha256=observer_digest(model.state_model),
        backbone_sha256=observer_digest(model.generator.baseline),film_reset_to_identity=True,optimizer_restored=False)
    output=Path(output)
    if output.exists():raise ValueError('Choose a new initialization path; existing checkpoints are preserved')
    save_checkpoint(output,{'system':model},config,model.construction_info,'streaming_avatar_v3',step=0,
                    run_state={'initialization_only':True})
    output.with_suffix('.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    return config['experiment']


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--avatar',required=True)
    p.add_argument('--dynamics',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();print(json.dumps(prepare(a.avatar,a.dynamics,a.output)))


if __name__=='__main__':main()
