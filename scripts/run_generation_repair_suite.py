"""Materialize fair new experiments, then optionally execute them on GPU 2+3."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from emotion_ssm.config_v3 import read_config,write_config
from emotion_ssm.train.generation_stability import STABILITY_PROTOCOL
from emotion_ssm.models.condition_router import MODES


def build_suite(base,initialization,output,*,steps=1000,seeds=(6666,6667,6668),
                boundary_weights=(0.,.1),routes=('film_off','full_state'),mean=None,teacher=None,
                semantic_heads=(),long_history=False,visual_losses=None,selection=None):
    output=Path(output);specs=[]
    for seed in seeds:
        for weight in boundary_weights:
            for route in routes:
                if route not in MODES:raise ValueError('Unknown condition route')
                cfg=copy.deepcopy(base);name=f'{route}_boundary{weight:g}_seed{seed}'
                cfg['paths'].update(avatar_initialization=str(initialization),resume='',output=str(output/name))
                cfg['generation'].update(variant='dyadic',train_observer=False,train_state=False,
                    history_seconds=3,chunk_frames=25,condition_routing={'mode':route})
                cfg['experiment']=dict(controlled_condition_initialization=True,reset_condition_projector=True,
                    protocol='generation-boundary-condition-controls-v1',common_initialization=str(initialization))
                cfg['train'].update(seed=int(seed),max_steps=int(steps),global_chunks_per_step=32,
                    validation_max_dialogues=0,device='cuda:0',tbptt_seconds=4)
                cfg['generation_losses']=dict(visual_losses or {},boundary_weight=float(weight))
                cfg['generation_selection']=copy.deepcopy(selection or {'metric':'generation_total'})
                cfg['generation_stability']={**cfg.get('generation_stability',{}),'protocol':STABILITY_PROTOCOL,
                    'conversations_per_rank':4,'blocks_per_conversation':4,'retain_initial_candidate':False,
                    'cosine_steps':int(steps),'warmup_steps':min(200,max(0,int(steps)//10))}
                cfg['generation_execution']={**cfg.get('generation_execution',{}),'prefetch_batches':1,'defer_metrics':True}
                cfg['long_history']=dict(enabled=long_history,age_edges=[0,8,16,32,64],natural_probability=.5,cached_dialogues=2)
                status='ready'
                if teacher:cfg['generation']['visual_teacher']=dict(enabled=True,checkpoint=str(teacher))
                elif any(float(v)>0 for k,v in (visual_losses or {}).items() if k.startswith('visual_') and k.endswith('_weight')):
                    status='awaiting_validated_visual_teacher'
                if route in ('mean','mean_memory'):
                    if mean is None:status='awaiting_train_condition_mean'
                    else:cfg['generation']['condition_routing'].update(copy.deepcopy(mean),mode=route)
                if route in ('actual_semantic','oracle_visual_pseudo'):
                    if not teacher or not semantic_heads:status='awaiting_validated_visual_teacher'
                    else:
                        cfg['generation']['visual_teacher']=dict(enabled=True,checkpoint=str(teacher))
                        cfg['generation']['condition_routing']['semantic_heads']=list(semantic_heads)
                if route=='oracle_visual_pseudo' and long_history:status='oracle_prefix_not_causal_use_separate_short_window_diagnostic'
                if not teacher:
                    cfg['generation'].pop('visual_teacher',None)
                path=output/(name+'.json');write_config(path,cfg)
                specs.append(dict(name=name,config=str(path),status=status,steps=steps,effective_new_blocks=32,
                    estimated_training_seconds=None,gpus=[2,3],state_producer='identical_dyadic',condition_route=route))
    manifest=dict(protocol='generation-boundary-condition-controls-v1',common_initialization=str(initialization),
        runs=specs,validation_only_selection=True,full_retraining_required_for_effect_claim=True)
    output.mkdir(parents=True,exist_ok=True)
    (output/'suite.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base-config',required=True)
    p.add_argument('--initialize',required=True);p.add_argument('--output',required=True)
    p.add_argument('--steps',type=int,default=1000);p.add_argument('--seeds',nargs='+',type=int,default=[6666,6667,6668])
    p.add_argument('--boundary-weights',nargs='+',type=float,default=[0.,.1]);p.add_argument('--routes',nargs='+',choices=MODES,default=['film_off','full_state'])
    p.add_argument('--mean-json');p.add_argument('--visual-teacher');p.add_argument('--semantic-heads',nargs='+',choices=['emotion','vad','intensity'],default=[])
    p.add_argument('--long-history',action='store_true');p.add_argument('--execute',action='store_true')
    p.add_argument('--visual-losses-json');p.add_argument('--selection-json')
    a=p.parse_args();mean=json.loads(Path(a.mean_json).read_text()) if a.mean_json else None
    manifest=build_suite(read_config(a.base_config),a.initialize,a.output,steps=a.steps,seeds=a.seeds,
        boundary_weights=a.boundary_weights,routes=a.routes,mean=mean,teacher=a.visual_teacher,
        semantic_heads=a.semantic_heads,long_history=a.long_history,
        visual_losses=json.loads(Path(a.visual_losses_json).read_text()) if a.visual_losses_json else None,
        selection=json.loads(Path(a.selection_json).read_text()) if a.selection_json else None)
    if a.execute:
        blocked=[r for r in manifest['runs'] if r['status']!='ready']
        if blocked:raise ValueError('Resolve required inputs before executing suite: '+json.dumps(blocked))
        environment=dict(os.environ,CUDA_VISIBLE_DEVICES='2,3')
        for run in manifest['runs']:
            command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',
                     '-m','emotion_ssm.train.generation_v3','--config',run['config']]
            print(json.dumps(dict(starting=run['name'],command=command)),flush=True)
            with (Path(a.output)/(run['name']+'.log')).open('w',encoding='utf-8') as log:
                subprocess.run(command,env=environment,stdout=log,stderr=subprocess.STDOUT,check=True)
    print(str(Path(a.output)/'suite.json'))


if __name__=='__main__':main()
