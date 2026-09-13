"""Cross old/new replay origins and propagators under one frozen semantic decoder."""
import argparse
import json
from pathlib import Path
import time
import torch
from scripts.compare_staged_v33_checkpoints import (build_models,require_coordinates,verify_bank,
    FrozenViews,horizon_bank,target_hash,atomic_json)
from emotion_ssm.train.staged_v34.data import build_bank
from emotion_ssm.train.staged_v34.evaluation import evaluate
from emotion_ssm.train.staged_v33.baselines import fit
from emotion_ssm.train.staged_dynamics_support import memory_index,weight_digest
from emotion_ssm.train.dynamics_v3 import unit_state_readout
from emotion_ssm.utils.checkpoint_v3 import read_checkpoint


@torch.no_grad()
def trajectories(observer,core,bank,mean,device,batch=128):
    """Same real 32-second endpoints at 0..32 seconds; intermediate labels are not invented."""
    items=[(r,q) for r,q in bank['endpoints'] if q['nominal_horizon']==32]
    result=[]
    for start in range(0,len(items),batch):
        queries=items[start:start+batch];rows=torch.tensor([r for r,q in queries])
        state=memory_index(bank['states'],rows).to(device)
        roles=torch.tensor([q['role'] for r,q in queries],device=device);idx=torch.arange(len(rows),device=device)
        for h in [0,1,2,4,8,16,32]:
            seconds=torch.tensor([q['seconds'] if h==32 else h for r,q in queries],device=device)
            future=state if h==0 else core._propagate(state,seconds)
            z=future.z[idx,roles];fast=future.fast[idx,roles];slow=future.slow[idx,roles]
            baseline=future.baseline[idx,roles];relation=future.relation[idx,roles]
            logits=observer.decode_affect(unit_state_readout(z))['emotion_logits']
            probability=logits.softmax(-1)
            norm=lambda x:x.norm(dim=-1)
            similarity=lambda x,y:torch.nn.functional.cosine_similarity(x,y,dim=-1)
            for i,(r,q) in enumerate(queries):
                label=int(q['label'].get('emotion',-1))
                result.append(dict(dialogue=bank['keys'][r][0],domain=int(bank['domain'][r]),role=q['role'],
                    origin=bank['keys'][r][1],endpoint=q['endpoint'],horizon=h,actual_seconds=float(seconds[i]),
                    endpoint_emotion=label,emotion=int(logits[i].argmax()),probability=probability[i].cpu().tolist(),
                    neutral_minus_endpoint_true_logit=float(logits[i,4]-logits[i,label]) if 0<=label<7 else None,
                    z_norm=float(norm(z)[i]),fast_norm=float(norm(fast)[i]),slow_norm=float(norm(slow)[i]),
                    relation_norm=float(norm(relation)[i]),fast_slow_cosine=float(similarity(fast,slow)[i]),
                    cancellation_ratio=float((norm(fast+slow)/(norm(fast)+norm(slow)).clamp_min(1e-12))[i]),
                    baseline_cosine=float(similarity(z,baseline)[i]),
                    train_mean_cosine=float(similarity(z,mean.to(z).expand_as(z))[i])))
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--reference',required=True);p.add_argument('--candidate',required=True)
    p.add_argument('--bank-root',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True);started=time.monotonic();device=torch.device('cuda:0')
    old=read_checkpoint(args.reference);new=read_checkpoint(args.candidate);require_coordinates(old,new)
    settings={**new['config']['staged'],'replay_dialogue_batch':8,'prediction_batch':128}
    frozen=torch.load(Path(args.bank_root)/'fixed_validation.pt',map_location='cpu',weights_only=False)
    train=torch.load(Path(args.bank_root)/'fixed_calibration.pt',map_location='cpu',weights_only=False)
    verify_bank(frozen,old,new['run_state']['cohort']['validation'],settings)
    old_o,old_c=build_models(old,device,'optimized');new_o,new_c=build_models(new,device,'optimized')
    views=FrozenViews(frozen,device,settings)
    replayed=build_bank(new_o,new_c,views,None,list(range(len(views.rows))),frozen['horizons'],settings['origin_stride'])
    if target_hash(frozen)!=target_hash(replayed):raise ValueError('Cross replay changed real target sets')
    fitted=fit(train,old_c,settings,device);summary={}
    for origin,bank in [('old',frozen),('new',replayed)]:
        short=horizon_bank(bank,32)
        for propagation,core in [('old',old_c),('new',new_c)]:
            name=f'{origin}_origin_{propagation}_flow'
            metrics=evaluate(old_o,core,short,fitted,settings,device)
            atomic_json(out/(name+'.json'),metrics)
            atomic_json(out/(name+'_trajectory.json'),trajectories(old_o,core,bank,fitted['mean'],device))
            summary[name]=dict(vector_mse=metrics['methods']['learned']['macro_mse'],
                semantic=metrics['semantic']['learned'],by_domain=metrics['semantic_cells']['learned'])
            print(json.dumps(dict(event='cross_evaluated',name=name,elapsed=time.monotonic()-started)),flush=True)
    atomic_json(out/'summary.json',dict(status='complete',results=summary,target_hash=target_hash(frozen),
        teacher_hash=weight_digest({'teacher':old['models']['teacher']}),
        reference_step=old['global_step'],candidate_step=new['global_step'],
        intermediate_protocol='0..16 seconds are state diagnostics, labels belong to the actual future endpoint',
        calibration='same old training origins and fitted baselines for all four cells'))


if __name__=='__main__':main()
