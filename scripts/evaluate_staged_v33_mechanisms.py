"""Matched-current interventions; no future partner/affect observations are read."""
import argparse
import copy
import json
from pathlib import Path
import torch
from emotion_ssm.utils.checkpoint_v3 import read_checkpoint
from emotion_ssm.models.token_observer import TokenObserver
from emotion_ssm.models.state_core import UnifiedEmotionStateCore
from emotion_ssm.train.dynamics_v3 import DialogueCollection
from emotion_ssm.train.dynamics_staged_v3 import source_balanced_indices
from emotion_ssm.train.staged_dynamics_support import memory_index
from emotion_ssm.train.staged_v33.data import FeatureViews,build_bank


@torch.no_grad()
def evaluate(payload,output,execution='optimized'):
    config=payload['config'];settings=config['staged'];device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(config['train'].get('cpu_threads',4))
    observer=TokenObserver(payload['construction']['observer']).to(device).eval()
    teacher=copy.deepcopy(observer).requires_grad_(False)
    core=UnifiedEmotionStateCore.from_config(payload['construction']['state']).to(device)
    for name,module in [('observer',observer),('teacher',teacher),('state',core)]:module.load_state_dict(payload['models'][name])
    core.configure_execution(execution);core.requires_grad_(False)
    cache=FeatureViews(observer,teacher,config['paths']['feature_cache'],device,settings)
    collection=DialogueCollection(config['data']['token_roots'],'val')
    indices=source_balanced_indices(collection,settings['validation_dialogues_per_domain'],config['train']['seed']+17011)
    bank=build_bank(observer,core,cache,collection,indices,config['train']['forecast_seconds'],settings['origin_stride'],
        'clean',settings['enable_partner'],lambda a,b,n:print(json.dumps(dict(dialogues=a,total=b,origins=n)),flush=True))
    horizons=bank['horizons'];nd=int(bank['domain'].max())+1
    totals=torch.zeros(4,nd,len(horizons),2,device=device,dtype=torch.float64)
    methods=['correct_partner','mismatched_partner','partner_disabled','same_current_without_slow_or_relation']
    candidates={}
    for row,(identity,now) in enumerate(bank['keys']):
        domain=int(bank['domain'][row]);candidates.setdefault(domain,{}).setdefault(identity,[]).append(row)
    donors=[]
    for row,(identity,now) in enumerate(bank['keys']):
        domain=int(bank['domain'][row]);names=sorted(k for k in candidates[domain] if k!=identity)
        donor=candidates[domain][names[row%len(names)]] if names else []
        donors.append(donor[row%len(donor)] if donor else -1)
    eligible=[i for i,d in enumerate(donors) if d>=0]
    for start in range(0,len(eligible),settings['prediction_batch']):
        rows=torch.tensor(eligible[start:start+settings['prediction_batch']]);state=memory_index(bank['states'],rows).to(device)
        other=memory_index(bank['states'],torch.tensor([donors[int(r)] for r in rows])).to(device)
        domain=bank['domain'][rows].to(device)
        for role in (0,1):
            own=state if role==0 else state.role_swap();donor=other if role==0 else other.role_swap()
            wrong=own.clone();wrong.fast[:,1]=donor.fast[:,1];wrong.slow[:,1]=donor.slow[:,1]
            # Preserve own current z exactly; remove persistent decomposition and relationship.
            removed=own.clone();removed.fast+=removed.slow;removed.slow.zero_();removed.relation.zero_()
            predicted=[core.forecast(own,horizons,enable_partner=settings['enable_partner']),
                       core.forecast(wrong,horizons,enable_partner=settings['enable_partner']),
                       core.forecast(own,horizons,enable_partner=False),
                       core.forecast(removed,horizons,enable_partner=settings['enable_partner'])]
            for h in range(len(horizons)):
                target=bank['targets'][rows,h,role].to(device);valid=bank['valid'][rows,h,role].to(device)
                for m,values in enumerate(predicted):
                    error=(values[h].z[:,0].double()-target.double()).square().sum(-1)*valid
                    totals[m,:,h,0].index_add_(0,domain,error)
                    totals[m,:,h,1].index_add_(0,domain,valid.double()*target.shape[-1])
    rows=torch.linspace(0,len(bank['keys'])-1,min(8,len(bank['keys']))).long()
    state=memory_index(bank['states'],rows).to(device)
    long_times=[0.,1.,8.,32.,120.,600.,1800.];outputs=core.forecast(state,long_times[1:],enable_partner=settings['enable_partner'])
    long=[]
    for seconds,value in zip(long_times,[state,*outputs]):
        if not all(bool(torch.isfinite(t).all()) for t in (value.fast,value.slow,value.relation)):raise FloatingPointError('Long integration became non-finite')
        long.append(dict(seconds=seconds,z_norm=float(value.z.norm(dim=-1).mean()),fast_norm=float(value.fast.norm(dim=-1).mean()),
            slow_norm=float(value.slow.norm(dim=-1).mean()),relation_norm=float(value.relation.norm(dim=-1).mean()),
            cancellation_ratio=float(((value.fast.norm(dim=-1)+value.slow.norm(dim=-1))/(value.fast+value.slow).norm(dim=-1).clamp_min(1e-8)).mean())))
    result=dict(step=payload['global_step'],methods=methods,vector_sums=totals.cpu().tolist(),horizons=horizons,
        wrong_partner_protocol='same-domain different-dialogue partner state; own current and relation unchanged',
        history_protocol='same current z; slow merged into fast; relation zeroed',
        eligible_origins=len(eligible),origin_count=len(bank['keys']),coverage=len(eligible)/len(bank['keys']),
        long_integration=long,long_integration_has_gold_targets=False,
        interpretation='Sensitivity and predictive MSE tests; not evidence of learned comfort/causal social strategy')
    Path(output).write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    p.add_argument('--execution',default='optimized');a=p.parse_args();evaluate(read_checkpoint(a.checkpoint),a.output,a.execution)
