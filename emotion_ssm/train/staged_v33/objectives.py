"""Assigned-query losses: independent domain/horizon/task sums and global counts."""
import torch
from torch.nn import functional as F
from emotion_ssm.train.dynamics_v3 import unit_state_readout
from emotion_ssm.train.staged_dynamics_support import memory_index
from .data import replay_origins
from .optimization import global_mean


def supports(sampler,device):
    shape=(max(sampler.bank['domain'].tolist())+1,len(sampler.bank['horizons']))
    result={key:torch.zeros(shape,dtype=torch.bool,device=device) for key in ('vector','ce','vad','intensity')}
    for d,columns in sampler.vector.items():
        for h in columns:result['vector'][d,h]=True
    for d,h,task in sampler.gold:result[task][d,h]=True
    return result


def loss(core,observer,bank,vectors,gold,settings,support,online=False):
    device=next(core.parameters()).device;shape=support['vector'].shape
    rows=sorted(set([r for r,h in vectors]+[bank['endpoints'][i][0] for i,task in gold]))
    if not rows:raise ValueError('Each rank needs an origin; increase the global query budget')
    index=torch.tensor(rows);mapping={r:i for i,r in enumerate(rows)}
    partner=settings.get('enable_partner',True)
    origins=(replay_origins(observer,core,bank,rows,device,partner) if online else
             memory_index(bank['states'],index).to(device))
    # A connected zero supports a rank with no annotations without training unused heads.
    zero=origins.z.sum()*0
    if not zero.requires_grad:
        zero=next(p for p in core.parameters() if p.requires_grad).sum()*0
    sums={key:torch.zeros(shape,device=device)+zero for key in support}
    counts={key:torch.zeros(shape,device=device,dtype=torch.float64) for key in support}
    if vectors:
        vi=torch.tensor([mapping[r] for r,h in vectors],device=device)
        h=torch.tensor([h for r,h in vectors],device=device)
        query=torch.tensor([bank['horizons'][col] for _,col in vectors],device=device)
        predicted=core._propagate(memory_index(origins,vi),query,enable_partner=partner).z
        row=torch.tensor([r for r,h in vectors]);hc=h.cpu()
        target=bank['targets'][row,hc].to(device);valid=bank['valid'][row,hc].to(device)
        errors=((predicted-target.detach()).square().sum(-1)*valid).sum(-1)
        cell=bank['domain'][row].to(device)*shape[1]+h
        sums['vector']=sums['vector'].flatten().index_add(0,cell,errors).view(shape)
        counts['vector'].view(-1).index_add_(0,cell,valid.sum(-1).double())
    if gold:
        queries=[bank['endpoints'][i][1] for i,task in gold]
        row=torch.tensor([bank['endpoints'][i][0] for i,task in gold])
        chosen=torch.tensor([mapping[int(r)] for r in row],device=device)
        seconds=torch.tensor([q['seconds'] for q in queries],device=device)
        state=core._propagate(memory_index(origins,chosen),seconds,enable_partner=partner)
        role=torch.tensor([q['role'] for q in queries],device=device)
        decoded=observer.decode_affect(unit_state_readout(state.z[torch.arange(len(role),device=device),role]))
        labels=[q['label'] for q in queries]
        h=torch.tensor([bank['horizons'].index(q['nominal_horizon']) for q in queries],device=device)
        cell=bank['domain'][row].to(device)*shape[1]+h
        for task in ('ce','vad','intensity'):
            assigned=torch.tensor([t==task for _,t in gold],device=device)
            if task=='ce':
                truth=torch.tensor([min(6,max(0,int(l.get('emotion',-1)))) for l in labels],device=device)
                error=F.cross_entropy(decoded['emotion_logits'],truth,reduction='none')
                denominator=assigned.double()
            elif task=='vad':
                mask=torch.tensor([l.get('vad_mask',[False]*3) for l in labels],device=device)
                truth=torch.tensor([l.get('vad',[0.]*3) for l in labels],device=device)
                error=torch.where(mask,decoded['vad']-truth,0.).square().sum(-1)
                denominator=mask.sum(-1)*assigned.double()
            else:
                truth=torch.tensor([float(l.get('intensity',0.)) for l in labels],device=device)
                error=torch.where(assigned,decoded['intensity'].reshape(-1)-truth,0.).square();denominator=assigned.double()
            sums[task]=sums[task].flatten().index_add(0,cell,error*assigned).view(shape)
            counts[task].view(-1).index_add_(0,cell,denominator)
    terms={};stats={}
    for key in support:
        terms[key],stats[key]=global_mean(sums[key],counts[key],support[key],require_complete=True)
    # The anchor is a separate per-domain mean over sampled current valid teacher roles.
    anchor_sums=torch.zeros(shape[0],1,device=device)+zero
    anchor_counts=torch.zeros(shape[0],1,device=device,dtype=torch.float64)
    if online:
        # Preserve query multiplicity across shards; rank-local deduplication would
        # change the anchor objective compared with an unsharded optimizer step.
        anchor_rows=[r for r,h in vectors]+[bank['endpoints'][i][0] for i,task in gold]
        anchor_index=torch.tensor(anchor_rows)
        local_index=torch.tensor([mapping[r] for r in anchor_rows],device=device)
        target=bank['current_gold'][anchor_index].to(device);valid=bank['current_valid'][anchor_index].to(device)
        # Missing inputs cannot earn a current reconstruction objective.
        evidence=[]
        for r in anchor_rows:
            record,_,stop=bank['replay_rows'][r];enc=bank['encoded_rows'][record]
            from .missing import update_missing
            present=enc['observations']['fresh_observation'][stop-1].bool().any(-1).clone()
            for role in (0,1):
                if update_missing(enc['plan'],enc['times'][stop-1],role):present[role]=False
            evidence.append(present)
        valid=valid & torch.stack(evidence).to(device)
        error=((origins.z[local_index]-target.detach()).square().sum(-1)*valid).sum(-1)
        domain=bank['domain'][anchor_index].to(device)
        anchor_sums[:,0].index_add_(0,domain,error)
        anchor_counts[:,0].index_add_(0,domain,valid.sum(-1).double())
    terms['anchor'],stats['anchor']=global_mean(anchor_sums,anchor_counts)
    label_weight=0. if settings.get('objective_mode')=='vector_only' else settings['endpoint_label_weight']
    total=terms['vector']+label_weight*sum(settings.get('task_weights',{}).get(k,1.)*terms[k] for k in ('ce','vad','intensity'))
    total=total+settings['state_anchor_weight']*terms['anchor']
    stats['vector_mse']=stats['vector']['value']/origins.fast.shape[-1]
    stats['total']=float(total.detach());stats['label_weight']=label_weight
    return total,terms,stats
