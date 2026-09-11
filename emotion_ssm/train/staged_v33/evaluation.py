"""Identical query keys, globally reduced sums and frozen-head semantic controls."""
import math
import torch
import torch.distributed as dist
from torch.nn import functional as F
from emotion_ssm.train.dynamics_v3 import unit_state_readout
from emotion_ssm.train.staged_dynamics_support import memory_index
from .baselines import METHODS,predict_all
from .optimization import reduced
from .missing import digest


def ratio(a,b):return float(a/b) if float(b)>0 else None


def vector_summary(value):
    sse,count=value[...,0],value[...,1];present=count>0
    per_h=[]
    for h in range(count.shape[-1]):
        if present[:,h].any():per_h.append((sse[present[:,h],h]/count[present[:,h],h]).mean())
    return dict(mse=ratio(sse.sum(),count.sum()),macro_mse=float(torch.stack(per_h).mean()) if per_h else None,
        elements=int(count.sum()),by_domain=[ratio(s.sum(),c.sum()) for s,c in zip(sse,count)])


def semantic_summary(value,confusion):
    result={}
    for protocol,i in [('query_weighted',0),('endpoint_balanced',1)]:
        v=value[...,i,:].sum((0,1));cm=confusion[...,i,:,:].sum((0,1))
        true=cm.sum(-1);pred=cm.sum(-2);active=true>0
        recall=cm.diag()/true.clamp_min(1e-12)
        f1=2*cm.diag()/(true+pred).clamp_min(1e-12)
        result[protocol]=dict(ce=ratio(v[0],v[1]),accuracy=ratio(v[2],v[1]),emotion_count=float(v[1]),
            vad_mse=ratio(v[3],v[4]),vad_coordinate_count=float(v[4]),
            intensity_mse=ratio(v[5],v[6]),intensity_count=float(v[6]),
            macro_f1=float(f1[active].mean()) if active.any() else None,
            uar=float(recall[active].mean()) if active.any() else None,confusion=cm.tolist())
    return result


@torch.no_grad()
def evaluate(observer,core,bank,fitted,settings,device):
    rank=dist.get_rank() if dist.is_initialized() else 0;world=dist.get_world_size() if dist.is_initialized() else 1
    nd=int(bank['domain'].max())+1;nh=len(bank['horizons']);nm=len(METHODS);batch=settings['prediction_batch']
    sums=torch.zeros(nm,nd,nh,2,device=device,dtype=torch.float64)
    delta=torch.zeros(nd,nh,8,device=device,dtype=torch.float64)
    semantic=torch.zeros(nm,nd,nh,2,7,device=device,dtype=torch.float64)
    confusion=torch.zeros(nm,nd,nh,2,7,7,device=device,dtype=torch.float64)
    dialogue_ids=sorted(set(k[0] for k in bank['keys']));mapping={k:i for i,k in enumerate(dialogue_ids)}
    dialogue=torch.zeros(len(dialogue_ids),nm,nh,2,device=device,dtype=torch.float64)
    rows=list(range(rank,len(bank['keys']),world));partner=settings.get('enable_partner',True)
    for start in range(0,len(rows),batch):
        chosen=torch.tensor(rows[start:start+batch]);state=memory_index(bank['states'],chosen).to(device)
        truth=bank['targets'][chosen].to(device);valid=bank['valid'][chosen].to(device);domain=bank['domain'][chosen].to(device)
        identities=torch.tensor([mapping[bank['keys'][int(i)][0]] for i in chosen],device=device)
        outputs=core.forecast(state,bank['horizons'],enable_partner=partner)
        for h,seconds in enumerate(bank['horizons']):
            predictions=predict_all(core,state,bank['history'][chosen],seconds,fitted,outputs[h].z,partner)
            for m,name in enumerate(METHODS):
                error=(predictions[name].double()-truth[:,h].double()).square().sum(-1)*valid[:,h]
                counts=valid[:,h].sum(-1)*truth.shape[-1]
                sums[m,:,h,0].index_add_(0,domain,error.sum(-1));sums[m,:,h,1].index_add_(0,domain,counts.double())
                dialogue[:,m,h,0].index_add_(0,identities,error.sum(-1));dialogue[:,m,h,1].index_add_(0,identities,counts.double())
            inc=(outputs[h].z-state.z).double();true_inc=(truth[:,h]-state.z).double()
            norm=inc.norm(dim=-1);true_norm=true_inc.norm(dim=-1);dot=(inc*true_inc).sum(-1)
            defined=(norm>1e-8)&(true_norm>1e-8)&valid[:,h]
            center=fitted['mean'].to(state.z)-state.baseline
            center=center.double()/center.double().norm(dim=-1,keepdim=True).clamp_min(1e-12)
            residual=(outputs[h].z.double()-truth[:,h].double())
            values=torch.stack([inc.square().sum(-1),dot,(norm-true_norm).square(),
                torch.where(defined,1-dot/(norm*true_norm).clamp_min(1e-12),0.),defined.double(),
                (residual*center).sum(-1),(residual*center).sum(-1).square(),torch.ones_like(norm)],-1)
            delta[:,h].index_add_(0,domain,(values*valid[:,h,:,None]).sum(1))
    endpoints=bank['endpoints'][rank::world]
    for start in range(0,len(endpoints),batch):
        items=endpoints[start:start+batch];chosen=torch.tensor([r for r,q in items]);queries=[q for r,q in items]
        state=memory_index(bank['states'],chosen).to(device)
        seconds=torch.tensor([q['seconds'] for q in queries],device=device)
        predictions=predict_all(core,state,bank['history'][chosen],seconds,fitted,enable_partner=partner)
        role=torch.tensor([q['role'] for q in queries],device=device);arange=torch.arange(len(role),device=device)
        labels=[q['label'] for q in queries];emotion=torch.tensor([int(l.get('emotion',-1)) for l in labels],device=device)
        emotion_valid=(emotion>=0)&(emotion<7);ev=emotion.clamp(0,6)
        vad_mask=torch.tensor([l.get('vad_mask',[False]*3) for l in labels],device=device)
        vad=torch.tensor([l.get('vad',[0.]*3) for l in labels],device=device)
        im=torch.tensor([bool(l.get('intensity_mask',False)) for l in labels],device=device)
        intensity=torch.tensor([float(l.get('intensity',0.)) for l in labels],device=device)
        domain=bank['domain'][chosen].to(device);h=torch.tensor([bank['horizons'].index(q['nominal_horizon']) for q in queries],device=device)
        cell=domain*nh+h;weight=torch.tensor([q['endpoint_weight'] for q in queries],device=device,dtype=torch.float64)
        for m,name in enumerate(METHODS):
            decoded=observer.decode_affect(unit_state_readout(predictions[name][arange,role]))
            pred=decoded['emotion_logits'].argmax(-1)
            v=torch.stack([F.cross_entropy(decoded['emotion_logits'],ev,reduction='none')*emotion_valid,
                emotion_valid.double(),((pred==emotion)&emotion_valid).double(),
                torch.where(vad_mask,decoded['vad']-vad,0.).square().sum(-1),vad_mask.sum(-1).double(),
                torch.where(im,decoded['intensity'].reshape(-1)-intensity,0.).square(),im.double()],-1).double()
            for p,w in enumerate((torch.ones_like(weight),weight)):
                accumulated=torch.zeros(nd*nh,7,device=device,dtype=torch.float64)
                accumulated.index_add_(0,cell,v*w[:,None])
                semantic[m,:,:,p,:]+=accumulated.view(nd,nh,7)
                # A contiguous temporary avoids indexing into a non-contiguous reshaped view.
                cm=torch.zeros(nd*nh*49,device=device,dtype=torch.float64)
                cm.index_add_(0,cell*49+ev*7+pred,w*emotion_valid)
                confusion[m,:,:,p]+=cm.view(nd,nh,7,7)
    sums,delta,semantic,confusion,dialogue=[reduced(v).cpu() for v in (sums,delta,semantic,confusion,dialogue)]
    result=dict(methods={name:vector_summary(sums[m]) for m,name in enumerate(METHODS)},
        semantic={name:semantic_summary(semantic[m],confusion[m]) for m,name in enumerate(METHODS)},
        methods_order=list(METHODS),vector_sums=sums.tolist(),semantic_sums=semantic.tolist(),
        semantic_confusions=confusion.tolist(),delta_sums=delta.tolist(),
        delta_columns=['increment_l2','increment_dot_truth','magnitude_squared_error','direction_error_sum',
                       'direction_valid_count','center_residual_sum','center_residual_squared','valid_roles'],
        dialogues={name:dialogue[i].tolist() for i,name in enumerate(dialogue_ids)},
        independent_dialogues=len(dialogue_ids),producer=bank['producer'],protocol=bank['protocol'],
        query_hash=digest((bank['keys'],bank['valid'].tolist(),[(r,q['endpoint_key'],q['seconds']) for r,q in bank['endpoints']])),
        calibration_producer=fitted['producer'],calibration_keys_hash=fitted['calibration_keys_hash'],
        current=bank.get('current'),coverage=bank.get('coverage'),horizons=bank['horizons'])
    result['gain_over_hold']=1-result['methods']['learned']['macro_mse']/result['methods']['hold']['macro_mse']
    return result


def deployment_gate(metrics,initial,settings):
    tolerance=settings.get('acceptance_tolerance',.02);checks={}
    for protocol in ('fixed','live','update_gap','sensor_gap'):
        current=metrics[protocol];old=initial[protocol]
        checks[protocol+'_absolute']=current['methods']['learned']['macro_mse']<=old['methods']['learned']['macro_mse']*(1+tolerance)
        checks[protocol+'_beats_hold']=current['gain_over_hold']>0
        for domain in settings.get('deployment_domains',[2]):
            if domain>=len(current['methods']['learned']['by_domain']):continue
            a=current['methods']['learned']['by_domain'][domain];b=old['methods']['learned']['by_domain'][domain]
            if a is not None and b is not None:checks[f'{protocol}_target_domain{domain}']=a<=b*(1+tolerance)
        values=torch.tensor(current['vector_sums'])[0];previous=torch.tensor(old['vector_sums'])[0]
        cols=[h for h,s in enumerate(current['horizons']) if s>=settings.get('long_horizon_min',16)]
        if cols:
            checks[protocol+'_long_horizon']=values[:,cols,0].sum()/values[:,cols,1].sum().clamp_min(1)<=previous[:,cols,0].sum()/previous[:,cols,1].sum().clamp_min(1)*(1+tolerance)
        a=current['semantic']['learned']['endpoint_balanced'];b=old['semantic']['learned']['endpoint_balanced']
        for key in ('vad_mse','intensity_mse','ce'):
            if a[key] is not None and b[key] is not None:checks[f'{protocol}_{key}']=a[key]<=b[key]*(1+tolerance)
        for key in ('accuracy','macro_f1','uar'):
            if a[key] is not None and b[key] is not None:checks[f'{protocol}_{key}']=a[key]>=b[key]-settings.get('semantic_absolute_tolerance',.02)
    checks['fixed_improved']=metrics['fixed']['methods']['learned']['macro_mse']<=initial['fixed']['methods']['learned']['macro_mse']*(1-settings.get('minimum_fixed_improvement',.005))
    checks['coverage_locked']=metrics['correction_parameters_unchanged']
    checks={k:bool(v) for k,v in checks.items()}
    return dict(gate_passed=all(checks.values()),checks=checks)
