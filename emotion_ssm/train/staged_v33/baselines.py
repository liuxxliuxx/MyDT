"""Train-cohort-only baselines; actual query seconds are explicit."""
import torch
from emotion_ssm.train.staged_dynamics_support import memory_index
from .missing import digest

METHODS=('learned','hold','decay','mean','shrink','continuous_shrink',
         'calibrated_hold','current_linear','history_linear','center_amplitude')


def domain_weights(domain,valid):
    weights=torch.zeros_like(valid,dtype=torch.float64)
    for d in domain.unique():
        mask=valid & (domain==d)[:,None]
        if mask.any():weights[mask]=1./mask.sum()
    return weights/weights.sum().clamp_min(1e-12)


def regression(x,y,w,ridge):
    x=torch.cat((x.double(),torch.ones(*x.shape[:-1],1,dtype=torch.float64)),dim=-1).reshape(-1,x.shape[-1]+1)
    y=y.double().reshape(-1,y.shape[-1]);w=w.reshape(-1)
    mask=w>0;x=x[mask];y=y[mask];w=w[mask]
    gram=x.T@(x*w[:,None]);rhs=x.T@(y*w[:,None])
    penalty=torch.eye(gram.shape[0],dtype=gram.dtype)*ridge;penalty[-1,-1]=1e-10
    return torch.linalg.solve(gram+penalty,rhs).float()


@torch.no_grad()
def learned_predictions(core,bank,device,batch,partner=True):
    result=[]
    for start in range(0,len(bank['keys']),batch):
        state=memory_index(bank['states'],torch.arange(start,min(start+batch,len(bank['keys'])))).to(device)
        result.append(torch.stack([s.z.cpu() for s in core.forecast(state,bank['horizons'],enable_partner=partner)],1))
    return torch.cat(result)


@torch.no_grad()
def fit(bank,core,settings,device):
    z=bank['states'].z.double();target=bank['targets'].double();domain=bank['domain']
    mean=bank['teacher_sum']/max(1,bank['teacher_count'])
    current=torch.cat([z,z.flip(1)],-1).float()
    predicted=learned_predictions(core,bank,device,settings['prediction_batch'],settings.get('enable_partner',True)).double()
    result=dict(mean=mean.float(),horizons=list(bank['horizons']),producer=bank['producer'],
        decay_rates=torch.stack([v.detach().cpu() for v in core.rates()[:2]]),
        calibration_keys_hash=digest(bank['keys']),
        fit_split='train',ridge=settings.get('baseline_ridge',.001),
        endpoint_interpolation='piecewise_linear_coefficients_zero_anchor_clamp_beyond_last',
        shrink=[],current_linear=[],history_linear=[],center_amplitude=[])
    weights=[];ridge=settings.get('baseline_ridge',.001)
    result['calibrated_hold']=regression(current,bank['current_gold'],domain_weights(domain,bank['current_valid']),ridge)
    for h in range(len(bank['horizons'])):
        w=domain_weights(domain,bank['valid'][:,h]);weights.append(w)
        x=z-mean;y=target[:,h]-mean
        a=((x*y*w[...,None]).sum()/(x.square()*w[...,None]).sum().clamp_min(1e-12)).clamp(0,1)
        result['shrink'].append(float(a))
        result['current_linear'].append(regression(current,target[:,h],w,ridge))
        result['history_linear'].append(regression(bank['history'],target[:,h],w,ridge))
        delta=predicted[:,h]-z;center=mean-bank['states'].baseline.double()
        x=torch.stack([delta,center],-1).reshape(-1,2)
        y=(target[:,h]-z).reshape(-1,1);ww=w[...,None].expand_as(delta).reshape(-1,1)
        gram=x.T@(x*ww);rhs=x.T@(y*ww)
        coefficient=torch.linalg.solve(gram+1e-8*torch.eye(2,dtype=torch.float64),rhs).reshape(2)
        result['center_amplitude'].append(coefficient.float())
    candidates=torch.cat([torch.zeros(1),torch.logspace(-4,1,121)])
    scores=[]
    for rate in candidates:
        error=0.
        for h,seconds in enumerate(bank['horizons']):
            value=mean+torch.exp(-rate.double()*seconds)*(z-mean)
            error+=((value-target[:,h]).square().sum(-1)*weights[h]).sum()
        scores.append(float(error))
    result['continuous_rate']=float(candidates[min(range(len(scores)),key=scores.__getitem__)])
    for key in ('shrink','current_linear','history_linear','center_amplitude'):
        result[key]=torch.stack(result[key]) if torch.is_tensor(result[key][0]) else torch.tensor(result[key])
    return result


@torch.no_grad()
def refit_output_calibration(fitted,bank,core,settings,device):
    result=dict(fitted);z=bank['states'].z.double();mean=fitted['mean'].double()
    predicted=learned_predictions(core,bank,device,settings['prediction_batch'],settings.get('enable_partner',True)).double()
    values=[]
    for h in range(len(bank['horizons'])):
        w=domain_weights(bank['domain'],bank['valid'][:,h]);delta=predicted[:,h]-z
        center=mean-bank['states'].baseline.double()
        x=torch.stack([delta,center],-1).reshape(-1,2);y=(bank['targets'][:,h].double()-z).reshape(-1,1)
        ww=w[...,None].expand_as(delta).reshape(-1,1)
        values.append(torch.linalg.solve(x.T@(x*ww)+1e-8*torch.eye(2,dtype=torch.float64),x.T@(y*ww)).reshape(2).float())
    result['center_amplitude']=torch.stack(values)
    return result


def interpolate(values,seconds,horizons,zero):
    device=seconds.device
    values=values.to(device);zero=zero.to(device)
    knots=torch.tensor([0.,*horizons],device=device)
    table=torch.cat([zero[None],values],0)
    t=seconds.clamp(0,float(horizons[-1]));hi=torch.searchsorted(knots,t).clamp(1,len(knots)-1);lo=hi-1
    fraction=(t-knots[lo])/(knots[hi]-knots[lo])
    fraction=fraction.reshape(-1,*([1]*(values.ndim-1)))
    return table[lo]*(1-fraction)+table[hi]*fraction


def predict_all(core,state,history,seconds,fitted,learned=None,enable_partner=True):
    if not torch.is_tensor(seconds):seconds=torch.full((len(state.fast),),float(seconds),device=state.fast.device)
    seconds=seconds.to(state.fast.device)
    if learned is None:learned=core._propagate(state,seconds,enable_partner=enable_partner).z
    z=state.z;mean=fitted['mean'].to(z);h=fitted['horizons']
    current=torch.cat([z,z.flip(1)],-1)
    def linear(x,coefficient):
        x=torch.cat([x,torch.ones_like(x[...,:1])],-1)
        return torch.einsum('bri,bio->bro',x,coefficient)
    zero_current=torch.zeros(current.shape[-1]+1,z.shape[-1],device=z.device)
    zero_current[:z.shape[-1]]=torch.eye(z.shape[-1],device=z.device)
    zero_history=torch.zeros(history.shape[-1]+1,z.shape[-1],device=z.device)
    zero_history[:z.shape[-1]]=torch.eye(z.shape[-1],device=z.device)
    a=interpolate(fitted['shrink'],seconds,h,torch.tensor(1.)).reshape(-1,1,1)
    ca=interpolate(fitted['center_amplitude'],seconds,h,torch.zeros(2))
    fast_rate,slow_rate=fitted['decay_rates'].to(z)
    decay=state.baseline+state.fast*torch.exp(-seconds[:,None,None]*fast_rate)+state.slow*torch.exp(-seconds[:,None,None]*slow_rate)
    return dict(learned=learned,hold=z,decay=decay,mean=mean.expand_as(z),shrink=mean+a*(z-mean),
        continuous_shrink=mean+torch.exp(-fitted['continuous_rate']*seconds[:,None,None])*(z-mean),
        calibrated_hold=linear(current,fitted['calibrated_hold'].to(z)[None].expand(len(z),-1,-1)),
        current_linear=linear(current,interpolate(fitted['current_linear'],seconds,h,zero_current)),
        history_linear=linear(history.to(z),interpolate(fitted['history_linear'],seconds,h,zero_history)),
        center_amplitude=z+ca[:,0,None,None]*(learned-z)+ca[:,1,None,None]*(mean-state.baseline))
