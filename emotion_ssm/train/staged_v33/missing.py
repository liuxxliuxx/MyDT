"""Deterministic sensor/update gaps, applied before contextual observation."""
import copy
import hashlib
import json
import math
import random
import torch

REVISION = 'dependency-aware-gap-plan-v1'


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,default=str).encode()).hexdigest()


def make_plan(identity, duration, protocol, seed, period=32., gap=8.):
    if protocol not in ('clean','update_gap','sensor_gap','train_mixed','train_context'):
        raise ValueError('Unknown missing-input protocol')
    if not 0 <= gap < period or period<=0:
        raise ValueError('Gap must be shorter than its positive period')
    rng = random.Random(int(digest([identity,seed,protocol])[:16],16))
    intervals=[]
    if protocol!='clean':
        for start in range(int(math.ceil(duration/period))):
            edge = start*period
            if protocol=='train_mixed' and rng.random()<.5:
                continue
            kind = rng.choice(['update_gap','sensor_gap']) if protocol=='train_mixed' else protocol
            offset = rng.uniform(0,period-gap) if protocol=='train_mixed' else period-gap
            roles = [rng.randrange(2)] if protocol=='train_mixed' and rng.random()<.5 else [0,1]
            modes = rng.choice([['A'],['T'],['V'],['A','T','V']]) if protocol=='train_mixed' else ['A','T','V']
            if protocol=='train_context':
                kind='sensor_gap';roles=[0,1]
                if rng.random()<.8:
                    offset=rng.uniform(0,period-gap)
                    modes=rng.choice([['A','T','V'],['A','T'],['A','V'],['T','V']])
                else:
                    offset=0.;modes=[rng.choice(['A','T','V'])]
            if kind=='update_gap':modes=['A','T','V']
            if edge+offset>=duration or gap==0:continue
            width=period if protocol=='train_context' and len(modes)==1 else gap
            intervals.append(dict(start=edge+offset,end=min(duration,edge+offset+width),
                                  roles=roles,modes=modes,kind=kind))
    plan=dict(revision=REVISION,identity=identity,protocol=protocol,seed=seed,
              period=period,gap=gap,intervals=intervals)
    plan['sha256']=digest(plan)
    return plan


def update_missing(plan, now, role):
    return any(i['kind']=='update_gap' and role in i['roles'] and i['start']<now<=i['end']
               for i in plan['intervals'])


def mask_sensor_features(features, role, plan):
    """Mask independent acoustic atoms/lexical embeddings by true dependencies.

    Text outage is defined at external-system availability time. Shared text
    is masked by sender role, not receiver. Unknown extractor contracts are
    rejected by the feature cache instead of assuming token independence.
    """
    result={k:v.clone() if isinstance(v,torch.Tensor) else copy.deepcopy(v) for k,v in features.items()}
    now,since=float(result['now']),float(result['now']-result['dt'])
    for interval in plan['intervals']:
        if interval['kind']!='sensor_gap' or interval['start']>=now:
            continue
        lo,hi=interval['start'],interval['end']
        if role in interval['roles']:
            if 'A' in interval['modes']:
                blocked=(result['audio_starts'].double()<hi)&(result['audio_times'].double()>lo)
                result['audio_mask'] &= ~blocked
                result['prosody_mask'] &= ~blocked
            if 'V' in interval['modes']:
                for mode in ('au','flame'):
                    blocked=(result[mode+'_times'].double()>lo)&(result[mode+'_times'].double()<=hi)
                    result[mode+'_mask'] &= ~blocked
        if 'T' in interval['modes']:
            sender=torch.where(result['text_roles']==0,role,1-role)
            blocked=(result['text_times'].double()>lo)&(result['text_times'].double()<=hi)
            role_mask=torch.zeros_like(blocked)
            for sender_role in interval['roles']:
                role_mask |= sender==sender_role
            result['text_mask'] &= ~(blocked&role_mask)
    for mode in ('audio','prosody','au','flame','text'):
        result[mode+'_tokens']=result[mode+'_tokens']*result[mode+'_mask'][:,None]
    result['audio_fresh_mask']=result['audio_mask']&(result['audio_times']>since+1e-9)
    result['text_fresh_mask']=result['text_mask']&(result['text_times']>since+1e-9)
    visual_fresh=any(bool((result[m+'_mask']&(result[m+'_times']>since+1e-9)).any()) for m in ('au','flame'))
    result['modality_mask']=torch.tensor([bool(result['audio_mask'].any()),
        bool(result['au_mask'].any()|result['flame_mask'].any()),bool(result['text_mask'].any())])
    result['fresh_observation']=torch.tensor([bool(result['audio_fresh_mask'].any()),visual_fresh,
                                           bool(result['text_fresh_mask'].any())])
    result['context_available']=result['modality_mask'][2].clone()
    result['event_present']=(result['text_fresh_mask']&(result['text_roles']==0)).any()
    action=bool(result['audio_fresh_mask'].any()) or visual_fresh
    result['action_present']=torch.tensor(action)
    audio_duration=((result['audio_times'].clamp(max=now)-result['audio_starts'].clamp(min=since)).clamp_min(0)*result['audio_fresh_mask']).sum()
    result['action_duration']=torch.tensor(min(now-since,max(float(audio_duration),now-since if visual_fresh else 0.)) if action else 0.)
    return result
