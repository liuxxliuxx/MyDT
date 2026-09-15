"""Batch only causal, no-gradient origin replay; query sampling is unchanged."""
from dataclasses import fields
import torch
from emotion_ssm.models.state_core import EmotionMemory, StateObservation
from emotion_ssm.train.dynamics_v3 import future_matches,unit_state_readout
from emotion_ssm.train.staged_dynamics_support import OBS_FIELDS,memory_cat,memory_index,weight_digest
from emotion_ssm.train.staged_v33.data import FeatureViews,QuerySampler,assemble,history_vector,planned_queries,replay_origins
from emotion_ssm.train.staged_v33.missing import digest,update_missing


def replay_current_origins(observer,core,bank,rows,device,enable_partner=True,with_grad=False):
    """Replay shared full prefixes with current weights, then optional TBPTT.

    Prefixes are recomputed once per selected dialogue, not once per query.
    Old bank snapshots cannot change the deployed origin distribution here.
    """
    rows=list(map(int,rows));requested={};snapshots={}
    for row in rows:
        record,start,stop=bank['replay_rows'][row]
        requested.setdefault(record,set()).add(start if with_grad else stop)
    records=sorted(requested);encoded=[bank['encoded_rows'][r] for r in records]
    with torch.no_grad():
        state=core.initialize(len(records),device)
        for i,record in enumerate(records):
            if 0 in requested[record]:snapshots[record,0]=memory_index(state,slice(i,i+1)).detach()
        for tick in range(max(max(v) for v in requested.values())):
            state,_,_=step_batch(observer,core,encoded,state,tick,device,enable_partner)
            for i,record in enumerate(records):
                if tick+1 in requested[record]:snapshots[record,tick+1]=memory_index(state,slice(i,i+1)).detach()
    state=memory_cat([snapshots[record,start if with_grad else stop]
                     for record,start,stop in (bank['replay_rows'][r] for r in rows)])
    if not with_grad:return state.detach()
    replay_bank={**bank,'prefixes':state,'replay_rows':[bank['replay_rows'][r] for r in rows]}
    return replay_origins(observer,core,replay_bank,range(len(rows)),device,enable_partner)


def step_batch(observer,core,encoded_rows,state,tick,device,partner=True):
    """Advance active dialogues once; padding cannot change any memory field."""
    observations=[[],[]];active=[];intervals=[]
    for encoded in encoded_rows:
        live=tick<len(encoded['times']);used=min(tick,len(encoded['times'])-1)
        active.append(live);intervals.append(encoded['dt'][tick] if live else 0.)
        for role in (0,1):
            values={k:encoded['observations'][k][used,role] for k in (*OBS_FIELDS,'event_input','action_input')}
            if not live or update_missing(encoded['plan'],encoded['times'][used],role):
                values={**values,**{k:torch.zeros_like(values[k]) for k in
                    ('modality_mask','fresh_observation','event_present','action_present','action_duration')}}
            observations[role].append(values)
    pairs=[]
    for values in observations:
        batch={k:torch.stack([v[k] for v in values]).to(device) for k in values[0]}
        batch['event']=observer.event_head(batch.pop('event_input'))*batch['event_present'][:,None]
        batch['action']=observer.action_head(batch.pop('action_input'))*batch['action_present'][:,None]
        batch['event_id']=torch.full((len(encoded_rows),),tick,device=device,dtype=torch.long)
        pairs.append(StateObservation(**batch))
    diagnostic={'include_autonomous_prior':True}
    updated=core.advance(state,pairs,torch.tensor(intervals,device=device),enable_partner=partner,diagnostics=diagnostic)
    mask=torch.tensor(active,device=device)
    selected=EmotionMemory(**{f.name:torch.where(mask.reshape(-1,*([1]*(getattr(state,f.name).ndim-1))),
        getattr(updated,f.name),getattr(state,f.name)) for f in fields(EmotionMemory)})
    return selected,diagnostic,mask


@torch.no_grad()
def build_bank(observer,core,cache,collection,indices,horizons,stride=4,protocol='clean',enable_partner=True,progress=None):
    encoded_rows=[cache.get(collection,i,protocol) for i in indices]
    memories_by_row=[[] for _ in indices];past_by_row=[[] for _ in indices]
    current=torch.zeros(7,device=cache.device,dtype=torch.float64)
    batch=max(1,int(cache.settings.get('replay_dialogue_batch',8)))
    for start in range(0,len(indices),batch):
        group=encoded_rows[start:start+batch];state=core.initialize(len(group),cache.device)
        initial=state.detach().to('cpu')
        for i in range(len(group)):past_by_row[start+i].append(memory_index(initial,slice(i,i+1)))
        for tick in range(max(len(e['times']) for e in group)):
            state,diagnostics,active=step_batch(observer,core,group,state,tick,cache.device,enable_partner)
            mask=torch.stack([e['valid'][min(tick,len(e['times'])-1)] for e in group]).to(cache.device)&active[:,None]
            truth=torch.stack([e['gold'][min(tick,len(e['times'])-1)] for e in group]).to(cache.device)
            for col,pred in enumerate((state.z,diagnostics['input_conditioned_prior'],diagnostics['autonomous_prior'])):
                current[col]+=(pred[mask]-truth[mask]).double().square().sum()
            current[3]+=mask.sum()*state.fast.shape[-1]
            current[4]+=diagnostics['correction_gain'][active].double().sum()
            current[5]+=diagnostics['correction_gain'][active].numel()
            current[6]+=diagnostics['evidence'][active].double().sum()
            saved=state.detach().to('cpu')
            for i,e in enumerate(group):
                if tick<len(e['times']):
                    local=memory_index(saved,slice(i,i+1))
                    memories_by_row[start+i].append(local);past_by_row[start+i].append(local)
        if progress:progress(min(start+batch,len(indices)),len(indices),sum(map(len,memories_by_row)))
    memories,records,histories,prefixes,replay_rows=[],[],[],[],[]
    available=set();planned=0
    for number,encoded in enumerate(encoded_rows):
        queries=planned_queries(encoded,horizons);planned+=sum(map(len,queries.values()))
        for labels in encoded['labels']:
            for role,values in enumerate(labels):
                for label in values:
                    if 0<=int(label.get('emotion',-1))<7 or any(label.get('vad_mask',[])) or label.get('intensity_mask',False):
                        available.add((encoded['identity'],role,float(label['end'])))
        history=[s.z for s in memories_by_row[number]]
        for tick,state in enumerate(memories_by_row[number]):
            vector=tick%stride==stride-1 and any(bool(encoded['valid'][j].any()) for _,j in future_matches(encoded['times'],tick,horizons))
            if vector or tick in queries:
                memories.append(state);records.append((number,tick));histories.append(history_vector(history,encoded['times'],tick))
                begin=max(0,tick+1-int(cache.settings.get('tbptt_events',32)))
                prefixes.append(past_by_row[number][begin]);replay_rows.append((number,begin,tick+1))
    producer=digest(dict(weights=weight_digest({'observer':observer,'state':core}),construction=core.get_config(),
        protocol=protocol,partner=enable_partner,plans=[e['plan']['sha256'] for e in encoded_rows]))
    bank=assemble(memories,records,encoded_rows,list(horizons),stride,producer,histories)
    bank.update(prefixes=memory_cat(prefixes),replay_rows=replay_rows,encoded_rows=encoded_rows)
    gold=torch.cat([e['gold'][e['valid']] for e in encoded_rows])
    bank.update(teacher_sum=gold.double().sum(0),teacher_count=len(gold),plans=[e['plan'] for e in encoded_rows],protocol=protocol,
        current=dict(zip(('posterior_sse','input_conditioned_prior_sse','autonomous_prior_sse','elements',
            'correction_gain_sum','correction_gain_count','evidence_sum'),current.cpu().tolist())))
    covered={(bank['keys'][r][0],q['role'],q['endpoint']) for r,q in bank['endpoints']}
    bank['coverage']=dict(available_endpoints=len(available),covered_endpoints=len(covered),planned_queries=planned,
        cached_queries=len(bank['endpoints']),queries_per_horizon={str(h):sum(q['nominal_horizon']==h for _,q in bank['endpoints']) for h in horizons},
        uncovered_no_causal_origin=len(available-covered))
    domain_by_id={e['identity']:e['domain'] for e in encoded_rows}
    bank['coverage']['by_domain']={str(d):dict(available_endpoints=sum(domain_by_id[k[0]]==d for k in available),
        covered_endpoints=sum(domain_by_id[k[0]]==d for k in covered),cached_queries_by_horizon={str(h):sum(int(bank['domain'][r])==d and q['nominal_horizon']==h
        for r,q in bank['endpoints']) for h in horizons}) for d in sorted(set(domain_by_id.values()))}
    if planned!=len(bank['endpoints']):raise AssertionError('Lost a real endpoint during batched replay')
    return bank


@torch.no_grad()
def origin_drift(observer,core,cache,collection,indices,bank,horizons,stride,protocol,settings):
    """Current full-history replay on fixed training probes, never validation."""
    grouped={}
    for i in indices:grouped.setdefault(collection.index[i][0],[]).append(i)
    chosen=[rows[0] for _,rows in sorted(grouped.items())]
    probe=build_bank(observer,core,cache,collection,chosen,horizons,stride,protocol,settings['enable_partner'])
    old={key:i for i,key in enumerate(bank['keys'])};rows=torch.tensor([old[k] for k in probe['keys']])
    before=memory_index(bank['states'],rows);after=probe['states']
    a=observer.decode_affect(unit_state_readout(before.z.to(cache.device)))['emotion_logits'].argmax(-1)
    b=observer.decode_affect(unit_state_readout(after.z.to(cache.device)))['emotion_logits'].argmax(-1)
    report=dict(z_mse=float((after.z-before.z).square().mean()),class_flip=float((a!=b).float().mean()),
        component_mse=float(torch.stack([(getattr(after,k)-getattr(before,k)).square().mean() for k in ('fast','slow','relation')]).mean()),
        probe_dialogues=[collection.identity(i) for i in chosen],origins=len(rows))
    report['refresh']=report['z_mse']>settings['origin_drift_mse'] or report['component_mse']>settings['origin_component_drift_mse'] or report['class_flip']>settings['origin_drift_class_flip']
    return report
