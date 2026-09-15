"""Independent vector/gold queries, reproducible feature views and origin replay."""
import bisect
import copy
import json
import os
from pathlib import Path
import random

import torch

from emotion_ssm.data.packets_v3 import collate_role_features
from emotion_ssm.models.state_core import EmotionMemory
from emotion_ssm.train.dynamics_v3 import future_matches, future_endpoint_queries
from emotion_ssm.train.staged_dynamics_support import (
    EncodedDialogueCache, OBS_FIELDS, cached_pair, memory_cat, weight_digest,
)
from .missing import digest, make_plan, mask_sensor_features, update_missing


class FeatureViews:
    def __init__(self, observer, teacher, directory, device, settings):
        self.base=EncodedDialogueCache(observer,teacher,directory,device,settings['cache_batch'])
        self.observer,self.teacher,self.device,self.settings=observer,teacher,device,settings
        self.loaded={}

    def get(self, collection, index, protocol='clean', epoch=0):
        identity=collection.identity(index)
        clean_key=(identity,'clean')
        if clean_key not in self.loaded:
            self.loaded[clean_key]=self.base.get(collection,index)
        clean=self.loaded[clean_key]
        plan=make_plan(identity,clean['times'][-1],protocol,
            self.settings['missing_seed']+epoch,self.settings['block_gap_period'],self.settings['block_gap_seconds'])
        key=(identity,plan['sha256'])
        if key in self.loaded:
            return self.loaded[key]
        result={**clean,'plan':plan}
        if any(i['kind']=='sensor_gap' for i in plan['intervals']):
            domain,_=collection.index[index]
            manifest=collection.datasets[domain].manifest
            for source in manifest['feature_sources'].values():
                if isinstance(source,str) and source.startswith('synthetic-'):
                    continue
                if not isinstance(source,dict) or source.get('token_audio')!='independent-500ms-full-backbone-v1' or source.get('token_text')!='lexical-input-embedding-explicit-order-v2':
                    raise ValueError('sensor_gap requires verified independent acoustic/lexical dependencies; rebuild incompatible tokens')
            metadata=dict(clean=clean['metadata'],plan=plan,revision='sensor-before-observer-v1')
            path=self.base.directory/('sensor_'+digest(metadata)+'.pt')
            if path.exists():
                cached=torch.load(path,map_location='cpu',weights_only=False)
                if cached['view_metadata']!=metadata:
                    raise ValueError('Sensor feature cache provenance changed')
                result=cached
            else:
                packets=collection[index]['packets']
                fields={k:[] for k in (*OBS_FIELDS,'event_input','action_input')}
                with torch.no_grad():
                    for start in range(0,len(packets),max(1,self.settings['cache_batch']//2)):
                        features=[mask_sensor_features(p['roles'][role],role,plan)
                            for p in packets[start:start+max(1,self.settings['cache_batch']//2)] for role in (0,1)]
                        batch=collate_role_features(features,self.device)
                        encoded=self.observer.encode(batch,subset='AVT')
                        for name in OBS_FIELDS:
                            fields[name].append(getattr(encoded['observation'],name).detach().cpu())
                        fields['event_input'].append(encoded['head_inputs']['event'].detach().cpu())
                        fields['action_input'].append(encoded['head_inputs']['action'].detach().cpu())
                result['observations']={k:torch.cat(v).reshape(len(packets),2,*v[0].shape[1:]) for k,v in fields.items()}
                result['view_metadata']=metadata
                temporary=path.with_suffix(f'.{os.getpid()}.tmp')
                torch.save(result,temporary);temporary.replace(path)
        self.loaded[key]=result
        return result


def pair_at(observer, encoded, tick, device):
    pair=cached_pair(observer,encoded,tick,device)
    for role,observation in enumerate(pair):
        if update_missing(encoded['plan'],encoded['times'][tick],role):
            for name in ('modality_mask','fresh_observation','event_present','action_present','action_duration'):
                setattr(observation,name,torch.zeros_like(getattr(observation,name)))
    return pair


def planned_queries(encoded,horizons):
    key=tuple(horizons)
    encoded.setdefault('_endpoint_plans',{})
    if key not in encoded['_endpoint_plans']:
        encoded['_endpoint_plans'][key]=future_endpoint_queries(encoded['times'],
            [{'targets':labels} for labels in encoded['labels']],horizons)
    return encoded['_endpoint_plans'][key]


def history_vector(z_history,times,tick,lags=(2.,8.,16.)):
    values=[];flags=[]
    for lag in (0.,*lags):
        prior=bisect.bisect_right(times,times[tick]-lag+1e-8)-1
        if prior>=0 and prior<len(z_history):
            z=z_history[prior]
            values.extend([z,z.flip(1)])
            flags.append(torch.ones_like(z[...,:1]))
        else:
            z=z_history[tick]
            values.extend([torch.zeros_like(z),torch.zeros_like(z)])
            flags.append(torch.zeros_like(z[...,:1]))
    return torch.cat(values+flags,-1)


def assemble(memories,records,encoded_rows,horizons,stride,producer,histories=None,all_vectors=False):
    if not memories:
        raise ValueError('No causal state origins')
    states=memory_cat(memories)
    n,d=len(memories),states.fast.shape[-1]
    targets=torch.zeros(n,len(horizons),2,d)
    valid=torch.zeros(n,len(horizons),2,dtype=torch.bool)
    current_gold,current_valid,domains,keys,endpoints=[],[],[],[],[]
    for row,(record,tick) in enumerate(records):
        encoded=encoded_rows[record]
        if all_vectors or tick%stride==stride-1:
            for h,future in future_matches(encoded['times'],tick,horizons):
                col=horizons.index(h)
                targets[row,col]=encoded['gold'][future]
                valid[row,col]=encoded['valid'][future]
        for query in planned_queries(encoded,horizons).get(tick,[]):
            endpoints.append((row,copy.deepcopy(query)))
        current_gold.append(encoded['gold'][tick]);current_valid.append(encoded['valid'][tick])
        domains.append(encoded['domain']);keys.append((encoded['identity'],encoded['times'][tick]))
    ages=torch.tensor([encoded_rows[r]['times'][t]-(encoded_rows[r]['times'][0]-encoded_rows[r]['dt'][0])
                       for r,t in records],dtype=torch.float64)
    return dict(states=states,targets=targets,valid=valid,domain=torch.tensor(domains),keys=keys,history_age=ages,
        endpoints=endpoints,horizons=list(horizons),producer=producer,
        current_gold=torch.stack(current_gold),current_valid=torch.stack(current_valid),
        history=torch.cat(histories) if histories else torch.cat([states.z,states.z.flip(1)]*4,-1),
        query_revision='vector_stride_gold_independent_v1')


@torch.no_grad()
def build_bank(observer,core,cache,collection,indices,horizons,stride=4,protocol='clean',
               enable_partner=True,progress=None):
    memories,records,encoded_rows,histories,prefixes,replay_rows=[],[],[],[],[],[]
    current=torch.zeros(7,dtype=torch.float64,device=cache.device)
    available_endpoints=set();planned=0
    for number,index in enumerate(indices):
        encoded=cache.get(collection,index,protocol)
        encoded_rows.append(encoded)
        queries=planned_queries(encoded,horizons)
        planned+=sum(len(rows) for rows in queries.values())
        for labels in encoded['labels']:
            for role,values in enumerate(labels):
                for label in values:
                    if 0<=int(label.get('emotion',-1))<7 or any(label.get('vad_mask',[])) or label.get('intensity_mask',False):
                        available_endpoints.add((encoded['identity'],role,float(label['end'])))
        state=core.initialize(1,cache.device);history=[];past=[state.detach().to('cpu')]
        for tick,dt in enumerate(encoded['dt']):
            diagnostics={'include_autonomous_prior':True}
            state=core.advance(state,pair_at(observer,encoded,tick,cache.device),dt,
                               enable_partner=enable_partner,diagnostics=diagnostics)
            history.append(state.z.detach().cpu())
            past.append(state.detach().to('cpu'))
            mask=encoded['valid'][tick].to(cache.device)[None]
            truth=encoded['gold'][tick].to(cache.device)[None]
            for col,pred in enumerate((state.z,diagnostics['input_conditioned_prior'],diagnostics['autonomous_prior'])):
                current[col]+=(pred[mask]-truth[mask]).double().square().sum()
            current[3]+=mask.sum()*state.fast.shape[-1]
            current[4]+=diagnostics['correction_gain'].double().sum()
            current[5]+=diagnostics['correction_gain'].numel()
            current[6]+=diagnostics['evidence'].double().sum()
            vector=tick%stride==stride-1 and any(bool(encoded['valid'][j].any()) for _,j in future_matches(encoded['times'],tick,horizons))
            if vector or tick in queries:
                memories.append(state.detach().to('cpu'));records.append((number,tick))
                histories.append(history_vector(history,encoded['times'],tick))
                start=max(0,tick+1-int(cache.settings.get('tbptt_events',32)))
                prefixes.append(past[start]);replay_rows.append((number,start,tick+1))
        if progress:
            progress(number+1,len(indices),len(memories))
    producer=digest(dict(weights=weight_digest({'observer':observer,'state':core}),
        construction=core.get_config(),protocol=protocol,partner=enable_partner,
        plans=[e['plan']['sha256'] for e in encoded_rows]))
    bank=assemble(memories,records,encoded_rows,list(horizons),stride,producer,histories)
    bank.update(prefixes=memory_cat(prefixes),replay_rows=replay_rows,encoded_rows=encoded_rows)
    gold=torch.cat([e['gold'][e['valid']] for e in encoded_rows])
    bank.update(teacher_sum=gold.double().sum(0),teacher_count=len(gold),
        plans=[e['plan'] for e in encoded_rows],protocol=protocol,
        current=dict(zip(('posterior_sse','input_conditioned_prior_sse','autonomous_prior_sse',
                          'elements','correction_gain_sum','correction_gain_count','evidence_sum'),current.cpu().tolist())))
    covered={(bank['keys'][row][0],q['role'],q['endpoint']) for row,q in bank['endpoints']}
    by_h={str(h):sum(q['nominal_horizon']==h for _,q in bank['endpoints']) for h in horizons}
    bank['coverage']=dict(available_endpoints=len(available_endpoints),covered_endpoints=len(covered),
        planned_queries=planned,cached_queries=len(bank['endpoints']),queries_per_horizon=by_h,
        uncovered_no_causal_origin=len(available_endpoints-covered))
    domain_by_identity={e['identity']:e['domain'] for e in encoded_rows}
    bank['coverage']['by_domain']={str(domain):dict(
        available_endpoints=sum(domain_by_identity[key[0]]==domain for key in available_endpoints),
        covered_endpoints=sum(domain_by_identity[key[0]]==domain for key in covered),
        cached_queries_by_horizon={str(h):sum(int(bank['domain'][row])==domain and q['nominal_horizon']==h
            for row,q in bank['endpoints']) for h in horizons}) for domain in sorted(set(domain_by_identity.values()))}
    if planned!=len(bank['endpoints']):
        raise AssertionError('Gold endpoint plan lost a cached query')
    return bank


def replay_origins(observer,core,bank,rows,device,enable_partner=True):
    """Batched TBPTT from causal full-dialogue prefix snapshots, never clean-gap replay.

    The prefix producer is explicit and refreshed between update/prediction blocks.
    All original event IDs, dt and missing plans survive reconstruction.
    """
    from emotion_ssm.train.staged_dynamics_support import memory_index
    from emotion_ssm.models.state_core import StateObservation
    from dataclasses import fields
    rows=list(map(int,rows));state=memory_index(bank['prefixes'],torch.tensor(rows)).to(device)
    replay=[bank['replay_rows'][row] for row in rows]
    length=max(stop-start for _,start,stop in replay)
    for offset in range(length):
        observations=[[],[]];intervals=[];active=[];ticks=[]
        for local,(record,start,stop) in enumerate(replay):
            tick=start+offset;valid=tick<stop
            encoded=bank['encoded_rows'][record]
            used_tick=min(tick,stop-1);ticks.append(used_tick)
            intervals.append(encoded['dt'][tick] if valid else 0.)
            active.append(valid)
            for role in (0,1):
                values={k:encoded['observations'][k][used_tick,role] for k in (*OBS_FIELDS,'event_input','action_input')}
                if not valid or update_missing(encoded['plan'],encoded['times'][used_tick],role):
                    values={**values,**{k:torch.zeros_like(values[k]) for k in
                        ('modality_mask','fresh_observation','event_present','action_present','action_duration')}}
                observations[role].append(values)
        pairs=[]
        for values in observations:
            batch={k:torch.stack([v[k] for v in values]).to(device) for k in values[0]}
            event_input=batch.pop('event_input');action_input=batch.pop('action_input')
            batch['event']=observer.event_head(event_input)*batch['event_present'][:,None]
            batch['action']=observer.action_head(action_input)*batch['action_present'][:,None]
            batch['event_id']=torch.tensor(ticks,device=device,dtype=torch.long)
            pairs.append(StateObservation(**batch))
        # dt=0 still permits observations in advance(); explicitly select padded rows out.
        updated=core.advance(state,pairs,torch.tensor(intervals,device=device),enable_partner=enable_partner)
        mask=torch.tensor(active,device=device)
        state=EmotionMemory(**{f.name:torch.where(mask.reshape(-1,*([1]*(getattr(state,f.name).ndim-1))),
            getattr(updated,f.name),getattr(state,f.name)) for f in fields(EmotionMemory)})
    return state


class QuerySampler:
    """32 global origins, balanced domains and horizons; only assigned h counts."""
    def __init__(self,bank,seed,chronological=False,age_sampling=None):
        self.bank=bank;self.generator=torch.Generator().manual_seed(seed)
        self.step=0;self.visits={};self.domain_counts={};self.cell_counts={};self.gold_visits={}
        self.chronological=chronological;self.cursors={};self.seed=seed;self.orders={};self.gold_cells={};self.gold_endpoints={}
        self.vector={}
        self.age_sampling=dict(age_sampling or {})
        self.age_edges=self.age_sampling.get('age_edges',[0,8,16,32,64])
        if self.age_edges!=sorted(set(self.age_edges)) or self.age_edges[0]!=0:
            raise ValueError('History age bins must be strictly ordered from zero')
        if not 0<=self.age_sampling.get('natural_probability',.5)<=1:
            raise ValueError('Invalid natural history sampling probability')
        if self.age_sampling and 'history_age' not in bank:
            raise ValueError('Age sampling needs real causal history ages')
        self.age_visits={}
        for domain in sorted(set(bank['domain'].tolist())):
            pools={h:torch.where((bank['domain']==domain)&bank['valid'][:,h].any(-1))[0]
                   for h in range(len(bank['horizons']))}
            pools={h:p for h,p in pools.items() if len(p)}
            if pools:self.vector[domain]=pools
        self.gold={}
        for index,(row,q) in enumerate(bank['endpoints']):
            label=q['label'];domain=int(bank['domain'][row]);h=bank['horizons'].index(q['nominal_horizon'])
            tasks=[]
            if 0<=int(label.get('emotion',-1))<7:tasks.append('ce')
            if any(label.get('vad_mask',[])):tasks.append('vad')
            if label.get('intensity_mask',False):tasks.append('intensity')
            for task in tasks:self.gold.setdefault((domain,h,task),[]).append(index)

    def draw(self,cell,pool,gold=False):
        if self.age_sampling and torch.rand((),generator=self.generator).item()>=self.age_sampling.get('natural_probability',.5):
            bins={}
            for item in pool:
                row=self.bank['endpoints'][int(item)][0] if gold else int(item)
                age=float(self.bank['history_age'][row])
                bucket=max(i for i,e in enumerate(self.age_edges) if age>=e)
                bins.setdefault(bucket,[]).append(int(item))
            bucket=sorted(bins)[int(torch.randint(len(bins),(1,),generator=self.generator))]
            pool=bins[bucket];cell=(*cell,'age',bucket)
            sources={}
            for item in pool:
                row=self.bank['endpoints'][item][0] if gold else item
                identity=self.bank['keys'][row][0]
                source=identity.split('_sub_video_',1)[0]
                sources.setdefault(source,[]).append(item)
            source=sorted(sources)[int(torch.randint(len(sources),(1,),generator=self.generator))]
            pool=sources[source];cell=(*cell,'source',source)
        chosen=self._draw(cell,pool,gold)
        if self.age_sampling:
            row=self.bank['endpoints'][chosen][0] if gold else chosen
            b=max(i for i,e in enumerate(self.age_edges) if float(self.bank['history_age'][row])>=e)
            key=str((gold,cell[:2],self.age_edges[b]))
            self.age_visits[key]=self.age_visits.get(key,0)+1
        return chosen

    def _draw(self,cell,pool,gold=False):
        if not self.chronological:
            return int(pool[int(torch.randint(len(pool),(1,),generator=self.generator))])
        key=str((gold,cell));position=self.cursors.get(key,0);epoch=position//len(pool)
        # Each domain/horizon cursor visits complete source dialogues in time order.
        # Full-prefix state snapshots carry history before the 32-event graph window.
        def source(item):
            row=self.bank['endpoints'][int(item)][0] if gold else int(item)
            identity,now=self.bank['keys'][row]
            return digest([identity,self.seed,epoch]),now
        order_key=(key,epoch)
        if order_key not in self.orders:
            self.orders[order_key]=sorted(map(int,pool),key=source)
        ordered=self.orders[order_key]
        self.cursors[key]=position+1
        return ordered[position%len(pool)]

    def sample(self,budget,gold_budget,rank=0,world=1):
        domains=sorted(self.vector)
        if budget<sum(len(v) for v in self.vector.values()):
            raise ValueError('Vector budget cannot cover every supported domain/horizon cell')
        selected=[]
        for position,domain in enumerate(domains):
            quota=budget//len(domains)+int((position-self.step)%len(domains)<budget%len(domains))
            columns=sorted(self.vector[domain])
            if quota<len(columns):raise ValueError('Domain quota cannot cover available horizons')
            for j in range(quota):
                h=columns[(j+self.step)%len(columns)];pool=self.vector[domain][h]
                row=self.draw((domain,h),pool)
                selected.append((row,h))
                key=str((self.bank['keys'][row],h));self.visits[key]=self.visits.get(key,0)+1
                self.domain_counts[domain]=self.domain_counts.get(domain,0)+1
                cell=str((domain,h));self.cell_counts[cell]=self.cell_counts.get(cell,0)+1
        cells=sorted(self.gold);gold=[]
        if cells and gold_budget<len(cells):
            raise ValueError('Gold budget cannot cover domain/horizon/task cells')
        for j in range(gold_budget if cells else 0):
            cell=cells[(j+self.step)%len(cells)];pool=self.gold[cell]
            index=self.draw(cell,pool,gold=True)
            # Task assignment prevents another annotated task on this query
            # from being implicitly reweighted by stratified sampling.
            gold.append((index,cell[2]))
            key=str((index,cell[2]));self.gold_visits[key]=self.gold_visits.get(key,0)+1
            key=str(cell);self.gold_cells[key]=self.gold_cells.get(key,0)+1
            row,query=self.bank['endpoints'][index]
            endpoint=str((self.bank['keys'][row][0],query['endpoint_key']))
            self.gold_endpoints[endpoint]=self.gold_endpoints.get(endpoint,0)+1
        self.step+=1
        return selected[rank::world],gold[rank::world]

    def state_dict(self):
        return dict(rng=self.generator.get_state(),step=self.step,visits=self.visits,
            domain_counts=self.domain_counts,cell_counts=self.cell_counts,gold_visits=self.gold_visits,
            cursors=self.cursors,chronological=self.chronological,gold_cells=self.gold_cells,gold_endpoints=self.gold_endpoints,
            age_sampling=self.age_sampling,age_visits=self.age_visits)

    def load_state_dict(self,value):
        if value.get('age_sampling',{})!=self.age_sampling:
            raise ValueError('History age sampling changed; initialize a new experiment')
        self.age_visits=value.get('age_visits',{})
        self.generator.set_state(value['rng']);self.step=value['step']
        for k in ('visits','domain_counts','cell_counts','gold_visits'):setattr(self,k,value[k])
        self.cursors=value.get('cursors',{});self.chronological=value.get('chronological',self.chronological)
        self.gold_cells=value.get('gold_cells',{});self.gold_endpoints=value.get('gold_endpoints',{})

    def diagnostics(self):
        total=sum(self.domain_counts.values())
        return dict(history_age_visits=self.age_visits,domain_counts=self.domain_counts,domain_share={k:v/max(1,total) for k,v in self.domain_counts.items()},
            cell_counts=self.cell_counts,unique_origin_queries=len(self.visits),
            repeated_origin_queries=sum(max(0,n-1) for n in self.visits.values()),
            gold_task_visits=sum(self.gold_visits.values()),unique_gold_task_queries=len(self.gold_visits),
            gold_domain_horizon_task_visits=self.gold_cells,trained_unique_gold_endpoints=len(self.gold_endpoints))
