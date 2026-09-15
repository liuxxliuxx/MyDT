"""Report observed history age separately from actual future endpoint distance."""
from collections import defaultdict
from emotion_ssm.train.dynamics_v3 import future_endpoint_queries
import torch


def endpoint_coverage(dialogues,horizons=(1,2,4,8,16,32),age_edges=(0,8,16,32,64)):
    cells={(a,h):dict(vector_queries=0,real_endpoint_queries=0,sources=set(),actual_seconds=[])
           for a in age_edges for h in horizons}
    lengths=[];event_pairs={h:dict(queries=0,sources=set()) for h in (16,32,64)}
    from emotion_ssm.train.dynamics_v3 import future_matches
    for identity,packets in dialogues:
        if not packets:continue
        times=[float(p['time'] if 'time' in p else p['end']) for p in packets]
        beginning=float(packets[0].get('start_time',packets[0].get('start',times[0]-float(packets[0].get('dt',1.)))))
        if times!=sorted(set(times)):raise ValueError('History audit requires a strictly increasing physical timeline')
        lengths.append(dict(source=identity,seconds=times[-1]-beginning,blocks=len(times)))
        queries=future_endpoint_queries(times,packets,list(horizons))
        event_queries=future_endpoint_queries(times,packets,[16,32,64])
        for origin,following in event_queries.items():
            roles=packets[origin].get('roles',[])
            for query in following:
                partner=1-query['role']
                if len(roles)==2 and isinstance(roles[partner],dict) and bool(torch.as_tensor(roles[partner].get('event_present',False)).any()):
                    cell=event_pairs[query['nominal_horizon']];cell['queries']+=1;cell['sources'].add(identity.split('_sub_video_',1)[0])
        for tick,now in enumerate(times):
            age=now-beginning;bucket=max(a for a in age_edges if age>=a)
            for h,_ in future_matches(times,tick,list(horizons)):
                cell=cells[bucket,h];cell['vector_queries']+=1;cell['sources'].add(identity.split('_sub_video_',1)[0])
            for query in queries.get(tick,[]):
                cell=cells[bucket,query['nominal_horizon']];cell['real_endpoint_queries']+=1
                cell['sources'].add(identity.split('_sub_video_',1)[0]);cell['actual_seconds'].append(query['seconds'])
    rows=[]
    for (a,h),cell in cells.items():
        seconds=cell.pop('actual_seconds');sources=cell.pop('sources')
        rows.append(dict(history_age_start=a,nominal_horizon=h,**cell,independent_sources=len(sources),
            source_ids=sorted(sources),actual_seconds_min=min(seconds) if seconds else None,
            actual_seconds_max=max(seconds) if seconds else None))
    return dict(protocol='real-history-age-by-endpoint-v1',dialogues=lengths,cells=rows,
                vector_count_meaning='Timestamp candidates; teacher validity is counted after bank encoding',
                partner_event_to_later_endpoint=[dict(nominal_horizon=h,queries=v['queries'],independent_sources=len(v['sources'])) for h,v in event_pairs.items()],
                event_pair_interpretation='Observed event and later partner label availability; these are not causal influence labels.',
                missing_long_data=not any(r['history_age_start']>=32 and r['real_endpoint_queries'] for r in rows))


def bank_coverage(bank,age_edges=(0,8,16,32,64)):
    rows=[]
    for a,b in zip(age_edges,(*age_edges[1:],float('inf'))):
        for h_index,h in enumerate(bank['horizons']):
            selected=(bank['history_age']>=a)&(bank['history_age']<b)
            vector=selected&bank['valid'][:,h_index].any(-1)
            queries=[(r,q) for r,q in bank['endpoints'] if selected[r] and q['nominal_horizon']==h]
            rows.append(dict(history_age_start=a,nominal_horizon=h,vector_origins=int(vector.sum()),
                real_endpoint_queries=len(queries),independent_sources=len({bank['keys'][r][0].split('_sub_video_',1)[0] for r,q in queries})))
    return rows
