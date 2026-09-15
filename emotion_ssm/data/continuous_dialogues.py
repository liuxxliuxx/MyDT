"""Verified physical timelines; file numbers never imply temporal adjacency."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import torch

from emotion_ssm.data.packets_v3 import empty_role_features, collate_role_features

PROTOCOL='verified-continuous-dialogues-v1'


def read_timeline(value, dataset):
    data=copy.deepcopy(value) if isinstance(value,dict) else json.loads(Path(value).read_text(encoding='utf-8'))
    if data.get('protocol')!=PROTOCOL or data.get('split')!=dataset.split:
        raise ValueError('Timeline split/protocol mismatch')
    if data.get('token_manifest_digest')!=dataset.manifest_digest:
        raise ValueError('Timeline belongs to a different token/feature manifest')
    names=set(dataset.names);used=set();ids=set()
    for row in data.get('conversations',[]):
        if row['session_id'] in ids or len(row['roles'])!=2 or len(set(row['roles']))!=2:
            raise ValueError('Duplicate conversation or ambiguous role identity')
        ids.add(row['session_id']);end=None
        if not row.get('source_id') or not row.get('segments'):
            raise ValueError('A continuous conversation needs source identity and segments')
        for part in row['segments']:
            if part['name'] not in names or part['name'] in used:
                raise ValueError('Unavailable or repeated directed clip in timeline')
            used.add(part['name'])
            if not part.get('verified') or not part.get('evidence') or tuple(part['roles'])!=tuple(row['roles']):
                raise ValueError('Continuity and ordered role mapping require explicit evidence')
            a,b=float(part['start']),float(part['end'])
            if b<=a or any(abs(x*25-round(x*25))>1e-4 for x in (a,b)) or (end is not None and a<end-1e-8):
                raise ValueError('Timeline overlaps or is not on the 25 FPS clock')
            end=b
    if not ids:
        raise ValueError('No verified continuous conversations; provide original timestamps and role mapping')
    return data


class ContinuousDialogueDataset:
    prefetch_rng_neutral=True

    def __init__(self, dataset, timeline):
        self.base=dataset
        self.timeline=read_timeline(timeline,dataset)
        self.split=dataset.split
        self.names=[r['session_id'] for r in self.timeline['conversations']]
        self.source_ids=[r['source_id'] for r in self.timeline['conversations']]
        self.lookup={name:i for i,name in enumerate(dataset.names)}
        self.lengths=[]
        for row in self.timeline['conversations']:
            n=sum(dataset.lengths[self.lookup[p['name']]] for p in row['segments'])
            # Gaps may end with a fractional-second packet.
            n+=sum((max(0,round((q['start']-p['end'])*25))+24)//25 for p,q in zip(row['segments'],row['segments'][1:]))
            self.lengths.append(n)
        self.feature_sources=dataset.feature_sources
        raw=json.dumps(self.timeline,sort_keys=True,separators=(',',':')).encode()
        self.manifest_digest=hashlib.sha256(raw).hexdigest()
        self.feature_digest=dataset.feature_digest

    def __len__(self):
        return len(self.names)

    def _missing(self,row,start,end,template):
        n=round((end-start)*25)
        packet=dict(session_id=row['session_id'],roles=tuple(row['roles']),start_time=row['segments'][0]['start'],time=end,
                    target_audio=torch.zeros(1,n*640),partner_audio=torch.zeros(1,n*640),
                    target_speech_active=False,partner_speech_active=False,
                    partner_blendshape=torch.zeros(1,n,56),partner_visual_mask=torch.zeros(1,n,dtype=torch.bool),
                    source_domain=template.get('source_domain',2),feature_source=template.get('feature_source'),
                    physical_gap=True,training_source_id=row['source_id'],continuous_timeline=True)
        for role in ('target','partner'):
            f=template[role+'_features']
            empty=empty_role_features(f['audio_tokens'].shape[-1],f['text_tokens'].shape[-1],end,end-start,int(f['domain_id'][0]))
            packet[role+'_features']=collate_role_features([empty])
        return packet,torch.zeros(1,n,56),torch.zeros(1,n,dtype=torch.bool)

    def packets(self,index):
        row=self.timeline['conversations'][index]
        previous_end=row['segments'][0]['start']
        text=[{},{}]
        for part in row['segments']:
            iterator=iter(self.base.packets(self.lookup[part['name']]))
            first=next(iterator)
            import itertools
            while previous_end<part['start']-1e-8:
                end=min(part['start'],previous_end+1.)
                yield self._missing(row,previous_end,end,first[0])
                previous_end=end
            for raw,target,valid in itertools.chain([first],iterator):
                packet=copy.deepcopy(raw)
                local_time=float(raw['time']);now=part['start']+local_time
                if now>part['end']+1e-6:
                    raise ValueError('Declared timeline duration differs from actual frames')
                packet.update(session_id=row['session_id'],roles=tuple(row['roles']),time=now,
                              start_time=row['segments'][0]['start'],training_source_id=row['source_id'],continuous_timeline=True)
                for role,prefix in enumerate(('target','partner')):
                    f=packet[prefix+'_features']
                    for key in list(f):
                        if key.endswith('_times') or key in ('audio_starts','now'):
                            f[key]=f[key]+part['start']
                    f['available_at']=now
                    # Only fresh acoustic atoms enter the persistent stream history.
                    keep=f['audio_times']>previous_end+1e-8
                    if not keep.any():
                        keep[:, -1]=True
                        for mode in ('audio','prosody'):
                            f[mode+'_mask'][:,-1]=False
                            f[mode+'_times'][:,-1]=now
                        f['audio_starts'][:,-1]=previous_end
                    for mode in ('audio','prosody'):
                        for suffix in ('_tokens','_mask','_times'):
                            f[mode+suffix]=f[mode+suffix][:,keep[0]]
                        if mode=='audio':
                            f['audio_starts']=f['audio_starts'][:,keep[0]]
                            f['audio_fresh_mask']=f['audio_fresh_mask'][:,keep[0]]
                    f['audio_history_complete']=torch.tensor([False])
                    history=text[role]
                    for j in f['text_mask'][0].nonzero().flatten().tolist():
                        identity=(part['name'],int(f['text_positions'][0,j]))
                        if identity not in history:
                            history[identity]=(len(history),f['text_tokens'][0,j],f['text_times'][0,j],f['text_roles'][0,j])
                    kept=sorted(history.items(),key=lambda item:item[1][0])[-256:]
                    if kept:
                        values=[v for _,v in kept]
                        f['text_tokens']=torch.stack([v[1] for v in values])[None]
                        f['text_times']=torch.stack([v[2] for v in values])[None]
                        f['text_roles']=torch.stack([v[3] for v in values])[None]
                        f['text_positions']=torch.tensor([[v[0] for v in values]])
                        f['text_mask']=torch.ones(1,len(values),dtype=torch.bool)
                        f['text_fresh_mask']=f['text_times']>previous_end+1e-8
                        f['modality_mask'][:,2]=True
                        f['fresh_observation'][:,2]=f['text_fresh_mask'].any(-1)
                    if 'event_ids' in f:
                        f['event_ids']=[part['name']+':'+str(x) for x in f['event_ids']]
                if packet.get('words'):
                    for word in packet['words']:
                        word['id']=part['name']+':'+str(word.get('id',''))
                        for key in ('start','end','available_at'):
                            word[key]+=part['start']
                        word['role']=row['roles'][tuple(raw['roles']).index(word['role'])]
                previous_end=now
                yield packet,target,valid
            if abs(previous_end-part['end'])>1e-6:
                raise ValueError('Timeline end does not match actual clip duration')
