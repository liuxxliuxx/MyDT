"""Source-balanced windows with causal prefix replay, without longer Avatar graphs."""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import random
from collections import OrderedDict

import torch

from emotion_ssm.train.generation_sampling import source_id

PROTOCOL='history-age-prefix-windows-v1'


class HistoryWindowCursor:
    def __init__(self,dataset,seed=6666,rank=0,world_size=1,conversations=4,blocks_per_conversation=4,
                 config=None,saved=None):
        self.dataset=dataset;self.seed=seed;self.rank=rank;self.world=world_size
        self.conversations=conversations;self.blocks=blocks_per_conversation
        self.config=dict(config or {})
        self.age_edges=list(self.config.get('age_edges',[0,8,16,32,64]))
        self.natural_probability=float(self.config.get('natural_probability',.5))
        if self.age_edges[0]!=0 or self.age_edges!=sorted(set(self.age_edges)) or not 0<=self.natural_probability<=1:
            raise ValueError('Invalid history age sampling configuration')
        sources=getattr(dataset,'source_ids',[source_id(n) for n in dataset.names])
        self.sources=sources
        owners={s:i%world_size for i,s in enumerate(sorted(set(sources)))}
        self.indices=[i for i,s in enumerate(sources) if owners[s]==rank]
        self.rng=random.Random(seed+rank);self.seen=0;self.draws=0
        self.cache=OrderedDict();self.candidates=[]
        # Cache only deterministic data-index metadata, never model features,
        # state, sampled windows or RNG. All controls share the same index.
        cache_root=self.config.get('candidate_cache_directory')
        identity=dict(protocol=PROTOCOL,manifest_digest=dataset.manifest_digest,
                      names=dataset.names,sources=self.sources,indices=self.indices,
                      settings=self.settings())
        token_dataset=getattr(dataset,'dataset',None)
        if cache_root and token_dataset is not None and hasattr(token_dataset,'manifest'):
            signatures=[]
            for i in self.indices:
                item=token_dataset.manifest['dialogues'][token_dataset.ids[dataset.indices[i]]]
                path=token_dataset.root/item['path'];stat=path.stat()
                signatures.append((str(path.resolve()),stat.st_size,stat.st_mtime_ns))
            identity['token_files']=[list(row) for row in signatures]
        encoded=json.dumps(identity,sort_keys=True,separators=(',',':')).encode()
        cache_key=hashlib.sha256(encoded).hexdigest()
        cache_file=Path(cache_root)/(cache_key+'.json') if cache_root else None
        if cache_file and cache_file.exists():
            cached=json.loads(cache_file.read_text(encoding='utf-8'))
            content=json.dumps(cached['candidates'],separators=(',',':')).encode()
            if cached['identity']!=identity or cached['sha256']!=hashlib.sha256(content).hexdigest():
                raise ValueError('History candidate cache identity/content mismatch')
            self.candidates=[tuple(row) for row in cached['candidates']]
        for i in ([] if self.candidates else self.indices):
            rows=self._rows(i)
            valid=[j for j,(_,_,mask) in enumerate(rows) if mask.any()]
            if len(valid)<self.blocks:
                continue
            for k,start in enumerate(valid[:len(valid)-self.blocks+1]):
                end=valid[k+self.blocks-1]+1
                age=float(rows[start][0]['time'])-rows[start][1].shape[1]/25-float(rows[0][0].get('start_time',0.))
                bucket=max(j for j,e in enumerate(self.age_edges) if age>=e-1e-6)
                self.candidates.append((i,start,end,bucket,age))
        if cache_file and not cache_file.exists():
            content=json.dumps(self.candidates,separators=(',',':')).encode()
            cache_file.parent.mkdir(parents=True,exist_ok=True)
            temporary=cache_file.with_suffix('.tmp')
            temporary.write_text(json.dumps(dict(identity=identity,candidates=self.candidates,
                sha256=hashlib.sha256(content).hexdigest())),encoding='utf-8')
            temporary.replace(cache_file)
        if not self.candidates:
            raise ValueError('No complete valid history windows in this source shard')
        if saved:
            if saved['protocol']!=PROTOCOL or saved['manifest_digest']!=dataset.manifest_digest or saved['config']!=self.settings():
                raise ValueError('History sampler protocol/data changed on resume')
            self.rng.setstate(saved['rng']);self.seen=saved['seen'];self.draws=saved['draws']

    def settings(self):
        return dict(seed=self.seed,rank=self.rank,world=self.world,conversations=self.conversations,
                    blocks=self.blocks,age_edges=self.age_edges,natural_probability=self.natural_probability)

    def _rows(self,index):
        if index not in self.cache:
            self.cache[index]=list(self.dataset.packets(index))
            while len(self.cache)>int(self.config.get('cached_dialogues',2)):
                self.cache.popitem(last=False)
        self.cache.move_to_end(index)
        return self.cache[index]

    def take_valid(self,count):
        if count!=self.conversations*self.blocks:
            raise ValueError('History windows preserve the declared valid new-block budget')
        result=[];used=set()
        for _ in range(self.conversations):
            choices=[c for c in self.candidates if self.sources[c[0]] not in used] or self.candidates
            if self.rng.random()>=self.natural_probability:
                bucket=self.rng.choice(sorted({c[3] for c in choices}))
                choices=[c for c in choices if c[3]==bucket]
                source=self.rng.choice(sorted({self.sources[c[0]] for c in choices}))
                choices=[c for c in choices if self.sources[c[0]]==source]
            i,start,end,bucket,age=self.rng.choice(choices)
            used.add(self.sources[i]);self.draws+=1
            rows=self._rows(i)
            session=f'history-window:{self.rank}:{self.draws}:{rows[start][0]["session_id"]}'
            prefix=[]
            for p,_,_ in rows[:start]:
                p=copy.deepcopy(p);p['session_id']=session;prefix.append(p)
            for position,(p,target,mask) in enumerate(rows[start:end]):
                p=copy.deepcopy(p)
                p.update(session_id=session,training_source_id=self.sources[i],training_record_name=self.dataset.names[i],
                         training_session_end=position==end-start-1,history_age_seconds=age,
                         history_age_bucket=bucket,history_window_origin=start)
                if position==0:
                    p['training_prefix']=prefix
                result.append((p,target,mask));self.seen+=1
        return result

    def peek(self,count):
        if count:
            raise ValueError('Frozen history windows do not support joint future lookahead')
        return []

    def state_dict(self):
        return dict(protocol=PROTOCOL,config=self.settings(),manifest_digest=self.dataset.manifest_digest,
                    rng=self.rng.getstate(),seen=self.seen,draws=self.draws)

    def coverage(self):
        result=[]
        for b,a in enumerate(self.age_edges):
            rows=[c for c in self.candidates if c[3]==b]
            result.append(dict(age_start=a,windows=len(rows),sources=len({self.sources[c[0]] for c in rows})))
        return result
