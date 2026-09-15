"""Replay memory ablations on identical causal observer outputs."""
from collections import deque
from dataclasses import fields
import torch
from emotion_ssm.models.state_core import StateObservation


class HistoryConditionReplay:
    modes=('state_recent4','state_recent16','state_recent32','uncoupled_history')
    def __init__(self,core,variant='dyadic'):
        self.core=core;self.variant=variant;self.rows=deque();self.uncoupled=None

    def advance(self,observations,now,dt):
        # TokenObserver's legacy EventObservation carries freshness/action as
        # runtime attributes. Convert to the complete V3 type before copying.
        pair=[]
        for observation in observations:
            values={f.name:getattr(observation,f.name,None) for f in fields(StateObservation)}
            pair.append(StateObservation(**{k:v.detach() if torch.is_tensor(v) else v for k,v in values.items()}))
        self.rows.append((float(now),float(dt),pair))
        while self.rows and self.rows[0][0]<=now-32+1e-8:self.rows.popleft()
        def initial():return self.core.initialize(len(pair[0].aff),pair[0].aff.device,pair[0].aff.dtype)
        if self.uncoupled is None:self.uncoupled=initial()
        self.uncoupled=self.core.advance(self.uncoupled,pair,dt,enable_partner=False)
        result={'uncoupled_history':self.core.context(self.uncoupled,pair,variant=self.variant)}
        for window in (4,16,32):
            state=initial()
            for end,delta,inputs in self.rows:
                # Include only complete physical observations inside the window.
                if end-delta>=now-window-1e-8:
                    state=self.core.advance(state,inputs,delta,enable_partner=self.variant=='dyadic')
            result['state_recent'+str(window)]=self.core.context(state,pair,variant=self.variant)
        return result
