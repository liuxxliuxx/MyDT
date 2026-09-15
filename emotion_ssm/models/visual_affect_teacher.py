"""Frozen FLAME-only teacher. Target motion never enters the online observer."""
from __future__ import annotations

import copy
import hashlib
import json
import torch
from torch import nn

from emotion_ssm.data.packets_v3 import empty_role_features, collate_role_features
from emotion_ssm.models.token_observer import TokenObserver

FLAME_SCHEMA = "flame56-expression50-jaw3-neck3-native-v1"


def observer_digest(observer):
    h=hashlib.sha256()
    for name, value in sorted(observer.state_dict().items()):
        v=value.detach().cpu().contiguous()
        h.update(name.encode());h.update(str(v.dtype).encode());h.update(str(tuple(v.shape)).encode())
        h.update(v.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def affect_coordinate_digest(observer):
    return observer_digest(nn.ModuleDict({name:getattr(observer,name) for name in
        ('affect_head','emotion_head','vad_head','intensity_head')}))


class FrozenFlameAffect(nn.Module):
    def __init__(self, observer, config=None):
        super().__init__()
        self.observer=copy.deepcopy(observer)
        self.config=copy.deepcopy(config or {})
        self.schema=self.config.get("motion_schema", FLAME_SCHEMA)
        if self.schema != FLAME_SCHEMA or self.observer.config.flame_dim!=56:
            raise ValueError("Visual teacher requires the declared native FLAME coefficient convention")
        self.domains=tuple(self.config.get("domains", [2]))
        self.validation=copy.deepcopy(self.config.get("validation", {}))
        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        # The trainer may call model.train(); this branch stays frozen/eval.
        super().train(False)
        self.observer.eval()
        return self

    def require_validated(self, heads=("affect",)):
        r=self.validation
        signature=(tuple(sorted(heads)),tuple(p._version for p in self.observer.parameters()),json.dumps(r,sort_keys=True))
        if getattr(self,'_validated_signature',None)==signature:
            return
        if (r.get("status")!="passed" or r.get("independent_human_rows", 0)<1
                or r.get("calibration_split")!="train" or r.get("validation_split")!="val"
                or not r.get("source_disjoint") or not r.get("input_gradient_finite_nonzero")
                or r.get("motion_schema")!=self.schema
                or set(heads)-set(r.get("validated_heads", []))
                or r.get("teacher_sha256")!=observer_digest(self.observer)):
            raise ValueError("Visual teacher has no matching independent validation; keep visual losses disabled")
        self._validated_signature=signature

    def require_coordinate(self,observer):
        expected=self.validation.get('affect_coordinate_sha256')
        signature=(id(observer),expected,tuple(p._version for name in ('affect_head','emotion_head','vad_head','intensity_head')
                                             for p in getattr(observer,name).parameters()))
        if getattr(self,'_coordinate_signature',None)==signature:return
        if not expected or expected!=affect_coordinate_digest(observer):
            raise ValueError('Visual teacher and online observer have different affect coordinates/label heads')
        self._coordinate_signature=signature

    def forward(self, motion, frame_mask, domain_id=2, now=None, frame_times=None, motion_schema=None):
        if motion_schema is not None and motion_schema != self.schema:
            raise ValueError("FLAME extraction/normalization schema changed")
        if motion.ndim!=3 or motion.shape[-1]!=56 or frame_mask.shape!=motion.shape[:2]:
            raise ValueError("Visual affect inputs must be [B,T,56] and [B,T]")
        mask=frame_mask.bool()
        if not torch.isfinite(motion[mask]).all():
            raise FloatingPointError("Nonfinite motion in a data-valid visual window")
        batch, length=mask.shape
        domains=torch.as_tensor(domain_id,device=motion.device).long().expand(batch)
        if any(int(d) not in self.domains for d in domains):
            raise ValueError("Visual teacher was not calibrated for this domain")
        ends=torch.as_tensor(length/25 if now is None else now,device=motion.device,dtype=torch.float64).expand(batch)
        if frame_times is None:
            frame_times=ends[:,None]-torch.arange(length-1,-1,-1,device=motion.device,dtype=torch.float64)[None]/25
        if frame_times.shape!=mask.shape or (frame_times[mask]>ends[:,None].expand_as(frame_times)[mask]+1e-8).any():
            raise ValueError("Visual frame timestamps must be causal and aligned")
        cfg=self.observer.config
        rows=[empty_role_features(cfg.audio_dim,cfg.text_dim,float(ends[i]),length/25,int(domains[i])) for i in range(batch)]
        features=collate_role_features(rows,motion.device)
        # Every nonvisual mask remains false. No packet/audio/text fields are accepted.
        features.update(flame_tokens=torch.where(mask[...,None],motion,0.),flame_mask=mask,
                        flame_times=frame_times,now=ends,domain_id=domains)
        features['modality_mask'][:,1]=mask.any(1)
        features['fresh_observation'][:,1]=mask.any(1)
        encoded=self.observer.encode(features,subset="V")
        affect=encoded["observation"].aff
        return dict(affect=affect,valid=mask.any(1),**self.observer.decode_affect(affect))

    def construction(self):
        return dict(observer=self.observer.construction(), config=dict(self.config,
                    motion_schema=self.schema,domains=list(self.domains),validation=copy.deepcopy(self.validation)))

    @classmethod
    def from_construction(cls, construction):
        return cls(TokenObserver(construction["observer"]),construction["config"])
