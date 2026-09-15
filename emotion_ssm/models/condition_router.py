"""Generator condition choice is independent of dyadic state formation."""
from __future__ import annotations

import copy
import torch
from torch import nn
from torch.nn import functional as F

MODES = ("full_state", "affect_only", "mean_memory", "mean", "film_off",
         "actual_semantic", "oracle_visual_pseudo")
HEAD_DIMS = dict(emotion=7, vad=3, intensity=1)


def semantic_code(outputs, heads):
    if not heads or len(set(heads)) != len(heads) or set(heads)-set(HEAD_DIMS):
        raise ValueError("Declare a nonempty ordered list of validated semantic heads")
    parts=[]
    for name in heads:
        value = (F.softmax(outputs["emotion_logits"].float(), -1) if name=="emotion" else
                 outputs[name].float() if name=="vad" else outputs[name].float()[..., None])
        parts.append(value)
    return torch.cat(parts, -1)


class ConditionRouter(nn.Module):
    def __init__(self, context_dim, layout, config=None):
        super().__init__()
        self.config = copy.deepcopy(config or {})
        self.mode = self.config.get("mode", "full_state")
        self.layout, self.context_dim = dict(layout), int(context_dim)
        if self.mode not in MODES:
            raise ValueError("Unknown condition routing mode")
        self.heads = tuple(self.config.get("semantic_heads", ()))
        semantic = self.mode in ("actual_semantic", "oracle_visual_pseudo")
        if semantic and (not self.heads or set(self.heads)-set(HEAD_DIMS) or len(set(self.heads))!=len(self.heads)):
            raise ValueError("Semantic modes require explicit validated head selection")
        self.projector = nn.Linear(sum(HEAD_DIMS[h] for h in self.heads), context_dim) if semantic else None
        mean = self.config.get("train_mean")
        self.register_buffer("train_mean", torch.zeros(context_dim) if mean is None else torch.tensor(mean, dtype=torch.float32))
        if self.train_mean.shape != (context_dim,) or not torch.isfinite(self.train_mean).all():
            raise ValueError("Condition mean has an incompatible layout")
        self.mean_provenance = copy.deepcopy(self.config.get("mean_provenance", {}))
        if mean is not None and self.mean_provenance.get("split") != "train":
            raise ValueError("Condition means must come only from the training split")

    def forward(self, context, semantic=None, override=None, diagnostic=False):
        if context.shape[-1] != self.context_dim:
            raise ValueError("Condition layout differs from the generator")
        if override is not None and self.mode != "oracle_visual_pseudo":
            raise ValueError("Target-derived overrides are prohibited in ordinary deployment")
        if self.mode in ("mean", "mean_memory") and not self.mean_provenance:
            raise ValueError("Compute a source-bound training condition mean before using this mode")
        if self.mode == "film_off":
            return context, False
        if self.mode == "full_state":
            return context, True
        if self.mode == "mean":
            return self.train_mean.to(context).expand_as(context), True
        if self.mode in ("affect_only", "mean_memory"):
            result = torch.zeros_like(context) if self.mode=="affect_only" else self.train_mean.to(context).expand_as(context).clone()
            result[..., self.layout["affect"]] = context[..., self.layout["affect"]]
            return result, True
        if self.mode == "oracle_visual_pseudo":
            if not diagnostic or override is None:
                raise ValueError("Oracle checkpoints are diagnostic-only and require an explicit target override")
            semantic = override.detach()
        elif override is not None:
            raise ValueError("Actual semantic conditions cannot read a target override")
        if semantic is None or semantic.shape[-1] != self.projector.in_features or not torch.isfinite(semantic).all():
            raise ValueError("Missing or incompatible semantic condition")
        return self.projector(semantic.to(self.projector.weight.dtype)), True

    def construction(self):
        result=copy.deepcopy(self.config)
        result.update(mode=self.mode, train_mean=self.train_mean.detach().cpu().tolist(),
                      mean_provenance=self.mean_provenance)
        # An unused zero buffer is not a fitted training mean.
        if not self.mean_provenance:
            result.pop("train_mean", None)
            result.pop('mean_provenance',None)
        return result
