"""Differentiable generation objectives with data-defined global denominators."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch.nn import functional as F

REVISION = "generation-boundary-visual-v1"
VISUAL_NAMES = ("visual_feature", "visual_class_distill", "visual_vad_distill", "visual_intensity_distill")


def loss_settings(config=None):
    values = dict(config or {})
    allowed = {"revision", "boundary_weight", "temperature", "min_visual_frames", 'vad_dimensions', *[n+"_weight" for n in VISUAL_NAMES]}
    if set(values)-allowed:
        raise ValueError("Unknown generation loss option: " + str(sorted(set(values)-allowed)))
    if values.get("revision", REVISION) != REVISION:
        raise ValueError("Unknown generation loss revision")
    out = dict(revision=REVISION, boundary_weight=0., temperature=2., min_visual_frames=12,vad_dimensions=[0,1,2],
               **{n+"_weight": 0. for n in VISUAL_NAMES})
    out.update(values)
    for key, value in out.items():
        if key.endswith("_weight") and (not math.isfinite(float(value)) or float(value)<0):
            raise ValueError("Loss weights must be finite and nonnegative")
    if not math.isfinite(float(out["temperature"])) or out["temperature"] <= 0:
        raise ValueError("Distillation temperature must be positive")
    if type(out["min_visual_frames"]) is not int or out["min_visual_frames"]<1:
        raise ValueError("min_visual_frames must be a positive integer")
    dims=out['vad_dimensions']
    if not dims or len(set(dims))!=len(dims) or any(type(d) is not int or d not in (0,1,2) for d in dims):
        raise ValueError('Explicit VAD dimensions must be selected from 0,1,2')
    return out


def required_visual_heads(settings):
    heads=[]
    for name,head in zip(VISUAL_NAMES,('affect','emotion','vad','intensity')):
        if settings[name+'_weight']>0:
            heads.extend(['vad:'+str(d) for d in settings['vad_dimensions']] if head=='vad' else [head])
    return heads


@dataclass
class BoundaryFrame:
    session: str
    roles: tuple
    end_frame: int
    prediction: object
    target: object
    valid: torch.Tensor


def packet_end_frame(packet, frames, fps=25):
    end = float(packet["time"])*fps
    if abs(end-round(end))>1e-4:
        raise ValueError("Boundary timestamps must lie on the physical frame grid")
    if frames<1:
        raise ValueError("A generation packet must contain a physical frame")
    return round(end)


def matching_boundary(packet, frames, previous):
    if previous is None:
        return None
    same = (previous.session == str(packet["session_id"]) and previous.roles == tuple(packet["roles"])
            and previous.end_frame == packet_end_frame(packet, frames)-frames
            and not packet.get("discontinuity", False))
    return previous if same else None


def boundary_frame(packet, generated, target, valid):
    return BoundaryFrame(str(packet["session_id"]), tuple(packet["roles"]),
                         packet_end_frame(packet, valid.shape[1]),
                         None if generated is None else generated[:, -1],
                         None if target is None else target[:, -1].detach(), valid[:, -1].bool())


def boundary_loss_terms(generated, target, valid, previous):
    if previous is None:
        return torch.where(valid[..., None], generated.float(), 0.).sum()*0., generated.new_zeros((), dtype=torch.float64)
    if isinstance(previous, BoundaryFrame):
        previous = (previous.prediction, previous.target, previous.valid)
    pg, pt, pv = previous
    mask = valid[:, 0].bool() & pv.bool()
    def select(value):
        return torch.where(mask[:, None], value.float(), 0.)
    error = select(generated[:, 0])-select(pg)-select(target[:, 0].detach())+select(pt.detach())
    return error.square().sum(), mask.sum().double()*generated.shape[-1]


def visual_window_mask(target, valid, settings):
    # This predicate is fixed before generation and cannot be escaped by the generator.
    finite = torch.isfinite(target).all(-1)
    return (valid & finite).sum(1) >= settings["min_visual_frames"]


def plan_objective_counts(packets, previous=None, settings=None):
    """Pure metadata simulation. Does not mutate state or perform communication."""
    settings = loss_settings(settings)
    cache = dict(previous or {})
    boundaries = windows = 0
    for packet, target, valid in packets:
        key = str(packet["session_id"]), tuple(packet["roles"])
        before = matching_boundary(packet, valid.shape[1], cache.get(key))
        if before is not None:
            boundaries += int((valid[:, 0].cpu().bool() & before.valid.cpu().bool()).sum())*target.shape[-1]
        windows += int(visual_window_mask(target, valid.bool(), settings).sum())
        if packet.get("training_session_end", False):
            cache.pop(key, None)
        else:
            cache[key] = boundary_frame(packet, None, None, valid.cpu())
    return boundaries, windows


def visual_loss_terms(predicted, reference, valid, temperature=2.,vad_dimensions=(0,1,2)):
    """Return SSE-like sums per valid window; no mean over local rank batches."""
    # Slice before nonlinear operations: a masked NaN cannot poison autograd.
    p, q = predicted["affect"][valid].float(), reference["affect"][valid].detach().float()
    zero = p.sum()*0.
    out = {name: zero for name in VISUAL_NAMES}
    if len(p):
        out["visual_feature"] = (1-F.cosine_similarity(p, q, dim=-1)).sum()
        a = predicted["emotion_logits"][valid].float()/temperature
        b = reference["emotion_logits"][valid].detach().float()/temperature
        out["visual_class_distill"] = F.kl_div(F.log_softmax(a, -1), F.softmax(b, -1), reduction="none").sum()*temperature**2
        out["visual_vad_distill"] = (predicted["vad"][valid][:,vad_dimensions].float()-reference["vad"][valid][:,vad_dimensions].detach().float()).square().mean(-1).sum()
        out["visual_intensity_distill"] = (predicted["intensity"][valid].float()-reference["intensity"][valid].detach().float()).square().sum()
    return out
