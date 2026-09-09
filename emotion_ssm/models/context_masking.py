"""Predominantly span masking, with occasional actual modality removal.

Clean features/targets are never modified. Audio/prosody and simultaneous
AU/FLAME observations are dependency groups, not independent answer copies.
"""
from __future__ import annotations

import itertools
import torch

POLICY = "context_mixture_v1"
GROUPS = {"A": ("audio", "prosody"), "V": ("au", "flame"), "T": ("text",)}


def validate_masking(config):
    if not config:
        return
    if config.get("policy") != POLICY:
        raise ValueError("Unknown context masking policy")
    probability = float(config.get("whole_modality_probability", .1))
    lower, upper = config.get("span_ratio", [.4, .7])
    if not 0 <= probability <= 1 or not 0 < lower <= upper < 1:
        raise ValueError("Invalid modality probability or span-mask ratio")
    if int(config.get("max_span_tokens", 4)) < 1:
        raise ValueError("Mask spans must have positive length")


def sample_training_subset(config=None, generator=None):
    """Keep all modalities most often; still cover all seven nonempty subsets."""
    if not config:
        return ("A", "V", "T", "AV", "AT", "VT", "AVT")[int(torch.randint(7, (), generator=generator))]
    validate_masking(config)
    if float(torch.rand((), generator=generator)) >= config.get("whole_modality_probability", .1):
        return "AVT"
    return ("A", "V", "T", "AV", "AT", "VT")[int(torch.randint(6, (), generator=generator))]


def _visual_matches(features, au_valid, flame_valid, row):
    au_time, flame_time = features.get("au_times"), features.get("flame_times")
    if au_time is None or flame_time is None:
        if au_valid.shape != flame_valid.shape:
            raise ValueError("Joint visual masking requires aligned timestamps")
        same = torch.eye(len(au_valid), dtype=torch.bool, device=au_valid.device)
    else:
        same = torch.isclose(au_time[row, :, None], flame_time[row, None, :], rtol=0, atol=1e-6)
    return same & au_valid[:, None] & flame_valid[None, :]


def _visual_sync(features, masks):
    au, flame = masks["au"], masks["flame"]
    for row in range(len(au)):
        av, fv = features["au_mask"][row].bool(), features["flame_mask"][row].bool()
        if not av.any() or not fv.any():
            continue
        same = _visual_matches(features, av, fv, row)
        old_au, old_flame = au[row].clone(), flame[row].clone()
        au[row] |= (same & old_flame[None, :]).any(-1)
        flame[row] |= (same & old_au[:, None]).any(0)


def sample_context_mask(model, features, config, generator=None):
    validate_masking(config)
    student = dict(features)
    valid, masks = {}, {}
    for modes in GROUPS.values():
        for mode in modes:
            value, available = model._mode_features(features, mode)
            available = available & torch.isfinite(value).all(-1)
            valid[mode] = available
            student[mode+"_mask"] = available.clone()
            masks[mode] = torch.zeros_like(available)
    batch = len(valid["audio"])
    whole_rows = torch.zeros(batch, dtype=torch.bool, device=valid["audio"].device)
    rescued = torch.zeros_like(whole_rows)
    low, high = config.get("span_ratio", [.4, .7])
    for row in range(batch):
        groups = [g for g, modes in GROUPS.items() if any(bool(valid[m][row].any()) for m in modes)]
        if not groups:
            continue
        whole = float(torch.rand((), generator=generator)) < config.get("whole_modality_probability", .1)
        if whole and len(groups) > 1:
            # Remove one or more complete physical modalities, but never all.
            retained = [x for n in range(1, len(groups)) for x in itertools.combinations(groups, n)]
            keep = retained[int(torch.randint(len(retained), (), generator=generator))]
            for group in groups:
                if group not in keep:
                    for mode in GROUPS[group]:
                        student[mode+"_mask"][row].zero_()
            whole_rows[row] = True
            continue
        ratio = low+(high-low)*float(torch.rand((), generator=generator))
        for mode, available in valid.items():
            if mode == "prosody" and bool(valid["audio"][row].any()):
                continue
            eligible = available[row]
            if mode == "flame" and bool(valid["au"][row].any()) and bool(eligible.any()):
                # Sample simultaneous AU/FLAME only once; unioning two
                # independently sampled masks would inflate the hidden ratio.
                same = _visual_matches(features, valid["au"][row], eligible, row)
                eligible = eligible & ~same.any(0)
            indices = eligible.nonzero().flatten()
            if not len(indices):
                continue
            count = max(1, round(len(indices)*ratio))
            selected = torch.zeros(len(indices), dtype=torch.bool)
            # Random disjoint spans. Every iteration removes at least one
            # available position, including very short/tail sequences.
            while int(selected.sum()) < count:
                candidates = (~selected).nonzero().flatten()
                start = int(candidates[int(torch.randint(len(candidates), (), generator=generator))])
                length = int(torch.randint(1, int(config.get("max_span_tokens", 4))+1, (), generator=generator))
                for pos in range(start, min(len(indices), start+length)):
                    if not selected[pos]:
                        selected[pos] = True
                        if int(selected.sum()) == count:
                            break
            masks[mode][row, indices[selected.to(indices.device)]] = True
    masks = model.synchronize_corruption(features, masks)
    _visual_sync(student, masks)
    for row in range(batch):
        present = [g for g, modes in GROUPS.items()
                   if any(bool(student[m+"_mask"][row].any()) for m in modes)]
        visible = any(bool((student[m+"_mask"][row] & ~masks[m][row]).any()) for m in masks)
        if present and not visible:
            # A single acoustic atom plus its prosody counts as one observation.
            group = present[int(torch.randint(len(present), (), generator=generator))]
            for mode in GROUPS[group]:
                masks[mode][row].zero_()
            rescued[row] = True
    counts = torch.zeros(batch, device=whole_rows.device)
    hidden = counts.clone()
    # Visible input-token count on span rows; AU/FLAME have separate token
    # streams. Exclude duplicate prosody where audio exists.
    for mode in ("audio", "au", "flame", "text"):
        available = student[mode+"_mask"] & ~whole_rows[:, None]
        counts += available.sum(-1)
        hidden += (available & masks[mode]).sum(-1)
    available = student["prosody_mask"] & ~whole_rows[:, None] & ~student["audio_mask"].any(-1)[:, None]
    counts += available.sum(-1)
    hidden += (available & masks["prosody"]).sum(-1)
    diagnostics = {"whole_modality_fraction": whole_rows.float().mean(),
                   "visible_rescue_fraction": rescued.float().mean(),
                   "span_hidden_fraction": hidden.sum()/counts.sum().clamp_min(1)}
    return student, masks, diagnostics
