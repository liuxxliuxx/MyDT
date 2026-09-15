"""Test-only auditing of PyTorch legacy/new weight-norm key formats.

Transformers 4.44 can report legacy weight_g/weight_v as unexpected and the new
parametrizations names as missing even when PyTorch's compatibility pre-hook
has loaded them correctly. Verify the actual tensors; never accept arbitrary
missing acoustic parameters or repair an unrelated name by suffix guessing.
This helper is deliberately not used by production feature extraction: its
strict assumptions are tested on explicit fixtures, while CTC checkpoints may
legitimately omit training-only masked_spec_embed parameters during inference.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

LOADING_PROTOCOL = "strict-acoustic-state-legacy-weightnorm-audit-v1"
_ORIGINAL = ".parametrizations.weight.original"


def _legacy_candidates(model, name):
    """Only the exact module name and the architecture's declared base prefix."""
    prefix = str(getattr(model, "base_model_prefix", ""))
    output = [name]
    if prefix:
        output.append(name[len(prefix)+1:] if name.startswith(prefix + ".") else prefix + "." + name)
    return tuple(dict.fromkeys(output))


def _source_state(model, source, local_files_only, revision, candidates):
    from transformers.utils import (SAFE_WEIGHTS_NAME, SAFE_WEIGHTS_INDEX_NAME,
                                    WEIGHTS_NAME, WEIGHTS_INDEX_NAME)
    from transformers.utils.hub import cached_file
    options = {"local_files_only": local_files_only, "revision": revision,
               "_commit_hash": getattr(model.config, "_commit_hash", None),
               "_raise_exceptions_for_missing_entries": False}
    # Matches from_pretrained's default preference for safetensors over pickle.
    location = None
    for filename in (SAFE_WEIGHTS_NAME, SAFE_WEIGHTS_INDEX_NAME, WEIGHTS_NAME, WEIGHTS_INDEX_NAME):
        found = cached_file(source, filename, **options)
        if found is not None:
            location = Path(found)
            break
    if location is None:
        raise ValueError("Cannot independently verify the source of missing acoustic weights")
    if location.name.endswith(".index.json"):
        index = json.loads(location.read_text(encoding="utf-8"))
        weight_map = index["weight_map"]
        files = sorted({weight_map[key] for key in candidates if key in weight_map})
        if not files:
            raise ValueError("Checkpoint index has no exact legacy weight-norm aliases")
        paths = [cached_file(source, name, **options) for name in files]
        if any(path is None for path in paths):
            raise ValueError("A source weight-norm checkpoint shard is unavailable")
    else:
        paths = [location]
    selected = {}
    for path in paths:
        if str(path).endswith(".safetensors"):
            from safetensors import safe_open
            with safe_open(str(path), framework="pt", device="cpu") as checkpoint:
                for key in candidates.intersection(checkpoint.keys()):
                    if key in selected:
                        raise ValueError("Duplicate source weight-norm tensor across checkpoint shards")
                    selected[key] = checkpoint.get_tensor(key)
        else:
            payload = torch.load(path, map_location="cpu", weights_only=True)
            for key in candidates.intersection(payload):
                if key in selected:
                    raise ValueError("Duplicate source weight-norm tensor across checkpoint shards")
                selected[key] = payload[key]
    return selected


def verify_weight_norm_loading(model, loading_info, source_state):
    """Verify proven legacy aliases; fail on all genuinely missing weights.

    ``source_state`` is the independently read original checkpoint, not the
    state_dict already mutated by a compatibility hook. This guard never changes
    model parameters: a true mismatch fails closed instead of silently migrating
    or initializing a different feature extractor.
    """
    missing = list(loading_info.get("missing_keys", []))
    if loading_info.get("mismatched_keys") or loading_info.get("error_msgs"):
        raise ValueError("Pretrained acoustic checkpoint has shape mismatches or loading errors: " + str(loading_info))
    unsupported = [key for key in missing if not (key.endswith(_ORIGINAL + "0") or key.endswith(_ORIGINAL + "1"))]
    if unsupported:
        raise ValueError("Pretrained acoustic backbone has unsupported missing weights: " + str(unsupported))
    parameters = dict(model.named_parameters())
    plans, handled = [], set()
    for key in missing:
        if key in handled:
            continue
        root = key.rsplit(_ORIGINAL, 1)[0]
        targets = (root + _ORIGINAL + "0", root + _ORIGINAL + "1")
        if any(name not in missing or name not in parameters for name in targets):
            raise ValueError("Legacy weight-norm verification requires both exact original0/original1 parameters")
        legacy = (root + ".weight_g", root + ".weight_v")
        sources = []
        for old, new in zip(legacy, targets):
            found = [candidate for candidate in _legacy_candidates(model, old) if candidate in source_state]
            if len(found) != 1:
                raise ValueError("Missing or ambiguous exact source weight-norm tensor for " + new)
            name = found[0]
            value, target = source_state[name], parameters[new]
            if not torch.is_tensor(value) or value.shape != target.shape:
                raise ValueError("Legacy weight-norm source shape does not match " + new)
            if not torch.is_floating_point(value) or not torch.isfinite(value).all():
                raise ValueError("Legacy weight-norm source must contain finite floating tensors")
            sources.append((name, value, new, target))
        # Both magnitude and direction must originate from the same base module.
        if sources[0][0].rsplit(".weight_g", 1)[0] != sources[1][0].rsplit(".weight_v", 1)[0]:
            raise ValueError("Legacy magnitude and direction have different source module identities")
        plans.extend(sources)
        handled.update(targets)
    verified = []
    with torch.no_grad():
        for old, value, new, target in plans:
            expected = value.to(device=target.device, dtype=target.dtype)
            if not torch.equal(target, expected):
                raise ValueError("Weight-norm compatibility hook did not restore the exact source tensor: " + new)
            verified.append({"source": old, "target": new, "shape": list(value.shape), "max_abs_difference": 0.})
    return {"protocol": LOADING_PROTOCOL, "missing_after_audit": [], "verified_legacy_aliases": verified,
            "model_parameters_modified": False, "reported_missing_keys": missing,
            "reported_unexpected_keys": list(loading_info.get("unexpected_keys", []))}


def load_pretrained_audio(source, local_files_only=False, revision=None):
    from transformers import AutoModel
    model, info = AutoModel.from_pretrained(source, local_files_only=local_files_only,
                                            revision=revision, output_loading_info=True)
    missing = list(info.get("missing_keys", []))
    candidates = set()
    for key in missing:
        if key.endswith(_ORIGINAL + "0") or key.endswith(_ORIGINAL + "1"):
            old = key.rsplit(_ORIGINAL, 1)[0] + (".weight_g" if key.endswith("0") else ".weight_v")
            candidates.update(_legacy_candidates(model, old))
    checkpoint = _source_state(model, source, local_files_only, revision, candidates) if candidates else {}
    audit = verify_weight_norm_loading(model, info, checkpoint)
    model.pretrained_loading_audit = audit
    if missing:
        print(json.dumps({"acoustic_loading_audit": audit, "source": str(source)}), flush=True)
    return model
