"""V3.2 learning protocol. Module names remain stable; artifacts are versioned."""
from __future__ import annotations

import copy
import json
from pathlib import Path

PROTOCOL = "emotion-token-packets-v3.1"
FORMAT_VERSION = 4
DYNAMICS_REVISION = "v3.2-adaptive-dyadic-flow"
STAGED_DYNAMICS_REVISION = "v3.2.1-fixed-gain-calibration-replayed-origins"
REAL_ENDPOINT_DYNAMICS_REVISIONS = ("v3.1.3-real-endpoint-forecast", DYNAMICS_REVISION)
UNIT_LABEL_DYNAMICS_REVISIONS = ("v3.1.2-state-memory-forecast", *REAL_ENDPOINT_DYNAMICS_REVISIONS)
FUTURE_LABEL_PROTOCOL = "gold_endpoint_latest_complete_origin_v1"
LEGACY_FUTURE_LABEL_PROTOCOL = "exact_grid_packet_end_v1"
GENERATION_REVISION = "v3.2-adaptive-dyadic-flow"
LEARNING_REVISION = "v3.2-mixed-context-mask"


def default_config():
    return {
        "format_version": FORMAT_VERSION, "protocol": PROTOCOL,
        "observer": {"audio_dim": 768, "text_dim": 768, "au_dim": 35, "prosody_dim": 8,
                     "flame_dim": 56, "model_dim": 256, "affect_dim": 128,
                     "num_layers": 2, "num_heads": 4, "num_domains": 3, "dropout": .1,
                     "token_protocol": PROTOCOL},
        "state": {"flow_kind": "adaptive_dyadic_v1", "flow_rank": 8,
                  "max_cross_rate": .05, "max_feedback": 2., "max_integration_step": .5},
        "generation": {"variant": "dyadic", "fps": 25, "chunk_frames": 25,
                       "history_seconds": 3, "feature_dim": 256, "blendshape_dim": 56,
                       "film_scale": .1, "speech_rms_threshold": 1e-4},
        "train": {"seed": 6666, "device": "cuda:0", "deterministic": True,
                  "max_steps": 1000, "global_chunks_per_step": 32, "tbptt_seconds": 32,
                  "lr": 1e-4, "observer_lr": 1e-5, "state_lr": 1e-4,
                  "weight_decay": .01, "clip_grad": 5., "amp": True,
                  "validate_every": 250, "log_every": 10,
                  "validation_max_dialogues": 16, "observer_batch_size": 16,
                  "forecast_seconds": [1, 2, 4, 8, 16, 32],
                  "future_weight": 1., "coordinate_weight": .1,
                  "masked_weight": .2, "label_weight": 1.,
                  "modality_dropout": True, "observation_dropout": .15,
                  "observation_steps": 10000, "dynamics_steps": 10000,
                   "calibration_steps": 2500, "learning_revision": LEARNING_REVISION,
                   "masking": {"policy": "context_mixture_v1", "whole_modality_probability": .1,
                               "span_ratio": [.4, .7], "max_span_tokens": 4},
                   "dynamics_revision": DYNAMICS_REVISION,
                   "future_label_protocol": FUTURE_LABEL_PROTOCOL,
                   "generation_revision": GENERATION_REVISION,
                  "require_a0_gate": True},
        "data": {"token_roots": [], "dualtalk_tokens": "", "dualtalk_raw": "",
                 "source_names": ["emotiontalk", "iemocap", "dualtalk"]},
        "paths": {"output": "runs/v3", "observation_checkpoint": "",
                  "dynamics_checkpoint": "", "baseline": "model/dualtalk_baseline.pth",
                  "resume": ""},
    }


def merge_config(base, changes):
    out = copy.deepcopy(base)
    for key, value in changes.items():
        out[key] = merge_config(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else copy.deepcopy(value)
    return out


def validate_config(config):
    if config.get("format_version") != FORMAT_VERSION or config.get("protocol") != PROTOCOL:
        raise ValueError("V3.1 requires format 4 and new token semantics; legacy experiment optimizers cannot resume")
    generation = config["generation"]
    if config.get('long_history',{}).get('enabled',False):
        if not config.get('generation_stability') or generation.get('train_observer',True) or generation.get('train_state',True):
            raise ValueError('Long-history windows require the frozen upstream interleaved generation protocol')
    if config.get('generation_selection',{}).get('metric','generation_total') not in ('generation_total','generation_total_with_boundary'):
        raise ValueError('Unknown generation selection metric')
    from emotion_ssm.utils.generation_losses import loss_settings
    loss_settings(config.get('generation_losses'))
    if generation.get('condition_routing') is not None:
        from emotion_ssm.models.condition_router import MODES
        if generation['condition_routing'].get('mode','full_state') not in MODES:
            raise ValueError('Unknown condition routing mode')
    if generation["fps"] != 25 or generation["chunk_frames"] != 25:
        raise ValueError("All V3 controls use 25 new frames per second")
    if not 0 <= generation["history_seconds"] <= 3:
        raise ValueError("Generator history must lie in [0,3] seconds")
    if generation["variant"] not in ("none", "affect", "self", "dyadic"):
        raise ValueError("Unknown conditioning control")
    train = config["train"]
    from emotion_ssm.models.context_masking import validate_masking
    validate_masking(train.get("masking"))
    if train["global_chunks_per_step"] < 1 or not 0 < train["tbptt_seconds"] <= 32:
        raise ValueError("Invalid optimization/TBPTT budget")
    horizons = train["forecast_seconds"]
    if not horizons or sorted(set(horizons)) != horizons or min(horizons) <= 0:
        raise ValueError("Future query times must be positive and strictly increasing")
    future_protocol = train.get("future_label_protocol", LEGACY_FUTURE_LABEL_PROTOCOL)
    if future_protocol not in (FUTURE_LABEL_PROTOCOL, LEGACY_FUTURE_LABEL_PROTOCOL):
        raise ValueError("Unknown future gold label time protocol")
    if train.get("dynamics_revision") in REAL_ENDPOINT_DYNAMICS_REVISIONS and future_protocol != FUTURE_LABEL_PROTOCOL:
        raise ValueError("New dynamics revision requires real-endpoint future supervision")
    return config


def read_config(path):
    # The stored config is authoritative on resume; defaults fill only a new config.
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    # Reading a saved legacy config must not silently assign a new loss protocol.
    stored_train = value.get("train", {})
    legacy = "dynamics_revision" in stored_train and stored_train["dynamics_revision"] != DYNAMICS_REVISION
    if legacy:
        stored_train.setdefault("future_label_protocol", FUTURE_LABEL_PROTOCOL
            if stored_train["dynamics_revision"] in REAL_ENDPOINT_DYNAMICS_REVISIONS else LEGACY_FUTURE_LABEL_PROTOCOL)
        stored_train.setdefault("masking", None)
        stored_train.setdefault("learning_revision", "v3.1-legacy-context-mask")
        stored_train.setdefault("generation_revision", "v3.1-legacy-generation")
    defaults = default_config()
    if legacy:
        # Old complete configs must not acquire new flow parameters on read.
        defaults["state"] = {}
    return validate_config(merge_config(defaults, value))


def write_config(path, config):
    validate_config(config)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
