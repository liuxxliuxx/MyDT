"""Versioned, self-contained construction for generation and calibration."""
from __future__ import annotations

import json
from pathlib import Path

import torch
from yacs.config import CfgNode

from emotion_ssm.config import get_cfg_defaults
from emotion_ssm.data.protocol import PROTOCOL_VERSION, fingerprint, select_adapter_source, require_compatible
from emotion_ssm.models.conditioned_dualtalk import EmotionConditionedDualTalk, BlendshapeAffectProjector
from emotion_ssm.models.dynamics import DyadicEmotionSSM
from emotion_ssm.models.features import FrozenFeatures, auto_config
from emotion_ssm.models.multimodal import MultimodalObserver
from emotion_ssm.models.observation import ObservationEncoder
from emotion_ssm.models.streaming import StreamingAvatar


def read_checkpoint(path):
    return torch.load(str(path), map_location="cpu", weights_only=False)


def checkpoint_config(payload):
    if not payload.get("config"):
        raise ValueError("Checkpoint has no configuration; use explicit legacy initialization")
    cfg = get_cfg_defaults()
    value = payload["config"]
    cfg.merge_from_other_cfg(CfgNode.load_cfg(value) if isinstance(value, str) else CfgNode(value))
    cfg.freeze()
    return cfg


def source_for_config(cfg):
    path = Path(cfg.DUALTALK.SOURCE_MANIFEST)
    if not path.is_file():
        raise ValueError("DUALTALK.SOURCE_MANIFEST must declare trained domains and exact feature sources")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    domain = select_adapter_source(cfg.DATA.SOURCES, cfg.DUALTALK.ADAPTER_INIT_SOURCE)
    name = "emotiontalk" if domain == 0 else "iemocap"
    source = manifest["domains"][name]
    if not source.get("trained", False):
        raise ValueError(f"Feature source {name} is not declared trained")
    source = {k: v for k, v in source.items() if k != "trained"}
    require_compatible(source, source)
    return domain, source


def build_streaming(cfg, device="cpu", construction=None, initialize=True):
    from emotion_ssm.train.dynamics_core import load_component_state
    from emotion_ssm.train.dualtalk import _load_global_dynamics, _load_raw
    if construction is None:
        domain, source = source_for_config(cfg)
        features = FrozenFeatures(source, local_files_only=cfg.DUALTALK.LOCAL_FILES_ONLY,
                                  max_tokens=cfg.DUALTALK.TEXT_MAX_TOKENS)
        audio_config = None
    else:
        domain, source = construction["adapter_source"], construction["feature_source"]
        features = FrozenFeatures(source, construction["features"], max_tokens=cfg.DUALTALK.TEXT_MAX_TOKENS)
        audio_config = auto_config(construction["generator_audio"])
    generator = EmotionConditionedDualTalk.from_config(cfg, audio_config, cfg.MODEL.OBSERVATION_DIM)
    encoder = ObservationEncoder.from_config(cfg)
    dynamics = DyadicEmotionSSM.from_config(cfg, num_speakers=1)
    observer = MultimodalObserver(encoder, cfg.MODEL.MODEL_DIM, cfg.MODEL.AU_NUM_HEADS,
                                  cfg.MODEL.AU_NUM_LAYERS, cfg.MODEL.DROPOUT, cfg.DUALTALK.ADAPTER_DOMAIN_ID)
    projector = None
    if cfg.LOSS.GENERATION_STATE:
        projector = BlendshapeAffectProjector(56, cfg.MODEL.OBSERVATION_DIM)
    if initialize:
        generator.load_baseline_state_dict(_load_raw(Path(cfg.DUALTALK.BASELINE_CHECKPOINT)))
        phase_b = Path(cfg.DUALTALK.PHASE_B_CHECKPOINT or cfg.TRAIN.PHASE_B_CHECKPOINT)
        phase_payload = read_checkpoint(phase_b)
        phase_config = checkpoint_config(phase_payload)
        selected_name = "emotiontalk" if domain == 0 else "iemocap"
        if selected_name not in phase_config.DATA.SOURCES:
            raise ValueError("Selected adapter domain was not trained in this Phase-B checkpoint")
        if selected_name not in phase_payload.get("feature_sources", {}):
            raise ValueError("Phase-B provenance is missing; rebuild with the v2 data protocol")
        require_compatible(source, phase_payload["feature_sources"][selected_name])
        encoder.load_state_dict(load_component_state(phase_b, "encoder"), strict=True)
        encoder.copy_domain_adapters(domain, cfg.DUALTALK.ADAPTER_DOMAIN_ID)
        _load_global_dynamics(dynamics, phase_b)
        if not cfg.DUALTALK.CALIBRATION_CHECKPOINT:
            raise ValueError("Run adapter calibration before generation training")
        calibration = read_checkpoint(cfg.DUALTALK.CALIBRATION_CHECKPOINT)
        if not calibration.get("calibrated", False):
            raise ValueError("FLAME calibration failed validation; inspect calibration metrics before training")
        require_compatible(source, calibration["feature_source"])
        if calibration.get("coordinate_id") != state_dict_digest(encoder.state_dict(), exclude_domain=2):
            raise ValueError("Calibration uses a different affect coordinate system")
        observer.load_state_dict(calibration["observer"], strict=True)
        if projector is not None:
            projection = read_checkpoint(cfg.DUALTALK.PROJECTOR_CHECKPOINT)
            require_compatible(source, projection["feature_source"])
            if projection.get("coordinate_id") != calibration["coordinate_id"] or not projection.get("calibrated"):
                raise ValueError("State loss requires a calibrated projector in the same coordinates")
            projector.load_state_dict(projection["projector"], strict=True)
    model = StreamingAvatar(generator, observer, dynamics, features, cfg.DUALTALK.VARIANT,
                            cfg.DUALTALK.FPS, cfg.DUALTALK.HISTORY_SECONDS, projector,
                            cfg.LOSS.GENERATION_STATE, cfg.DUALTALK.SPEECH_RMS_THRESHOLD).to(device)
    model.construction_info = {"adapter_source": domain, "feature_source": source,
                               "features": features.construction(),
                               "generator_audio": generator.baseline.joint_encoder.audio_encoder1.config.to_dict()}
    return model


def state_dict_digest(state, exclude_domain=None):
    import hashlib
    result = hashlib.sha256()
    for name, value in sorted(state.items()):
        if exclude_domain is not None and f"adapters.{exclude_domain}." in name:
            continue
        result.update(name.encode())
        result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def save_generation(path, model, cfg, step=0, optimizer=None, metrics=None, run_state=None):
    from emotion_ssm.utils.checkpoint import capture_rng_state
    payload = {"format_version": PROTOCOL_VERSION, "kind": "streaming_avatar",
               "config": cfg.dump(), "construction": model.construction_info,
               "models": {"system": model.state_dict()}, "global_step": step,
               "metrics": metrics or {}, "run_state": run_state or {},
               "optimizer": None if optimizer is None else optimizer.state_dict(),
               "rng_state": capture_rng_state(),
               "protocol": {"chunk_frames": cfg.DUALTALK.CHUNK_FRAMES,
                            "history_seconds": cfg.DUALTALK.HISTORY_SECONDS,
                            "clock": "observation_endpoint", "selection": "generation_total",
                            "max_steps": cfg.TRAIN.MAX_STEPS,
                            "global_chunks_per_step": cfg.TRAIN.GLOBAL_CHUNKS_PER_STEP}}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_generation(path, device="cpu", builder=build_streaming):
    payload = path if isinstance(path, dict) else read_checkpoint(path)
    if payload.get("format_version") != PROTOCOL_VERSION or payload.get("kind") != "streaming_avatar":
        raise ValueError("Not a v2 streaming checkpoint; use explicit legacy evaluation/initialization")
    cfg = checkpoint_config(payload)
    model = builder(cfg, device=device, construction=payload["construction"], initialize=False)
    model.load_state_dict(payload["models"]["system"], strict=True)
    return model, cfg, payload
