"""Restore complete A0/Phase-A/Phase-B bundles using embedded configuration."""
import torch

from emotion_ssm.models import ObservationEncoder, ObservationSupervisionHeads, AffectDecoder, DyadicEmotionSSM
from emotion_ssm.train.common import ObservationTrainingBundle
from emotion_ssm.train.dynamics_core import DynamicsTrainingBundle
from emotion_ssm.utils.generation_checkpoint import read_checkpoint, checkpoint_config


def load_emotion(path, device="cpu"):
    payload = read_checkpoint(path)
    if payload.get("format_version") != 2 or payload.get("data_semantics") != "masked-vad-endpoint-v2":
        raise ValueError("Legacy emotion weights are initialization-only; this loader requires a complete v2 checkpoint")
    cfg = checkpoint_config(payload)
    weights = payload["models"]["bundle"]
    encoder = ObservationEncoder.from_config(cfg)
    if "state_model.personal.baseline_delta.weight" in weights:
        speakers = weights["state_model.personal.baseline_delta.weight"].shape[0]
        model = DynamicsTrainingBundle(encoder, DyadicEmotionSSM.from_config(cfg, speakers),
                                       AffectDecoder(cfg.MODEL.OBSERVATION_DIM), cfg)
    else:
        speakers = weights["heads.speaker.2.weight"].shape[0]
        model = ObservationTrainingBundle(encoder, ObservationSupervisionHeads(cfg.MODEL.OBSERVATION_DIM,
                                                                               speakers, cfg.MODEL.NUM_DOMAINS))
    model.load_state_dict(weights, strict=True)
    teacher = ObservationEncoder.from_config(cfg)
    teacher.load_state_dict(payload["models"]["teacher"], strict=True)
    teacher.requires_grad_(False).eval()
    return model.to(device), teacher.to(device), cfg, payload


def deployment_paths(saved, requested):
    cfg = saved.clone(); cfg.defrost()
    allowed = {"DATA.ROOT", "DATA.EMOTIONTALK_ROOT", "DATA.IEMOCAP_FEATURE_ROOT",
               "DATA.IEMOCAP_RAW_ROOT", "DATA.DUALTALK_ROOT", "TRAIN.OUTPUT_ROOT",
               "DUALTALK.TIMED_FEATURE_ROOT", "DUALTALK.SPLIT_MANIFEST"}
    for path in requested.TRAIN.RESUME_PATH_OVERRIDES:
        if path not in allowed:
            raise ValueError(f"Resume cannot override model or experiment settings: {path}")
        section, key = path.split(".")
        cfg[section][key] = requested[section][key]
    cfg.TRAIN.RESUME = requested.TRAIN.RESUME
    cfg.TRAIN.RESUME_PATH_OVERRIDES = list(requested.TRAIN.RESUME_PATH_OVERRIDES)
    cfg.DEVICE = requested.DEVICE
    cfg.freeze()
    return cfg
