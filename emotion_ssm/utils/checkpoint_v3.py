"""Self-contained V3 construction and atomic checkpoint recovery.

No factory imports a training entry point. External initialization files are
consulted only for a new experiment, never after complete checkpoint loading.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import torch

from emotion_ssm.config_v3 import (FORMAT_VERSION, PROTOCOL, DYNAMICS_REVISION,
                                   FUTURE_LABEL_PROTOCOL, GENERATION_REVISION, LEARNING_REVISION,
                                   STAGED_DYNAMICS_REVISION, validate_config)
from emotion_ssm.utils.checkpoint import capture_rng_state, restore_rng_state


def read_checkpoint(path):
    payload = path if isinstance(path, dict) else torch.load(str(path), map_location="cpu", weights_only=False)
    if payload.get("format_version") != FORMAT_VERSION or payload.get("protocol") != PROTOCOL:
        raise ValueError("Expected a V3.1 format-4 checkpoint; legacy experiment weights/optimizers require their archived implementation")
    return payload


def manifest_provenance(config):
    result = {}
    roots = list(config.get("data", {}).get("token_roots", []))
    dualtalk = config.get("data", {}).get("dualtalk_tokens")
    if dualtalk and dualtalk not in roots:
        roots.append(dualtalk)
    for root in roots:
        path = Path(root) / "manifest.json"
        if not path.is_file():
            raise FileNotFoundError(f"Token manifest unavailable: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("protocol") != PROTOCOL:
            raise ValueError(f"Cannot reuse legacy token manifest: {path}")
        result[str(root)] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                             "feature_sources": data["feature_sources"], "splits": data["splits"]}
    for split,filename in config.get('data',{}).get('continuous_timelines',{}).items():
        raw=Path(filename).read_bytes()
        result['continuous:'+split]={'sha256':hashlib.sha256(raw).hexdigest()}
    return result


def save_checkpoint(path, models, config, construction, kind, step=0,
                    optimizer=None, metrics=None, run_state=None, scaler=None):
    validate_config(config)
    config,construction=copy.deepcopy(config),copy.deepcopy(construction)
    system=models.get('system')
    if system is not None:
        system=system.module if hasattr(system,'module') else system
        if getattr(system,'condition_router',None) is not None:
            route=system.condition_router.construction()
            construction['condition_routing']=route
            config['generation']['condition_routing']=route
        if getattr(system,'visual_teacher',None) is not None:
            construction['visual_teacher']=system.visual_teacher.construction()
    payload = {"format_version": FORMAT_VERSION, "protocol": PROTOCOL, "kind": kind,
               "config": copy.deepcopy(config), "construction": copy.deepcopy(construction),
               "models": {name: (model.module if hasattr(model, "module") else model).state_dict()
                          for name, model in models.items()},
               "global_step": int(step), "metrics": metrics or {}, "run_state": run_state or {},
               "optimizer": None if optimizer is None else optimizer.state_dict(),
               "scaler": None if scaler is None else scaler.state_dict(),
               "rng_state": capture_rng_state(), "provenance": manifest_provenance(config),
               "experiment": {"clock": "one_second_endpoint", "selection": config.get('generation_selection',{}).get('metric','generation_total') if "avatar" in kind else "validation_loss",
                              "target_flame_observed": config.get('generation',{}).get('condition_routing',{}).get('mode')=='oracle_visual_pseudo',
                              "deployable": config.get('generation',{}).get('condition_routing',{}).get('mode')!='oracle_visual_pseudo',
                              "teacher_future_inputs": "targets_only", "generation_losses": copy.deepcopy(config.get('generation_losses',{})),
                              "max_steps": config["train"]["max_steps"],
                              "global_new_blocks": config["train"]["global_chunks_per_step"]}}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return payload


def require_training_revision(payload):
    """Old weights remain readable; changed gradient semantics require a new run."""
    revisions = {"dynamics_v3": ("dynamics_revision", DYNAMICS_REVISION),
                 "dynamics_staged_v3": ("staged_revision", STAGED_DYNAMICS_REVISION),
                 "streaming_avatar_v3": ("generation_revision", GENERATION_REVISION),
                 "observation_v3": ("learning_revision", LEARNING_REVISION),
                 "calibration_v3": ("learning_revision", LEARNING_REVISION)}
    if payload.get("kind") == "dynamics_staged_v33":
        from emotion_ssm.train.staged_v33 import REVISION
        revisions['dynamics_staged_v33'] = ('staged_revision', REVISION)
    if payload.get("kind") == "dynamics_staged_v34":
        from emotion_ssm.train.staged_v34 import REVISION
        revisions['dynamics_staged_v34'] = ('staged_revision', REVISION)
    expected = revisions.get(payload.get("kind"))
    if expected is not None:
        key, revision = expected
        if payload.get("config", {}).get("train", {}).get(key) != revision:
            raise ValueError(f"Training revision changed ({key}={revision}); old weights may be "
                             "used as explicit initialization, but their optimizer cannot resume")
        if payload.get("kind") == "dynamics_v3" and payload["config"]["train"].get("future_label_protocol") != FUTURE_LABEL_PROTOCOL:
            raise ValueError("Future gold time protocol changed; its optimizer cannot resume")


def restore_training(payload, models, optimizer=None, scaler=None, config=None, restore_rng=True):
    payload = read_checkpoint(payload)
    if optimizer is not None:
        require_training_revision(payload)
        if config is not None:
            require_generation_objective_match(payload['config'],config)
            stored = payload["config"]["train"]
            requested = config["train"]
            if stored.get("masking") != requested.get("masking") or payload["config"].get("state") != config.get("state"):
                raise ValueError("Masking/flow construction changed; use initialization, optimizer cannot resume")
    if config is not None and payload["provenance"] != manifest_provenance(config):
        raise ValueError("Data/split/features changed; resume is prohibited, use explicit initialization")
    for name, model in models.items():
        model.load_state_dict(payload["models"][name], strict=True)
    if optimizer is not None:
        if payload.get("optimizer") is None:
            raise ValueError("This checkpoint has no optimizer; use as initialization")
        optimizer.load_state_dict(payload["optimizer"])
    if scaler is not None and payload.get("scaler") is not None:
        scaler.load_state_dict(payload["scaler"])
    if restore_rng and payload.get("rng_state"):
        restore_rng_state(payload["rng_state"])
    return payload


def require_generation_objective_match(stored, requested):
    from emotion_ssm.utils.generation_losses import loss_settings
    if loss_settings(stored.get('generation_losses')) != loss_settings(requested.get('generation_losses')):
        raise ValueError('Generation losses changed; use explicit initialization with a new optimizer')
    for key in ('condition_routing','visual_teacher'):
        def canonical(value):
            if key!='condition_routing' or value is None:return value
            value=copy.deepcopy(value);value.setdefault('mode','full_state')
            if not value.get('mean_provenance'):value.pop('mean_provenance',None)
            if 'train_mean' in value:value['train_mean']=torch.tensor(value['train_mean'],dtype=torch.float32).tolist()
            return value
        if canonical(stored.get('generation',{}).get(key))!=canonical(requested.get('generation',{}).get(key)):
            raise ValueError('Condition/teacher protocol changed; use explicit initialization')
    for key in ('generation_selection','long_history'):
        if stored.get(key)!=requested.get(key):
            raise ValueError('Selection/history sampling protocol changed; initialize a new experiment')


def load_observer(path, device="cpu", teacher=False):
    from emotion_ssm.models.token_observer import TokenObserver
    payload = read_checkpoint(path)
    observer = TokenObserver(payload["construction"]["observer"])
    key = "teacher" if teacher and "teacher" in payload["models"] else "observer"
    observer.load_state_dict(payload["models"][key], strict=True)
    return observer.to(device), payload


def build_avatar(config, device="cpu", construction=None, initialize=True):
    from emotion_ssm.models.token_observer import TokenObserver
    from emotion_ssm.models.state_core import UnifiedEmotionStateCore
    from emotion_ssm.models.conditioned_dualtalk import EmotionConditionedDualTalk
    from emotion_ssm.models.streaming_v3 import StreamingAvatarV3
    from emotion_ssm.models.features import auto_config
    validate_config(config)
    g = config["generation"]
    binding = None if construction is None else construction.get("adapter_source_binding")
    if construction is None:
        observer = TokenObserver(config["observer"])
        state_cfg = {"observation_dim": observer.config.affect_dim, **config["state"]}
        state = UnifiedEmotionStateCore(**state_cfg)
        # The original baseline checkpoint supplies every generator weight.
        # Loading two additional pretrained speech checkpoints would only waste
        # startup I/O and transient memory before those weights are overwritten.
        from transformers import AutoConfig
        audio_config = AutoConfig.from_pretrained(
            g.get("audio_model", "facebook/wav2vec2-large-960h-lv60-self"), local_files_only=True)
        features = None
    else:
        observer = TokenObserver(construction["observer"])
        state = UnifiedEmotionStateCore(**construction["state"])
        audio_config = auto_config(construction["generator_audio"])
        features = None
        if construction.get("features") is not None:
            from emotion_ssm.preprocess.tokens_v3 import LocalTokenFeatures
            saved = construction["features"]
            features = LocalTokenFeatures(saved["source"], construction=saved["backbone"])
    generator = EmotionConditionedDualTalk(blendshape_dim=g["blendshape_dim"], feature_dim=g["feature_dim"],
                        film_scale=g["film_scale"], audio_config=audio_config, context_dim=state.context_dim)
    teacher = copy.deepcopy(observer)
    if initialize:
        upstream = read_checkpoint(config["paths"]["dynamics_checkpoint"])
        binding = upstream["construction"].get("adapter_source_binding")
        if binding is not None:
            config["data"]["adapter_source_binding"] = copy.deepcopy(binding)
        if upstream["construction"]["observer"] != observer.construction() or upstream["construction"]["state"] != state.get_config():
            raise ValueError("Generation construction differs from the upstream V3 checkpoint")
        observer.load_state_dict(upstream["models"]["observer"], strict=True)
        teacher.load_state_dict(upstream["models"].get("teacher", upstream["models"]["observer"]), strict=True)
        state.load_state_dict(upstream["models"]["state"], strict=True)
        baseline = torch.load(config["paths"]["baseline"], map_location="cpu", weights_only=False)
        generator.load_baseline_state_dict(baseline)
        # Raw-input deployment includes its fixed local token extractor weights.
        source = config.get("data", {}).get("streaming_source")
        if source:
            from emotion_ssm.preprocess.tokens_v3 import LocalTokenFeatures
            features = LocalTokenFeatures(source)
            if binding is not None and features.source != binding["feature_source"]:
                raise ValueError("Streaming token extractor differs from the calibrated labelled adapter source")
    model = StreamingAvatarV3(generator, observer, state, features=features, variant=g["variant"],
                             fps=g["fps"], history_seconds=g["history_seconds"],
                             speech_rms_threshold=g["speech_rms_threshold"],condition_config=g.get('condition_routing'))
    model.teacher = teacher.requires_grad_(False).eval()
    model.construction_info = {"observer": observer.construction(), "state": state.get_config(),
                              "generator_audio": generator.baseline.joint_encoder.audio_encoder1.config.to_dict(),
                              "features": None if features is None else {"source": features.source,
                                                                         "backbone": features.construction()}}
    if binding is not None:
        model.construction_info["adapter_source_binding"] = copy.deepcopy(binding)
    visual = None if construction is None else construction.get('visual_teacher')
    if visual is not None or (g.get('visual_teacher') is not None and g['visual_teacher'].get('enabled',True)):
        from emotion_ssm.models.visual_affect_teacher import FrozenFlameAffect
        model.visual_teacher = (FrozenFlameAffect.from_construction(visual) if visual is not None else FrozenFlameAffect(observer,g['visual_teacher']))
        if visual is None and initialize and g['visual_teacher'].get('checkpoint'):
            load_visual_teacher_initialization(model,g['visual_teacher']['checkpoint'])
        model.construction_info['visual_teacher']=model.visual_teacher.construction()
    if model.condition_router is not None:
        model.construction_info['condition_routing']=model.condition_router.construction()
    from emotion_ssm.utils.generation_losses import loss_settings
    model._generation_loss_settings=loss_settings(config.get('generation_losses'))
    return model.to(device)


def load_visual_teacher_initialization(model, path):
    from emotion_ssm.models.visual_affect_teacher import FrozenFlameAffect
    artifact=torch.load(path,map_location='cpu',weights_only=False)
    if artifact.get('kind')!='validated_flame_teacher_v1':
        raise ValueError('Expected an explicitly calibrated FLAME teacher artifact')
    model.visual_teacher=FrozenFlameAffect.from_construction(artifact['construction'])
    model.visual_teacher.load_state_dict(artifact['state_dict'],strict=True)
    model.visual_teacher.require_validated(artifact['construction']['config']['validation']['validated_heads'])
    model.construction_info['visual_teacher']=model.visual_teacher.construction()


def initialize_avatar_weights(model, payload, config):
    """New objective: retain common weights; initialize only declared additions."""
    old=payload['models']['system']; allowed=('condition_router.','visual_teacher.')
    current=model.state_dict()
    common={k:v for k,v in old.items() if k in current and not k.startswith(allowed)}
    missing,unexpected=model.load_state_dict(common,strict=False)
    if unexpected or any(not key.startswith(allowed) for key in missing):
        raise ValueError('Initialization differs beyond declared condition/teacher modules')
    if any(k not in current and not k.startswith(allowed) for k in old):
        raise ValueError('Unexpected architecture change during initialization')
    previous_route=payload['config'].get('generation',{}).get('condition_routing')
    requested_route=config.get('generation',{}).get('condition_routing')
    if (getattr(model,'condition_router',None) is not None and previous_route==requested_route
            and not config.get('experiment',{}).get('reset_condition_projector',False)):
        model.condition_router.load_state_dict({k.removeprefix('condition_router.'):v for k,v in old.items()
                                               if k.startswith('condition_router.')},strict=True)
    if getattr(model,'visual_teacher',None) is not None:
        spec=config['generation'].get('visual_teacher',{})
        if spec.get('checkpoint'):
            load_visual_teacher_initialization(model,spec['checkpoint'])
            model.visual_teacher.to(next(model.parameters()).device)
        elif 'visual_teacher' not in payload.get('construction',{}):
            model.visual_teacher.observer.load_state_dict(model.observer.state_dict())
        else:
            model.visual_teacher.load_state_dict({k.removeprefix('visual_teacher.'):v for k,v in old.items() if k.startswith('visual_teacher.')},strict=True)
        model.construction_info['visual_teacher']=model.visual_teacher.construction()


def load_avatar(path, device="cpu", builder=build_avatar):
    payload = read_checkpoint(path)
    if payload["kind"] != "streaming_avatar_v3":
        raise ValueError("Expected a complete V3 avatar checkpoint")
    config = validate_config(copy.deepcopy(payload["config"]))
    model = builder(config, device=device, construction=payload["construction"], initialize=False)
    model.load_state_dict(payload["models"]["system"], strict=True)
    return model, config, payload
