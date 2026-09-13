"""Explicit generator fine-tuning policy and replay diagnostics."""
from __future__ import annotations

import math

STABILITY_PROTOCOL = "avatar-interleaved-group-lr-valid-mask-v1"


def speech_modules(model):
    joint = getattr(getattr(model.generator, "baseline", None), "joint_encoder", None)
    return [module for name in ("audio_encoder1", "audio_encoder2")
            if (module := getattr(joint, name, None)) is not None]


def configure_stability(model, config):
    settings = config.get("generation_stability")
    if not settings:
        return
    if settings["protocol"] != STABILITY_PROTOCOL:
        raise ValueError("Unknown generation stability protocol")
    if any(p.requires_grad for module in (model.observer, model.state_model) for p in module.parameters()):
        raise ValueError("Interleaved frozen-state protocol cannot shorten trainable dynamics TBPTT")
    for encoder in speech_modules(model):
        encoder.config.avatar_mask_policy = "valid_budget_v1"
        encoder.config.mask_time_prob = float(settings.get("mask_time_prob", .05))
        encoder.config.mask_time_length = int(settings.get("mask_time_length", 10))
        if settings.get("freeze_speech", False):
            encoder.requires_grad_(False)
    if speech_modules(model) and hasattr(model, "construction_info"):
        model.construction_info["generator_audio"] = speech_modules(model)[0].config.to_dict()


def optimizer_groups(model, config):
    train = config["train"]
    settings = config.get("generation_stability", {})
    observer = {id(p) for p in model.observer.parameters()}
    state = {id(p) for p in model.state_model.parameters()}
    speech = {id(p) for module in speech_modules(model) for p in module.parameters()} if settings else set()
    film_module = getattr(model.generator, "film", None)
    film = {id(p) for p in film_module.parameters()} if settings and film_module is not None else set()
    buckets = {name: [] for name in ("generator", "speech", "film", "observer", "state")}
    for p in model.parameters():
        if p.requires_grad:
            key = ("observer" if id(p) in observer else "state" if id(p) in state else
                   "speech" if id(p) in speech else "film" if id(p) in film else "generator")
            buckets[key].append(p)
    rates = dict(generator=train["lr"], speech=settings.get("speech_lr", train["lr"]),
                 film=settings.get("film_lr", train["lr"]), observer=train["observer_lr"], state=train["state_lr"])
    return [dict(params=params, lr=float(rates[name]), initial_lr=float(rates[name]), name=name)
            for name, params in buckets.items() if params]


def set_learning_rates(optimizer, config, completed_steps):
    settings = config.get("generation_stability", {})
    duration = int(settings.get("cosine_steps", config["train"]["max_steps"]))
    warmup = int(settings.get("warmup_steps", 0))
    floor = float(settings.get("minimum_lr_ratio", 0.))
    if not duration > warmup >= 0 or not 0 <= floor <= 1:
        raise ValueError("Invalid generator learning-rate schedule")
    if completed_steps < warmup:
        ratio = (completed_steps + 1) / warmup
    else:
        progress = min(1., max(0., (completed_steps - warmup) / (duration - warmup)))
        ratio = floor + (1 - floor) * .5 * (1 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * ratio
    return {group["name"]: group["lr"] for group in optimizer.param_groups}


def mask_diagnostics(model, reset=True):
    result = {}
    for i, encoder in enumerate(speech_modules(model)):
        masked, valid, new_masked, new_valid = getattr(encoder, "_avatar_mask_totals", [0, 0, 0, 0])
        result[str(i)] = dict(masked_fraction=masked / max(1, valid),
            current_block_masked_fraction=new_masked / max(1, new_valid), measured_positions=valid)
        if reset:
            encoder._avatar_mask_totals = [0, 0, 0, 0]
    return result


def run_fixed_probes(model, training, validation, config, context, step, output):
    import json
    from pathlib import Path
    from emotion_ssm.train.generation_v3 import evaluate
    from emotion_ssm.utils.checkpoint import capture_rng_state, restore_rng_state
    settings = config.get("generation_stability", {})
    rng = capture_rng_state()
    result = {}
    try:
        for group, spec in settings.get("diagnostic_groups", {}).items():
            data = training if spec["split"] == "train" else validation
            if spec["split"] not in ("train", "val"):
                raise ValueError("Test/OOD cannot enter training diagnostics")
            metrics, records = evaluate(model, data, context.device, context.rank, context.world_size,
                selected_names=spec["names"], max_blocks=settings.get("diagnostic_max_blocks", 24),
                representation_diagnostics=False)
            result[group] = dict(metrics=metrics, records=records)
        if context.is_main and result:
            entry = dict(step=step, protocol=STABILITY_PROTOCOL, groups=result)
            with (Path(output) / "fixed_probes.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(entry, allow_nan=False) + "\n")
            print(json.dumps(dict(fixed_probe_step=step, groups={k:v["metrics"] for k,v in result.items()})), flush=True)
    finally:
        restore_rng_state(rng)
    return result
