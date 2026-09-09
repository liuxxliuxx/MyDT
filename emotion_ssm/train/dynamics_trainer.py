from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import torch
from torch.cuda.amp import GradScaler, autocast

from emotion_ssm.data import DialogueWindowDataset, build_speaker_vocabulary
from emotion_ssm.models import ObservationEncoder
from emotion_ssm.train.common import (
    append_metrics,
    build_feature_stores,
    create_run_directory,
    make_loader,
    maybe_ddp,
    move_to_device,
)
from emotion_ssm.train.dynamics_core import (
    initialize_dynamics_bundle,
    load_component_state,
    make_teacher_aff,
    swap_dialogue_roles,
)
from emotion_ssm.utils.checkpoint import (
    load_training_checkpoint,
    save_training_checkpoint,
)
from emotion_ssm.utils.distributed import init_distributed, unwrap_model
from emotion_ssm.data.full_dialogues import FullDialogueDataset, collate_full_dialogues
from emotion_ssm.utils.statistics import WeightedStatistics


def validation_batches(loader, context, pool_size=0, max_batches=0):
    """Fixed CF groups are independent of minibatch size and DDP sharding."""
    if pool_size:
        groups = [list(range(start, min(start+pool_size, len(loader.dataset))))
                  for start in range(0, len(loader.dataset), pool_size)]
        if max_batches:
            groups = groups[:max_batches]
        for group in groups[context.rank::context.world_size]:
            yield loader.collate_fn([loader.dataset[index] for index in group])
    else:
        for index, batch in enumerate(loader):
            if max_batches and index >= max_batches:
                break
            yield batch


def _load_teacher(cfg, device: torch.device) -> ObservationEncoder:
    path = Path(cfg.TRAIN.EMA_TEACHER_CHECKPOINT)
    if not path.is_file():
        raise FileNotFoundError(
            "TRAIN.EMA_TEACHER_CHECKPOINT must point to ema_teacher.pt or an A0 checkpoint"
        )
    teacher = ObservationEncoder.from_config(cfg)
    teacher.load_state_dict(load_component_state(path, "teacher"), strict=True)
    teacher.requires_grad_(False)
    teacher.eval()
    return teacher.to(device)


def _load_previous_stage(bundle, path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    bundle.state_model.load_state_dict(
        load_component_state(path, "state_model"), strict=True
    )
    bundle.decoder.load_state_dict(load_component_state(path, "decoder"), strict=True)


@torch.no_grad()
def validate(
    bundle,
    teacher,
    loader,
    class_weights,
    context,
    enable_partner: bool,
    use_counterfactual: bool,
    max_batches: int = 0,
) -> Dict[str, float]:
    bundle.eval()
    totals = WeightedStatistics()
    from emotion_ssm.utils.prediction_statistics import PredictionStatistics
    prediction_totals = PredictionStatistics()
    from emotion_ssm.models.observation import SUBSET_MASKS, SUBSET_NAMES
    raw = unwrap_model(bundle)
    pool_size = getattr(raw, "cf_eval_pool_dialogues", 4) if use_counterfactual else 0
    batches = 0
    for raw_batch in validation_batches(loader, context, pool_size, max_batches):
        batch = move_to_device(raw_batch, context.device)
        target_aff = make_teacher_aff(teacher, batch)
        raw = unwrap_model(bundle)
        for subset, name in zip(SUBSET_MASKS, SUBSET_NAMES):
            losses = raw(batch, target_aff, class_weights, enable_partner,
                         use_counterfactual, subset_mask=subset[None])
            totals.dynamics(losses, name + "_")
            prediction_totals.merge(raw.prediction_statistics, name + "_")
            if name == "AVT":
                totals.dynamics(losses)
                prediction_totals.merge(raw.prediction_statistics)
                if enable_partner:
                    self_only = raw(batch, target_aff, class_weights, False, False, subset_mask=subset[None])
                    totals.add("partner_gain", self_only.get("h1", self_only["total"]) - losses.get("h1", losses["total"]),
                               losses.get("h1_count", 0))
        batches += 1
    aggregated = totals.finalize()
    aggregated.update(prediction_totals.metrics())
    # Recompute selection losses from global numerators/denominators. CCC is
    # nonlinear and must never be averaged across validation minibatches.
    raw = unwrap_model(bundle)
    for prefix in [""] + [name+"_" for name in SUBSET_NAMES]:
        horizon_losses = []
        modes = ("conditional", "open_loop") if raw.rollout_mode == "joint" else (raw.rollout_mode,)
        for horizon in raw.horizons:
            terms = {}
            for mode in modes:
                key = f"{prefix}{mode}_h{horizon}"
                terms[mode] = (aggregated.get(key+"_affect_loss", 0.) + raw.loss_emotion*aggregated.get(key+"_emotion_ce", 0.)
                    + raw.loss_intensity*aggregated.get(key+"_intensity_mse", 0.) + raw.loss_vad*aggregated.get(key+"_vad_loss", 0.))
                aggregated[key] = terms[mode]
            total = (terms["conditional"] + raw.open_loop_weight*terms["open_loop"]
                     if raw.rollout_mode == "joint" else terms[raw.rollout_mode])
            aggregated[f"{prefix}h{horizon}"] = total
            horizon_losses.append(total)
        first = horizon_losses[0] if horizon_losses else 0.
        rest = sum(horizon_losses[1:])/max(len(horizon_losses)-1, 1)
        aggregated[prefix+"total"] = (raw.loss_next*first + raw.loss_trajectory*rest
            + raw.loss_correction*aggregated.get(prefix+"correction", 0.)
            + raw.loss_counterfactual*aggregated.get(prefix+"counterfactual", 0.)
            + raw.loss_observation_anchor*aggregated.get(prefix+"observation_anchor", 0.))
    result = {f"val_{name}": value for name, value in aggregated.items()}
    if use_counterfactual:
        result["val_cf_pool_dialogues"] = pool_size
    bundle.train()
    return result


def run_dynamics_stage(
    cfg,
    stage_name: str,
    required_checkpoint_name: str,
    enable_partner: bool,
    use_counterfactual: bool,
    previous_stage_checkpoint: str = "",
) -> None:
    resumed = None
    if cfg.TRAIN.RESUME:
        from emotion_ssm.utils.emotion_checkpoint import load_emotion, deployment_paths
        resumed = load_emotion(cfg.TRAIN.RESUME)
        cfg = deployment_paths(resumed[2], cfg)
        from emotion_ssm.utils.checkpoint import validate_resume_data
        validate_resume_data(resumed[3], cfg)
    context = init_distributed(cfg.DEVICE, cfg.SEED, cfg.DETERMINISTIC)
    run_dir = create_run_directory(cfg, stage_name)
    if context.is_main:
        (run_dir / "config.yaml").write_text(cfg.dump(), encoding="utf-8")

    stores = build_feature_stores(cfg)
    speaker_vocab = build_speaker_vocabulary(stores)
    dataset_type = FullDialogueDataset if cfg.DYNAMICS.FULL_DIALOGUES else DialogueWindowDataset
    collate = collate_full_dialogues if cfg.DYNAMICS.FULL_DIALOGUES else None
    train_dataset = dataset_type(
        stores,
        "train",
        speaker_vocab,
        cfg.DATA.WINDOW_LENGTH,
        cfg.DATA.WINDOW_STRIDE,
    )
    val_dataset = dataset_type(
        stores,
        "val",
        speaker_vocab,
        cfg.DATA.WINDOW_LENGTH,
        cfg.DATA.WINDOW_STRIDE,
    )
    train_loader, train_sampler = make_loader(
        train_dataset,
        cfg.TRAIN.SEQUENCE_BATCH_SIZE,
        context,
        True,
        cfg.DATA.NUM_WORKERS,
        cfg.DATA.PIN_MEMORY,
        collate_fn=collate,
    )
    val_loader, _ = make_loader(
        val_dataset,
        cfg.TRAIN.SEQUENCE_BATCH_SIZE,
        context,
        False,
        cfg.DATA.NUM_WORKERS,
        cfg.DATA.PIN_MEMORY,
        collate_fn=collate,
    )
    bundle = (resumed[0] if resumed else initialize_dynamics_bundle(cfg, len(speaker_vocab))).to(context.device)
    if previous_stage_checkpoint and not resumed:
        _load_previous_stage(bundle, Path(previous_stage_checkpoint))
    teacher = resumed[1].to(context.device) if resumed else _load_teacher(cfg, context.device)
    # Register only trainable affect/event branches in DDP from the outset.
    if enable_partner:
        bundle.set_phase_b_affect_frozen(True)
    bundle = maybe_ddp(bundle, context)
    raw_bundle = unwrap_model(bundle)
    parameter_groups = [{
        "params": list(raw_bundle.state_model.parameters())
        + list(raw_bundle.decoder.parameters()),
        "lr": cfg.TRAIN.LR,
    }]
    if enable_partner:
        # Phase B trains event/action/fusion at the main learning rate.  The
        # shared affect branch starts frozen and is included here so that it
        # can be released without rebuilding optimizer state.
        raw_bundle.set_phase_b_affect_frozen(True)
        action_event_parameters, affect_parameters = raw_bundle.phase_b_parameter_groups()
        parameter_groups.extend(
            [
                {"params": action_event_parameters, "lr": cfg.TRAIN.LR},
                {
                    "params": affect_parameters,
                    "lr": cfg.TRAIN.LR * cfg.DYNAMICS.AFFECT_LR_SCALE,
                },
            ]
        )
    else:
        observation_parameters = [
            value for value in raw_bundle.encoder.parameters() if value.requires_grad
        ]
        if observation_parameters:
            parameter_groups.append(
                {
                    "params": observation_parameters,
                    "lr": cfg.TRAIN.LR * cfg.TRAIN.STATE_LR_SCALE,
                }
            )
    optimizer = torch.optim.AdamW(
        parameter_groups, weight_decay=cfg.TRAIN.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(cfg.TRAIN.EPOCHS, 1)
    )
    scaler = GradScaler(enabled=cfg.TRAIN.AMP and context.device.type == "cuda")
    class_weights = train_dataset.class_weights().to(context.device)
    start_epoch = 0
    global_step = 0
    best_score = float("inf")
    if cfg.TRAIN.RESUME:
        checkpoint = load_training_checkpoint(
            Path(cfg.TRAIN.RESUME),
            {"bundle": bundle, "teacher": teacher},
            optimizer,
            scheduler,
            scaler,
        )
        start_epoch = int(checkpoint["epoch"])
        global_step = int(checkpoint["global_step"])
        best_score = float(checkpoint.get("metrics", {}).get("val_total", best_score))

    for epoch in range(start_epoch, cfg.TRAIN.EPOCHS):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if enable_partner:
            raw_bundle.set_phase_b_affect_frozen(
                epoch < cfg.DYNAMICS.FREEZE_AFFECT_EPOCHS
            )
        bundle.train()
        totals = WeightedStatistics()
        batches = 0
        optimizer.zero_grad(set_to_none=True)
        for batch_index, raw_batch in enumerate(train_loader):
            if cfg.TRAIN.DRY_RUN and batch_index >= cfg.TRAIN.DRY_RUN_TRAIN_BATCHES:
                break
            batch = move_to_device(raw_batch, context.device)
            if enable_partner and cfg.DYNAMICS.RANDOM_ROLE_SWAP:
                batch = swap_dialogue_roles(batch)
            target_aff = make_teacher_aff(teacher, batch)
            with autocast(enabled=cfg.TRAIN.AMP and context.device.type == "cuda"):
                losses = bundle(
                    batch,
                    target_aff,
                    class_weights,
                    enable_partner,
                    use_counterfactual,
                )
                loss = losses["total"] / cfg.TRAIN.GRAD_ACCUMULATION
            scaler.scale(loss).backward()
            should_step = (
                (batch_index + 1) % cfg.TRAIN.GRAD_ACCUMULATION == 0
                or batch_index + 1 == len(train_loader)
                or (
                    cfg.TRAIN.DRY_RUN
                    and batch_index + 1 == cfg.TRAIN.DRY_RUN_TRAIN_BATCHES
                )
            )
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(bundle.parameters(), cfg.TRAIN.GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
            totals.dynamics(losses)
            batches += 1
        scheduler.step()
        totals = totals.finalize()
        val_metrics = validate(
            bundle,
            teacher,
            val_loader,
            class_weights,
            context,
            enable_partner,
            use_counterfactual,
            cfg.TRAIN.DRY_RUN_VAL_BATCHES if cfg.TRAIN.DRY_RUN else 0,
        )
        metrics = {"epoch": epoch + 1, **totals, **val_metrics}
        if context.is_main:
            append_metrics(run_dir / "metrics.jsonl", metrics)
            print(json.dumps(metrics, ensure_ascii=False))
            models = {"bundle": bundle, "teacher": teacher}
            save_training_checkpoint(
                run_dir / "last.pt",
                epoch + 1,
                global_step,
                models,
                optimizer,
                scheduler,
                scaler,
                metrics,
                config=cfg.dump(),
            )
            score = val_metrics["val_total"]
            if score < best_score:
                best_score = score
                best_path = run_dir / required_checkpoint_name
                save_training_checkpoint(
                    best_path,
                    epoch + 1,
                    global_step,
                    models,
                    optimizer,
                    scheduler,
                    scaler,
                    metrics,
                    config=cfg.dump(),
                )
                # Also keep the conventional name used by experiment scripts.
                save_training_checkpoint(
                    run_dir / "best.pt",
                    epoch + 1,
                    global_step,
                    models,
                    metrics=metrics,
                    config=cfg.dump(),
                )
        context.barrier()
        if cfg.TRAIN.DRY_RUN:
            break
