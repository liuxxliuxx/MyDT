from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Mapping

import torch
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast

from emotion_ssm.config import parse_config_args
from emotion_ssm.data import (
    UnifiedUtteranceDataset,
    build_speaker_vocabulary,
)
from emotion_ssm.losses import observation_losses
from emotion_ssm.metrics import classification_metrics, confusion_matrix
from emotion_ssm.models import (
    SUBSET_MASKS,
    SUBSET_NAMES,
    ObservationEncoder,
    ObservationSupervisionHeads,
    build_ema_teacher,
    update_ema,
)
from emotion_ssm.train.common import (
    ObservationTrainingBundle,
    append_metrics,
    build_feature_stores,
    create_run_directory,
    grl_alpha,
    make_loader,
    maybe_ddp,
    move_to_device,
)
from emotion_ssm.utils.checkpoint import (
    load_training_checkpoint,
    save_training_checkpoint,
)
from emotion_ssm.utils.distributed import init_distributed, unwrap_model


def _mask_feature_spans(value: torch.Tensor, ratio: float) -> torch.Tensor:
    """Mask one contiguous feature span per sample in an offline embedding."""
    if ratio <= 0.0:
        return value
    width = value.shape[-1]
    span = min(max(int(round(width * ratio)), 1), width)
    output = value.clone()
    starts = torch.randint(0, width - span + 1, (len(value),), device=value.device)
    for row, start in enumerate(starts.tolist()):
        output[row, start : start + span] = 0
    return output


def augment_student_batch(batch: Mapping[str, torch.Tensor], cfg) -> Dict[str, torch.Tensor]:
    """Create a noisy student view while leaving the EMA teacher batch clean."""
    output = dict(batch)
    options = cfg.TRAIN.AUGMENT
    if not options.ENABLED:
        return output

    audio = batch["audio"]
    output["audio"] = _mask_feature_spans(audio, options.AUDIO_MASK_RATIO)
    if options.AUDIO_NOISE_STD > 0:
        output["audio"] = output["audio"] + torch.randn_like(audio) * float(
            options.AUDIO_NOISE_STD
        )

    text = batch["text"]
    if options.TEXT_DROPOUT > 0:
        keep = torch.rand_like(text) >= float(options.TEXT_DROPOUT)
        output["text"] = text * keep.to(text.dtype)

    if "face_frame_mask" in batch and batch["face"].ndim == 3:
        frame_mask = batch["face_frame_mask"].clone().bool()
        ratio = float(options.FACE_MASK_RATIO)
        if ratio > 0:
            for row in range(len(frame_mask)):
                valid_indices = frame_mask[row].nonzero(as_tuple=False).flatten()
                if len(valid_indices) <= 1:
                    continue
                span = min(
                    max(int(round(len(valid_indices) * ratio)), 1),
                    len(valid_indices) - 1,
                )
                start = int(torch.randint(0, len(valid_indices) - span + 1, (), device=frame_mask.device))
                frame_mask[row, valid_indices[start : start + span]] = False
        output["face_frame_mask"] = frame_mask

    actual = batch["modality_mask"].clone().bool()
    drop_probability = float(options.MODALITY_DROPOUT)
    if drop_probability > 0:
        dropped = (torch.rand_like(actual.float()) < drop_probability) & actual
        augmented_mask = actual & ~dropped
        empty = ~augmented_mask.any(dim=-1)
        for row in empty.nonzero(as_tuple=False).flatten().tolist():
            available = actual[row].nonzero(as_tuple=False).flatten()
            if len(available):
                choice = available[torch.randint(len(available), (), device=actual.device)]
                augmented_mask[row, choice] = True
        output["modality_mask"] = augmented_mask

    if "reliability" in batch and options.RELIABILITY_JITTER > 0:
        jitter = (torch.rand_like(batch["reliability"]) * 2.0 - 1.0) * float(
            options.RELIABILITY_JITTER
        )
        output["reliability"] = (batch["reliability"] + jitter).clamp(0.0, 1.0)
    return output


@torch.no_grad()
def validate(bundle, loader, device, context, max_batches: int = 0) -> Dict[str, float]:
    bundle.eval()
    from emotion_ssm.utils.statistics import RepresentationTotals
    representations = RepresentationTotals()
    from emotion_ssm.utils.prediction_statistics import PredictionStatistics
    domain_metrics = PredictionStatistics()
    matrices = [torch.zeros(7, 7, dtype=torch.long) for _ in SUBSET_NAMES]
    intensity_sum = torch.zeros(len(SUBSET_NAMES))
    counts = torch.zeros(len(SUBSET_NAMES))
    for batch_index, raw_batch in enumerate(loader):
        if max_batches and batch_index >= max_batches:
            break
        batch = move_to_device(raw_batch, device)
        output, predictions = unwrap_model(bundle)(batch, SUBSET_MASKS.to(device), 0.0)
        representations.update(output, batch["dataset_id"], SUBSET_NAMES)
        domain_metrics.observation(output, predictions, batch, SUBSET_NAMES)
        emotion_prediction = predictions["emotion"].argmax(-1)
        for subset in range(len(SUBSET_NAMES)):
            valid = output.valid_subsets[:, subset] & (batch["emotion"] >= 0)
            matrices[subset] += confusion_matrix(
                batch["emotion"], emotion_prediction[:, subset], mask=valid
            )
            intensity_valid = output.valid_subsets[:, subset] & batch.get("intensity_mask", torch.ones_like(batch["intensity"], dtype=torch.bool))
            intensity_sum[subset] += (
                predictions["intensity"][:, subset] - batch["intensity"]
            ).abs()[intensity_valid].sum().cpu()
            counts[subset] += intensity_valid.sum().cpu()
    if context.enabled:
        matrix_tensor = torch.stack(matrices).to(device)
        error_tensor = intensity_sum.to(device)
        count_tensor = counts.to(device)
        dist.all_reduce(matrix_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(error_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
        matrices = [value.cpu() for value in matrix_tensor]
        intensity_sum = error_tensor.cpu()
        counts = count_tensor.cpu()
    metrics = {}
    for subset, name in enumerate(SUBSET_NAMES):
        values = classification_metrics(matrices[subset])
        metrics[f"{name}_macro_f1"] = values["macro_f1"]
        metrics[f"{name}_uar"] = values["uar"]
        metrics[f"{name}_intensity_mae"] = (
            intensity_sum[subset] / counts[subset].clamp_min(1)
        ).item()
    metrics["mean_subset_f1"] = sum(
        metrics[f"{name}_macro_f1"] for name in SUBSET_NAMES
    ) / len(SUBSET_NAMES)
    bundle.train()
    metrics.update(representations.metrics())
    metrics.update(domain_metrics.metrics())
    return metrics


def save_exported_models(run_dir, bundle, teacher, speaker_vocab, metrics) -> None:
    raw_bundle = unwrap_model(bundle)
    torch.save(raw_bundle.encoder.state_dict(), run_dir / "observation_encoder.pt")
    torch.save(teacher.state_dict(), run_dir / "ema_teacher.pt")
    torch.save(
        {
            "heads": raw_bundle.heads.state_dict(),
            "speaker_to_id": speaker_vocab.state_dict(),
            "subset_names": SUBSET_NAMES,
            "metrics": dict(metrics),
        },
        run_dir / "emotion_heads.pt",
    )


def main() -> None:
    cfg, _ = parse_config_args("Stage A0: multimodal emotion observation pretraining")
    resumed = None
    if cfg.TRAIN.RESUME:
        from emotion_ssm.utils.emotion_checkpoint import load_emotion, deployment_paths
        resumed = load_emotion(cfg.TRAIN.RESUME)
        cfg = deployment_paths(resumed[2], cfg)
        from emotion_ssm.utils.checkpoint import validate_resume_data
        validate_resume_data(resumed[3], cfg)
    context = init_distributed(cfg.DEVICE, cfg.SEED, cfg.DETERMINISTIC)
    run_dir = create_run_directory(cfg, "phase_a_observation")
    if context.is_main:
        (run_dir / "config.yaml").write_text(cfg.dump(), encoding="utf-8")

    stores = build_feature_stores(cfg)
    speaker_vocab = build_speaker_vocabulary(stores)
    train_dataset = UnifiedUtteranceDataset(stores, "train", speaker_vocab)
    val_dataset = UnifiedUtteranceDataset(stores, "val", speaker_vocab)
    train_loader, train_sampler = make_loader(
        train_dataset,
        cfg.TRAIN.BATCH_SIZE,
        context,
        True,
        cfg.DATA.NUM_WORKERS,
        cfg.DATA.PIN_MEMORY,
    )
    val_loader, _ = make_loader(
        val_dataset,
        cfg.TRAIN.BATCH_SIZE,
        context,
        False,
        cfg.DATA.NUM_WORKERS,
        cfg.DATA.PIN_MEMORY,
    )

    encoder = ObservationEncoder.from_config(cfg).to(context.device)
    teacher = build_ema_teacher(encoder).to(context.device)
    heads = ObservationSupervisionHeads(
        cfg.MODEL.OBSERVATION_DIM,
        len(speaker_vocab),
        cfg.MODEL.NUM_DOMAINS,
    ).to(context.device)
    if resumed:
        encoder, heads, teacher = resumed[0].encoder.to(context.device), resumed[0].heads.to(context.device), resumed[1].to(context.device)
    bundle = maybe_ddp(ObservationTrainingBundle(encoder, heads), context)
    optimizer = torch.optim.AdamW(
        bundle.parameters(), lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(cfg.TRAIN.EPOCHS, 1)
    )
    scaler = GradScaler(enabled=cfg.TRAIN.AMP and context.device.type == "cuda")
    class_weights = train_dataset.class_weights().to(context.device)
    start_epoch = 0
    global_step = 0
    best_score = -1.0
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
        best_score = float(
            checkpoint.get("metrics", {}).get("mean_subset_f1", best_score)
        )

    warmup_fraction = min(
        cfg.TRAIN.GRL_WARMUP_EPOCHS / max(cfg.TRAIN.EPOCHS, 1), 1.0
    )
    for epoch in range(start_epoch, cfg.TRAIN.EPOCHS):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        bundle.train()
        teacher.eval()
        totals: Dict[str, float] = {}
        batches = 0
        optimizer.zero_grad(set_to_none=True)
        for batch_index, raw_batch in enumerate(train_loader):
            if cfg.TRAIN.DRY_RUN and batch_index >= cfg.TRAIN.DRY_RUN_TRAIN_BATCHES:
                break
            batch = move_to_device(raw_batch, context.device)
            student_batch = augment_student_batch(batch, cfg)
            progress = (epoch + batch_index / max(len(train_loader), 1)) / max(
                cfg.TRAIN.EPOCHS, 1
            )
            strength = grl_alpha(progress, warmup_fraction)
            with autocast(enabled=cfg.TRAIN.AMP and context.device.type == "cuda"):
                output, predictions = bundle(
                    student_batch, SUBSET_MASKS.to(context.device), strength
                )
                with torch.no_grad():
                    teacher_output = teacher(
                        batch, SUBSET_MASKS[-1:].to(context.device)
                    )
                losses = observation_losses(
                    output,
                    teacher_output,
                    predictions,
                    student_batch,
                    class_weights,
                    cfg,
                    SUBSET_MASKS.to(context.device),
                    supervised=epoch >= cfg.TRAIN.SSL_ONLY_EPOCHS,
                )
                scaled_loss = losses["total"] / cfg.TRAIN.GRAD_ACCUMULATION
            scaler.scale(scaled_loss).backward()
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
                update_ema(
                    teacher,
                    unwrap_model(bundle).encoder,
                    cfg.TRAIN.EMA_DECAY,
                )
                global_step += 1
            for name, value in losses.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach())
            batches += 1
            if context.is_main and batches % cfg.TRAIN.LOG_INTERVAL == 0:
                print(json.dumps({"stage": "A0", "epoch": epoch+1, "batch": batches,
                    "global_step": global_step, "loss": float(losses["total"].detach()),
                    "vicreg": float(losses["vicreg"].detach())}), flush=True)
        scheduler.step()
        totals = {name: value / max(batches, 1) for name, value in totals.items()}
        totals = context.reduce_scalars(totals)
        val_metrics = validate(
            bundle,
            val_loader,
            context.device,
            context,
            cfg.TRAIN.DRY_RUN_VAL_BATCHES if cfg.TRAIN.DRY_RUN else 0,
        )
        score = val_metrics["mean_subset_f1"]
        metrics = {
            "epoch": epoch + 1,
            "training_stage": (
                "ssl" if epoch < cfg.TRAIN.SSL_ONLY_EPOCHS else "supervised"
            ),
            **totals,
            **val_metrics,
        }
        if context.is_main:
            append_metrics(run_dir / "metrics.jsonl", metrics)
            print(json.dumps(metrics, ensure_ascii=False))
            save_training_checkpoint(
                run_dir / "last.pt",
                epoch + 1,
                global_step,
                {"bundle": bundle, "teacher": teacher},
                optimizer,
                scheduler,
                scaler,
                metrics,
                config=cfg.dump(),
            )
            if score > best_score:
                best_score = score
                save_training_checkpoint(
                    run_dir / "best.pt",
                    epoch + 1,
                    global_step,
                    {"bundle": bundle, "teacher": teacher},
                    optimizer,
                    scheduler,
                    scaler,
                    metrics,
                    config=cfg.dump(),
                )
                save_exported_models(
                    run_dir, bundle, teacher, speaker_vocab, val_metrics
                )
        context.barrier()
        if cfg.TRAIN.DRY_RUN:
            break


if __name__ == "__main__":
    main()
