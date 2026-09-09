from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Dict, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

from emotion_ssm.config import parse_config_args
from emotion_ssm.data import DualTalkChunkDataset
from emotion_ssm.models import (
    AudioOnlyStateObserver,
    BlendshapeAffectProjector,
    DyadicAudioConditioner,
    DyadicEmotionSSM,
    EmotionConditionedDualTalk,
    ObservationEncoder,
    expression_state_consistency,
)
from emotion_ssm.train.common import (
    append_metrics,
    create_run_directory,
    make_loader,
    maybe_ddp,
    move_to_device,
)
from emotion_ssm.train.dynamics_core import load_component_state
from emotion_ssm.utils.checkpoint import load_training_checkpoint, save_training_checkpoint
from emotion_ssm.utils.distributed import init_distributed, unwrap_model


def _load_raw(path: Path):
    kwargs = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = False
    return torch.load(str(path), **kwargs)


def _load_global_dynamics(model: DyadicEmotionSSM, checkpoint: Path) -> None:
    state = load_component_state(checkpoint, "state_model")
    # DualTalk identities are unseen; learned training-actor residual tables are irrelevant.
    state = {
        name: value
        for name, value in state.items()
        if name not in {"personal.baseline_delta.weight", "personal.tau_delta.weight"}
    }
    missing, unexpected = model.load_state_dict(state, strict=False)
    allowed_missing = {
        "personal.baseline_delta.weight",
        "personal.tau_delta.weight",
    }
    if set(missing) != allowed_missing or unexpected:
        raise RuntimeError(
            f"Incompatible Phase B checkpoint, missing={missing}, unexpected={unexpected}"
        )


class ConditionedTrainingSystem(nn.Module):
    def __init__(self, generator, conditioner, projector, state_loss_weight: float) -> None:
        super().__init__()
        self.generator = generator
        self.conditioner = conditioner
        self.projector = projector
        self.state_loss_weight = state_loss_weight
        self.state_frozen = True

    def set_state_frozen(self, frozen: bool) -> None:
        self.state_frozen = frozen
        self.conditioner.requires_grad_(not frozen)
        if frozen:
            self.conditioner.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.state_frozen:
            self.conditioner.eval()
        return self

    def forward(self, batch: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        audio_target = batch["target_audio"]
        audio_partner = batch["partner_audio"]
        if self.state_frozen:
            with torch.no_grad():
                context, _, evidence = self.conditioner(
                    audio_target, audio_partner, batch["dt"]
                )
            context = context.detach()
            target_aff = evidence["target_aff"].detach()
        else:
            context, _, evidence = self.conditioner(
                audio_target, audio_partner, batch["dt"]
            )
            target_aff = evidence["target_aff"]
        generated = self.generator(
            audio_target,
            audio_partner,
            batch["partner_blendshape"],
            context,
            enable_film=True,
        )
        target = batch["target_blendshape"]
        length = min(generated.shape[1], target.shape[1])
        generated = generated[:, :length]
        target = target[:, :length]
        expression = F.mse_loss(generated[:, :, :50], target[:, :, :50])
        jaw = F.mse_loss(generated[:, :, 50:53], target[:, :, 50:53])
        neck = F.mse_loss(generated[:, :, 53:56], target[:, :, 53:56])
        generated_velocity = generated[:, 1:] - generated[:, :-1]
        target_velocity = target[:, 1:] - target[:, :-1]
        velocity = F.mse_loss(generated_velocity, target_velocity)
        state_consistency = expression_state_consistency(
            self.projector, generated, target_aff
        )
        total = expression + jaw + neck + velocity + self.state_loss_weight * state_consistency
        return {
            "total": total,
            "expression": expression.detach(),
            "jaw": jaw.detach(),
            "neck": neck.detach(),
            "velocity": velocity.detach(),
            "state_consistency": state_consistency.detach(),
        }


def _build_system(cfg, device: torch.device) -> ConditionedTrainingSystem:
    generator = EmotionConditionedDualTalk.from_config(cfg)
    baseline_path = Path(cfg.DUALTALK.BASELINE_CHECKPOINT)
    if baseline_path.is_file():
        generator.load_baseline_state_dict(_load_raw(baseline_path), strict=True)
    elif not cfg.TRAIN.DRY_RUN:
        raise FileNotFoundError(
            "DUALTALK.BASELINE_CHECKPOINT is required except for a synthetic dry run"
        )

    observation = ObservationEncoder.from_config(cfg)
    observation_path = Path(cfg.TRAIN.OBSERVATION_CHECKPOINT)
    if not observation_path.is_file():
        raise FileNotFoundError("TRAIN.OBSERVATION_CHECKPOINT is required")
    observation.load_state_dict(
        load_component_state(observation_path, "encoder"), strict=True
    )
    state_model = DyadicEmotionSSM.from_config(cfg, num_speakers=1)
    phase_b_path = Path(cfg.DUALTALK.PHASE_B_CHECKPOINT or cfg.TRAIN.PHASE_B_CHECKPOINT)
    if not phase_b_path.is_file():
        raise FileNotFoundError("DUALTALK.PHASE_B_CHECKPOINT is required")
    observation.load_state_dict(
        load_component_state(phase_b_path, "encoder"), strict=True
    )
    _load_global_dynamics(state_model, phase_b_path)
    audio_observer = AudioOnlyStateObserver(
        observation,
        cfg.DUALTALK.AUDIO_MODEL,
        cfg.DUALTALK.LOCAL_FILES_ONLY,
    )
    conditioner = DyadicAudioConditioner(audio_observer, state_model)
    projector = BlendshapeAffectProjector(
        cfg.DUALTALK.BLENDSHAPE_DIM, cfg.MODEL.OBSERVATION_DIM
    )
    return ConditionedTrainingSystem(
        generator, conditioner, projector, cfg.LOSS.GENERATION_STATE
    ).to(device)


@torch.no_grad()
def validate(system, loader, context, max_batches: int = 0) -> Dict[str, float]:
    system.eval()
    totals: Dict[str, float] = {}
    batches = 0
    for index, raw_batch in enumerate(loader):
        if max_batches and index >= max_batches:
            break
        losses = system(move_to_device(raw_batch, context.device))
        for name, value in losses.items():
            totals[name] = totals.get(name, 0.0) + float(value)
        batches += 1
    metrics = {f"val_{name}": value / max(batches, 1) for name, value in totals.items()}
    metrics = context.reduce_scalars(metrics)
    system.train()
    return metrics


def main() -> None:
    cfg, _ = parse_config_args("Emotion-conditioned DualTalk training")
    context = init_distributed(cfg.DEVICE, cfg.SEED, cfg.DETERMINISTIC)
    run_dir = create_run_directory(cfg, "dualtalk_conditioned")
    if context.is_main:
        (run_dir / "config.yaml").write_text(cfg.dump(), encoding="utf-8")
    train_dataset = DualTalkChunkDataset(
        Path(cfg.DATA.DUALTALK_ROOT) / "train",
        cfg.DUALTALK.CHUNK_FRAMES,
        cfg.DUALTALK.FPS,
    )
    val_dataset = DualTalkChunkDataset(
        Path(cfg.DATA.DUALTALK_ROOT) / "test",
        cfg.DUALTALK.CHUNK_FRAMES,
        cfg.DUALTALK.FPS,
    )
    train_loader, train_sampler = make_loader(
        train_dataset,
        cfg.TRAIN.SEQUENCE_BATCH_SIZE,
        context,
        True,
        cfg.DATA.NUM_WORKERS,
        cfg.DATA.PIN_MEMORY,
    )
    val_loader, _ = make_loader(
        val_dataset,
        cfg.TRAIN.SEQUENCE_BATCH_SIZE,
        context,
        False,
        cfg.DATA.NUM_WORKERS,
        cfg.DATA.PIN_MEMORY,
    )
    system = _build_system(cfg, context.device)
    generator_parameters = list(system.generator.parameters()) + list(system.projector.parameters())
    state_parameters = list(system.conditioner.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": generator_parameters, "lr": cfg.TRAIN.LR},
            {
                "params": state_parameters,
                "lr": cfg.TRAIN.LR * cfg.DUALTALK.JOINT_FINETUNE_LR_SCALE,
            },
        ],
        weight_decay=cfg.TRAIN.WEIGHT_DECAY,
    )
    system = maybe_ddp(system, context)
    unwrap_model(system).set_state_frozen(True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(cfg.TRAIN.EPOCHS, 1)
    )
    scaler = GradScaler(enabled=cfg.TRAIN.AMP and context.device.type == "cuda")
    start_epoch = 0
    global_step = 0
    best_score = float("inf")
    if cfg.TRAIN.RESUME:
        checkpoint = load_training_checkpoint(
            Path(cfg.TRAIN.RESUME),
            {"system": system},
            optimizer,
            scheduler,
            scaler,
        )
        start_epoch = int(checkpoint["epoch"])
        global_step = int(checkpoint["global_step"])
        best_score = float(checkpoint.get("metrics", {}).get("val_total", best_score))

    stop_epoch = (
        min(cfg.TRAIN.EPOCHS, cfg.TRAIN.STOP_AFTER_EPOCHS)
        if cfg.TRAIN.STOP_AFTER_EPOCHS > 0
        else cfg.TRAIN.EPOCHS
    )
    for epoch in range(start_epoch, stop_epoch):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        unwrap_model(system).set_state_frozen(epoch < cfg.DUALTALK.FREEZE_STATE_EPOCHS)
        system.train()
        totals: Dict[str, float] = {}
        batches = 0
        optimizer.zero_grad(set_to_none=True)
        for index, raw_batch in enumerate(train_loader):
            if cfg.TRAIN.DRY_RUN and index >= cfg.TRAIN.DRY_RUN_TRAIN_BATCHES:
                break
            batch = move_to_device(raw_batch, context.device)
            with autocast(enabled=cfg.TRAIN.AMP and context.device.type == "cuda"):
                losses = system(batch)
                loss = losses["total"] / cfg.TRAIN.GRAD_ACCUMULATION
            scaler.scale(loss).backward()
            should_step = (
                (index + 1) % cfg.TRAIN.GRAD_ACCUMULATION == 0
                or index + 1 == len(train_loader)
                or (
                    cfg.TRAIN.DRY_RUN
                    and index + 1 == cfg.TRAIN.DRY_RUN_TRAIN_BATCHES
                )
            )
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(system.parameters(), cfg.TRAIN.GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
            for name, value in losses.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach())
            batches += 1
        scheduler.step()
        totals = {name: value / max(batches, 1) for name, value in totals.items()}
        totals = context.reduce_scalars(totals)
        val_metrics = validate(
            system,
            val_loader,
            context,
            cfg.TRAIN.DRY_RUN_VAL_BATCHES if cfg.TRAIN.DRY_RUN else 0,
        )
        metrics = {"epoch": epoch + 1, **totals, **val_metrics}
        if context.is_main:
            append_metrics(run_dir / "metrics.jsonl", metrics)
            print(json.dumps(metrics, ensure_ascii=False))
            models = {"system": system}
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
            if val_metrics["val_total"] < best_score:
                best_score = val_metrics["val_total"]
                save_training_checkpoint(
                    run_dir / "dualtalk_emotion_best.pt",
                    epoch + 1,
                    global_step,
                    models,
                    optimizer,
                    scheduler,
                    scaler,
                    metrics,
                    config=cfg.dump(),
                )
        context.barrier()
        if cfg.TRAIN.DRY_RUN:
            break


if __name__ == "__main__":
    main()
