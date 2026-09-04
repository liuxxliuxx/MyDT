from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Mapping

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

from emotion_ssm.config import load_config
from emotion_ssm.data import DualTalkChunkDataset
from emotion_ssm.models import EmotionConditionedDualTalk
from emotion_ssm.train.common import (
    append_metrics,
    create_run_directory,
    make_loader,
    maybe_ddp,
    move_to_device,
)
from emotion_ssm.train.dualtalk import _load_raw
from emotion_ssm.utils.checkpoint import save_training_checkpoint
from emotion_ssm.utils.distributed import init_distributed


def reconstruction_losses(
    generated: torch.Tensor, target: torch.Tensor
) -> Dict[str, torch.Tensor]:
    length = min(generated.shape[1], target.shape[1])
    generated = generated[:, :length]
    target = target[:, :length]
    expression = F.mse_loss(generated[:, :, :50], target[:, :, :50])
    jaw = F.mse_loss(generated[:, :, 50:53], target[:, :, 50:53])
    neck = F.mse_loss(generated[:, :, 53:56], target[:, :, 53:56])
    velocity = F.mse_loss(
        generated[:, 1:] - generated[:, :-1], target[:, 1:] - target[:, :-1]
    )
    total = expression + jaw + neck + velocity
    return {
        "total": total,
        "expression": expression.detach(),
        "jaw": jaw.detach(),
        "neck": neck.detach(),
        "velocity": velocity.detach(),
    }


def forward_losses(model, batch: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    generated = model(
        batch["target_audio"],
        batch["partner_audio"],
        batch["partner_blendshape"],
    )
    return reconstruction_losses(generated, batch["target_blendshape"])


@torch.no_grad()
def validate(model, loader, context) -> Dict[str, float]:
    model.eval()
    totals: Dict[str, float] = {}
    batches = 0
    for raw_batch in loader:
        batch = move_to_device(raw_batch, context.device)
        with autocast(enabled=context.device.type == "cuda"):
            losses = forward_losses(model, batch)
        for name, value in losses.items():
            totals[name] = totals.get(name, 0.0) + float(value)
        batches += 1
    metrics = {
        f"val_{name}": value / max(batches, 1) for name, value in totals.items()
    }
    return context.reduce_scalars(metrics)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Equal-compute DualTalk-only fine-tuning control"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--epochs-to-run", type=int, default=3)
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.epochs_to_run <= 0:
        parser.error("--epochs-to-run must be positive")
    cfg = load_config(args.config, args.opts)
    if args.epochs_to_run > cfg.TRAIN.EPOCHS:
        parser.error("--epochs-to-run cannot exceed TRAIN.EPOCHS")

    context = init_distributed(cfg.DEVICE, cfg.SEED, cfg.DETERMINISTIC)
    run_dir = create_run_directory(cfg, "dualtalk_baseline_control")
    if context.is_main:
        (run_dir / "config.yaml").write_text(cfg.dump(), encoding="utf-8")
        (run_dir / "control.json").write_text(
            json.dumps(
                {
                    "epochs_to_run": args.epochs_to_run,
                    "scheduler_total_epochs": cfg.TRAIN.EPOCHS,
                    "conditioning": False,
                    "state_consistency": False,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

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

    wrapper = EmotionConditionedDualTalk.from_config(cfg)
    baseline_path = Path(cfg.DUALTALK.BASELINE_CHECKPOINT)
    if not baseline_path.is_file():
        raise FileNotFoundError(f"Baseline checkpoint does not exist: {baseline_path}")
    wrapper.load_baseline_state_dict(_load_raw(baseline_path), strict=True)
    model = maybe_ddp(wrapper.baseline.to(context.device), context)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(cfg.TRAIN.EPOCHS, 1)
    )
    scaler = GradScaler(enabled=cfg.TRAIN.AMP and context.device.type == "cuda")
    best_score = float("inf")
    global_step = 0

    for epoch in range(args.epochs_to_run):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        totals: Dict[str, float] = {}
        batches = 0
        optimizer.zero_grad(set_to_none=True)
        for index, raw_batch in enumerate(train_loader):
            batch = move_to_device(raw_batch, context.device)
            with autocast(enabled=cfg.TRAIN.AMP and context.device.type == "cuda"):
                losses = forward_losses(model, batch)
                loss = losses["total"] / cfg.TRAIN.GRAD_ACCUMULATION
            scaler.scale(loss).backward()
            should_step = (
                (index + 1) % cfg.TRAIN.GRAD_ACCUMULATION == 0
                or index + 1 == len(train_loader)
            )
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.TRAIN.GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
            for name, value in losses.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach())
            batches += 1

        scheduler.step()
        train_metrics = {
            name: value / max(batches, 1) for name, value in totals.items()
        }
        train_metrics = context.reduce_scalars(train_metrics)
        val_metrics = validate(model, val_loader, context)
        metrics = {"epoch": epoch + 1, **train_metrics, **val_metrics}
        if context.is_main:
            append_metrics(run_dir / "metrics.jsonl", metrics)
            print(json.dumps(metrics, ensure_ascii=False))
            save_training_checkpoint(
                run_dir / "last.pt",
                epoch + 1,
                global_step,
                {"baseline": model},
                optimizer,
                scheduler,
                scaler,
                metrics,
                config=cfg.dump(),
            )
            if val_metrics["val_total"] < best_score:
                best_score = val_metrics["val_total"]
                save_training_checkpoint(
                    run_dir / "baseline_control_best.pt",
                    epoch + 1,
                    global_step,
                    {"baseline": model},
                    optimizer,
                    scheduler,
                    scaler,
                    metrics,
                    config=cfg.dump(),
                )


if __name__ == "__main__":
    main()
