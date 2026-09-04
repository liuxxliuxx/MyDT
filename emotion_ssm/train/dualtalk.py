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
from emotion_ssm.data import (
    DualTalkDialogueDataset,
    collate_dualtalk_dialogues,
)
from emotion_ssm.models import (
    AudioOnlyStateObserver,
    BlendshapeAffectProjector,
    DyadicAudioConditioner,
    DyadicEmotionSSM,
    EmotionConditionedDualTalk,
    ObservationEncoder,
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
    def __init__(
        self,
        generator,
        conditioner,
        projector,
        state_loss_weight: float,
        state_anchor_weight: float = 0.0,
        causal_state_context: bool = False,
    ) -> None:
        super().__init__()
        self.generator = generator
        self.conditioner = conditioner
        self.projector = projector
        self.state_loss_weight = state_loss_weight
        self.state_anchor_weight = state_anchor_weight
        self.causal_state_context = causal_state_context
        self.state_frozen = True
        state_model = getattr(conditioner, "state_model", None)
        self._state_anchor_names = []
        if state_model is not None:
            for index, (name, parameter) in enumerate(state_model.named_parameters()):
                self._state_anchor_names.append(name)
                self.register_buffer(
                    f"_state_anchor_{index}",
                    parameter.detach().clone(),
                    persistent=False,
                )
        self.set_state_frozen(True)

    def set_state_frozen(self, frozen: bool) -> None:
        self.state_frozen = frozen
        self.conditioner.requires_grad_(False)
        for module in self._adapter_finetune_modules():
            module.requires_grad_(True)
        if not frozen:
            state_model = getattr(self.conditioner, "state_model", None)
            if state_model is not None:
                state_model.requires_grad_(True)
        audio_observer = getattr(self.conditioner, "audio_observer", None)
        backbone = getattr(audio_observer, "backbone", None)
        if backbone is not None:
            backbone.requires_grad_(False)
            backbone.eval()
        if frozen:
            self.conditioner.eval()

    def _state_finetune_modules(self):
        modules = []
        state_model = getattr(self.conditioner, "state_model", None)
        if state_model is not None:
            modules.append(state_model)
        return modules

    def _adapter_finetune_modules(self):
        modules = []
        audio_observer = getattr(self.conditioner, "audio_observer", None)
        observer = getattr(audio_observer, "observation_encoder", None)
        if observer is not None:
            domain_id = int(getattr(audio_observer, "dataset_id", 2))
            modules.append(observer.audio_adapter.adapters[domain_id])
        return modules

    def state_finetune_parameters(self):
        seen = set()
        parameters = []
        for module in self._state_finetune_modules():
            for parameter in module.parameters():
                if id(parameter) not in seen:
                    seen.add(id(parameter))
                    parameters.append(parameter)
        return parameters

    def adapter_finetune_parameters(self):
        return [
            parameter
            for module in self._adapter_finetune_modules()
            for parameter in module.parameters()
        ]

    def restore_shared_observer(self, phase_b_encoder_state) -> None:
        """Restore Phase-B coordinates while retaining the learned domain-2 adapter."""
        audio_observer = getattr(self.conditioner, "audio_observer", None)
        observer = getattr(audio_observer, "observation_encoder", None)
        if observer is None:
            return
        domain_id = int(getattr(audio_observer, "dataset_id", 2))
        adapter_state = {
            name: value.detach().clone()
            for name, value in observer.audio_adapter.adapters[
                domain_id
            ].state_dict().items()
        }
        observer.load_state_dict(phase_b_encoder_state, strict=True)
        observer.audio_adapter.adapters[domain_id].load_state_dict(adapter_state)

    def set_baseline_frozen(self, frozen: bool) -> None:
        """Train FiLM first, then expose only the baseline synthesis head."""
        self.generator.baseline.requires_grad_(False)
        self.generator.film.requires_grad_(True)
        if not frozen:
            self.generator.baseline.synthesis_module.requires_grad_(True)

    def train(self, mode: bool = True):
        super().train(mode)
        audio_observer = getattr(self.conditioner, "audio_observer", None)
        observer = getattr(audio_observer, "observation_encoder", None)
        if observer is not None:
            # The shared emotion coordinate system remains deterministic and
            # frozen. Only the new DualTalk audio adapter learns domain shift.
            observer.eval()
            domain_id = int(getattr(audio_observer, "dataset_id", 2))
            observer.audio_adapter.adapters[domain_id].train(mode)
        backbone = getattr(audio_observer, "backbone", None)
        if backbone is not None:
            backbone.eval()
        state_model = getattr(self.conditioner, "state_model", None)
        if state_model is not None:
            state_model.train(mode and not self.state_frozen)
        return self

    @staticmethod
    def _merge_state(previous, candidate, valid_mask: torch.Tensor):
        """Keep the previous state for padded dialogue positions."""
        if previous is None:
            return candidate
        state_mask = valid_mask[:, None, None]
        relation_mask = valid_mask[:, None]
        return type(candidate)(
            z=torch.where(state_mask, candidate.z, previous.z),
            relation=torch.where(relation_mask, candidate.relation, previous.relation),
            speaker_ids=previous.speaker_ids,
        )

    def _condition_chunk(
        self,
        batch: Mapping[str, torch.Tensor],
        state=None,
        enable_partner: bool = True,
    ):
        audio_target = batch["target_audio"]
        audio_partner = batch["partner_audio"]
        use_causal_context = (
            self.causal_state_context
            and hasattr(self.conditioner, "initialize_state")
            and hasattr(self.conditioner, "state_context")
        )
        if use_causal_context and state is None:
            state = self.conditioner.initialize_state(audio_target)
        conditioner_kwargs = {} if state is None else {"state": state}
        accepted = inspect.signature(self.conditioner.forward).parameters
        if "enable_partner" in accepted:
            conditioner_kwargs["enable_partner"] = enable_partner
        for name in ("target_speech_active", "partner_speech_active"):
            if name in batch and name in accepted:
                conditioner_kwargs[name] = batch[name]
        if use_causal_context:
            context = self.conditioner.state_context(state)
            target_aff = self.conditioner.state_model.state_to_aff(state.z[:, 0])
        # requires_grad flags freeze the state parameters. Keep autograd active
        # through state values so adapter 2 can learn from later causal chunks.
        updated_context, next_state, evidence = self.conditioner(
            audio_target, audio_partner, batch["dt"], **conditioner_kwargs
        )
        if not use_causal_context:
            context = updated_context
            target_aff = evidence.get("target_state_aff", evidence["target_aff"])
        return context, next_state, target_aff

    def _state_anchor_loss(self, reference: torch.Tensor) -> torch.Tensor:
        state_model = getattr(self.conditioner, "state_model", None)
        if state_model is None or not self._state_anchor_names:
            return reference.sum() * 0.0
        losses = []
        current = dict(state_model.named_parameters())
        for index, name in enumerate(self._state_anchor_names):
            parameter = current[name]
            anchor = getattr(self, f"_state_anchor_{index}")
            losses.append((parameter - anchor).square().mean())
        return torch.stack(losses).mean() if losses else reference.sum() * 0.0

    @staticmethod
    def _per_sample_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (prediction - target).square().flatten(1).mean(dim=1)

    def _chunk_losses(
        self,
        batch: Mapping[str, torch.Tensor],
        context: torch.Tensor,
        target_aff: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        audio_target = batch["target_audio"]
        audio_partner = batch["partner_audio"]
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
        expression = self._per_sample_mse(generated[:, :, :50], target[:, :, :50])
        jaw = self._per_sample_mse(generated[:, :, 50:53], target[:, :, 50:53])
        neck = self._per_sample_mse(generated[:, :, 53:56], target[:, :, 53:56])
        if length > 1:
            velocity = self._per_sample_mse(
                generated[:, 1:] - generated[:, :-1],
                target[:, 1:] - target[:, :-1],
            )
        else:
            velocity = generated.new_zeros(generated.shape[0])
        generated_aff = self.projector(generated)
        state_consistency = 1.0 - F.cosine_similarity(
            generated_aff, target_aff, dim=-1
        )
        return {
            "expression": expression,
            "jaw": jaw,
            "neck": neck,
            "velocity": velocity,
            "state_consistency": state_consistency,
        }

    def forward_sequence(
        self,
        batch: Mapping[str, torch.Tensor],
        bptt_chunks: int = 1,
    ) -> Dict[str, torch.Tensor]:
        """Generate an ordered dialogue while retaining only causal state.

        The collator pads just the chunk axis. Padded chunks may still pass
        through the batched modules for shape consistency, but their loss and
        their state update are both masked out.
        """
        if bptt_chunks < 1:
            raise ValueError("bptt_chunks must be at least one")
        chunk_mask = batch["chunk_mask"].bool()
        if chunk_mask.ndim != 2:
            raise ValueError("chunk_mask must have shape [batch, chunks]")
        batch_size, chunk_count = chunk_mask.shape
        if batch["target_audio"].shape[:2] != (batch_size, chunk_count):
            raise ValueError("target_audio and chunk_mask disagree on dialogue axes")

        sums = None
        valid_count = chunk_mask.sum().to(dtype=batch["target_audio"].dtype)
        state = None
        for chunk_index in range(chunk_count):
            valid = chunk_mask[:, chunk_index]
            chunk = {
                name: value[:, chunk_index]
                for name, value in batch.items()
                if name in {
                    "target_audio",
                    "partner_audio",
                    "target_blendshape",
                    "partner_blendshape",
                    "dt",
                    "target_speech_active",
                    "partner_speech_active",
                }
            }
            context, candidate_state, target_aff = self._condition_chunk(chunk, state)
            state = self._merge_state(state, candidate_state, valid)
            losses = self._chunk_losses(chunk, context, target_aff)
            if sums is None:
                sums = {name: value.new_zeros(()) for name, value in losses.items()}
            weight = valid.to(dtype=next(iter(losses.values())).dtype)
            for name, value in losses.items():
                sums[name] = sums[name] + (value * weight).sum()
            if (chunk_index + 1) % bptt_chunks == 0:
                state = state.detach()

        if sums is None:
            zero = batch["target_audio"].sum() * 0.0
            sums = {
                name: zero
                for name in ("expression", "jaw", "neck", "velocity", "state_consistency")
            }
        divisor = valid_count.clamp_min(1.0)
        means = {name: value / divisor for name, value in sums.items()}
        state_anchor = self._state_anchor_loss(batch["target_audio"])
        total = (
            means["expression"]
            + means["jaw"]
            + means["neck"]
            + means["velocity"]
            + self.state_loss_weight * means["state_consistency"]
            + self.state_anchor_weight * state_anchor
        )
        return {
            "total": total,
            "state_anchor": state_anchor.detach(),
            **{name: value.detach() for name, value in means.items()},
        }

    def forward(
        self,
        batch: Mapping[str, torch.Tensor],
        bptt_chunks: int = 1,
    ) -> Dict[str, torch.Tensor]:
        """Single-chunk compatibility entry point used by old callers/tests."""
        if batch["target_audio"].ndim == 3:
            return self.forward_sequence(batch, bptt_chunks)
        context, _, target_aff = self._condition_chunk(batch)
        losses = self._chunk_losses(batch, context, target_aff)
        means = {name: value.mean() for name, value in losses.items()}
        state_anchor = self._state_anchor_loss(batch["target_audio"])
        total = (
            means["expression"]
            + means["jaw"]
            + means["neck"]
            + means["velocity"]
            + self.state_loss_weight * means["state_consistency"]
            + self.state_anchor_weight * state_anchor
        )
        return {
            "total": total,
            "state_anchor": state_anchor.detach(),
            **{name: value.detach() for name, value in means.items()},
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
    observation.copy_domain_adapters(
        int(cfg.DUALTALK.ADAPTER_INIT_SOURCE),
        int(cfg.DUALTALK.ADAPTER_DOMAIN_ID),
    )
    _load_global_dynamics(state_model, phase_b_path)
    audio_observer = AudioOnlyStateObserver(
        observation,
        cfg.DUALTALK.AUDIO_MODEL,
        cfg.DUALTALK.LOCAL_FILES_ONLY,
        cfg.DUALTALK.ADAPTER_DOMAIN_ID,
    )
    conditioner = DyadicAudioConditioner(audio_observer, state_model)
    projector = BlendshapeAffectProjector(
        cfg.DUALTALK.BLENDSHAPE_DIM, cfg.MODEL.OBSERVATION_DIM
    )
    return ConditionedTrainingSystem(
        generator,
        conditioner,
        projector,
        cfg.LOSS.GENERATION_STATE,
        cfg.LOSS.DUALTALK_STATE_ANCHOR,
        cfg.DUALTALK.CAUSAL_STATE_CONTEXT,
    ).to(device)


@torch.no_grad()
def validate(
    system,
    loader,
    context,
    bptt_chunks: int,
    max_batches: int = 0,
) -> Dict[str, float]:
    system.eval()
    totals: Dict[str, float] = {}
    batches = 0
    for index, raw_batch in enumerate(loader):
        if max_batches and index >= max_batches:
            break
        losses = system(move_to_device(raw_batch, context.device), bptt_chunks)
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
    train_dataset = DualTalkDialogueDataset(
        Path(cfg.DATA.DUALTALK_ROOT) / "train",
        cfg.DUALTALK.CHUNK_FRAMES,
        cfg.DUALTALK.FPS,
        speech_rms_threshold=cfg.DUALTALK.SPEECH_RMS_THRESHOLD,
    )
    val_dataset = DualTalkDialogueDataset(
        Path(cfg.DATA.DUALTALK_ROOT) / "test",
        cfg.DUALTALK.CHUNK_FRAMES,
        cfg.DUALTALK.FPS,
        speech_rms_threshold=cfg.DUALTALK.SPEECH_RMS_THRESHOLD,
    )
    train_loader, train_sampler = make_loader(
        train_dataset,
        cfg.TRAIN.SEQUENCE_BATCH_SIZE,
        context,
        True,
        cfg.DATA.NUM_WORKERS,
        cfg.DATA.PIN_MEMORY,
        collate_fn=collate_dualtalk_dialogues,
    )
    val_loader, _ = make_loader(
        val_dataset,
        cfg.TRAIN.SEQUENCE_BATCH_SIZE,
        context,
        False,
        cfg.DATA.NUM_WORKERS,
        cfg.DATA.PIN_MEMORY,
        collate_fn=collate_dualtalk_dialogues,
    )
    system = _build_system(cfg, context.device)
    film_and_projector_parameters = list(system.generator.film.parameters()) + list(
        system.projector.parameters()
    )
    synthesis_parameters = list(system.generator.baseline.synthesis_module.parameters())
    # Preserve the historical optimizer group layout for checkpoint resume.
    # requires_grad controls which conditioner parameters actually update.
    state_parameters = list(system.conditioner.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": film_and_projector_parameters, "lr": cfg.TRAIN.LR},
            {
                "params": synthesis_parameters,
                "lr": cfg.TRAIN.LR * cfg.DUALTALK.SYNTHESIS_FINETUNE_LR_SCALE,
            },
            {
                "params": state_parameters,
                "lr": cfg.TRAIN.LR * cfg.DUALTALK.JOINT_FINETUNE_LR_SCALE,
            },
        ],
        weight_decay=cfg.TRAIN.WEIGHT_DECAY,
    )
    system = maybe_ddp(system, context)
    unwrap_model(system).set_state_frozen(True)
    unwrap_model(system).set_baseline_frozen(True)
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
        unwrap_model(system).restore_shared_observer(
            load_component_state(
                Path(cfg.DUALTALK.PHASE_B_CHECKPOINT or cfg.TRAIN.PHASE_B_CHECKPOINT),
                "encoder",
            )
        )

    stop_epoch = (
        min(cfg.TRAIN.EPOCHS, cfg.TRAIN.STOP_AFTER_EPOCHS)
        if cfg.TRAIN.STOP_AFTER_EPOCHS > 0
        else cfg.TRAIN.EPOCHS
    )
    for epoch in range(start_epoch, stop_epoch):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        unwrap_model(system).set_state_frozen(epoch < cfg.DUALTALK.FREEZE_STATE_EPOCHS)
        unwrap_model(system).set_baseline_frozen(
            epoch < cfg.DUALTALK.FREEZE_BASELINE_EPOCHS
        )
        system.train()
        totals: Dict[str, float] = {}
        batches = 0
        optimizer.zero_grad(set_to_none=True)
        for index, raw_batch in enumerate(train_loader):
            if cfg.TRAIN.DRY_RUN and index >= cfg.TRAIN.DRY_RUN_TRAIN_BATCHES:
                break
            batch = move_to_device(raw_batch, context.device)
            with autocast(enabled=cfg.TRAIN.AMP and context.device.type == "cuda"):
                losses = system(batch, cfg.DUALTALK.STATE_BPTT_CHUNKS)
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
            cfg.DUALTALK.STATE_BPTT_CHUNKS,
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
