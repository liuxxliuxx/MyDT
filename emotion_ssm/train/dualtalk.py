from __future__ import annotations

import inspect
import json
import time
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

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
from emotion_ssm.schema import EventObservation
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
        audio_feature_batch_size: int = 16,
    ) -> None:
        super().__init__()
        self.generator = generator
        self.conditioner = conditioner
        self.projector = projector
        self.state_loss_weight = state_loss_weight
        self.state_anchor_weight = state_anchor_weight
        self.causal_state_context = causal_state_context
        self.audio_feature_batch_size = max(int(audio_feature_batch_size), 1)
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

    @staticmethod
    def _restore_sequence_field(
        value: torch.Tensor, valid_indices: torch.Tensor, batch_size: int, chunk_count: int
    ) -> torch.Tensor:
        """Scatter valid flattened chunks back to [B, chunks, ...]."""
        flat = value.new_zeros((batch_size * chunk_count,) + tuple(value.shape[1:]))
        if value.shape[0] > 0:
            flat.index_copy_(0, valid_indices, value)
        return flat.reshape((batch_size, chunk_count) + tuple(value.shape[1:]))

    @classmethod
    def _restore_observation(
        cls,
        observation: EventObservation,
        valid_indices: torch.Tensor,
        batch_size: int,
        chunk_count: int,
    ) -> EventObservation:
        modality_mask = None
        if observation.modality_mask is not None:
            modality_mask = cls._restore_sequence_field(
                observation.modality_mask, valid_indices, batch_size, chunk_count
            )
        return EventObservation(
            aff=cls._restore_sequence_field(
                observation.aff, valid_indices, batch_size, chunk_count
            ),
            event=cls._restore_sequence_field(
                observation.event, valid_indices, batch_size, chunk_count
            ),
            action=cls._restore_sequence_field(
                observation.action, valid_indices, batch_size, chunk_count
            ),
            reliability=cls._restore_sequence_field(
                observation.reliability, valid_indices, batch_size, chunk_count
            ),
            modality_mask=modality_mask,
        )

    def _precompute_sequence_features(
        self, batch: Mapping[str, torch.Tensor], chunk_mask: torch.Tensor
    ):
        """Batch frozen audio backbones once, then reuse features per chunk.

        The observation encoder and state model remain outside ``no_grad`` so
        gradients still train the DualTalk adapter and dynamics parameters.
        """
        observer = getattr(self.conditioner, "audio_observer", None)
        baseline = getattr(self.generator, "baseline", None)
        joint_encoder = getattr(baseline, "joint_encoder", None)
        if observer is None or not hasattr(observer, "encode_pair"):
            return None
        if joint_encoder is None or not hasattr(
            joint_encoder, "extract_audio_features"
        ) or not hasattr(joint_encoder, "forward_from_audio_features"):
            return None

        batch_size, chunk_count = chunk_mask.shape
        flat_mask = chunk_mask.reshape(-1)
        valid_indices = flat_mask.nonzero(as_tuple=False).flatten()
        target_audio = batch["target_audio"].reshape(batch_size * chunk_count, -1)
        partner_audio = batch["partner_audio"].reshape(batch_size * chunk_count, -1)
        target_valid = target_audio.index_select(0, valid_indices)
        partner_valid = partner_audio.index_select(0, valid_indices)

        # Wav2Vec2 is frozen; extraction is batched in bounded micro-batches to
        # raise GPU occupancy without risking a large activation spike.
        target_embedding, partner_embedding = observer.encode_pair(
            target_valid,
            partner_valid,
            batch_size=self.audio_feature_batch_size,
        )
        target_observation = observer.observations_from_embeddings(target_embedding)
        partner_observation = observer.observations_from_embeddings(partner_embedding)
        with torch.no_grad():
            target_feature, partner_feature = joint_encoder.extract_audio_features(
                target_valid,
                partner_valid,
                batch_size=self.audio_feature_batch_size,
            )

        return {
            "observations": (
                self._restore_observation(
                    target_observation, valid_indices, batch_size, chunk_count
                ),
                self._restore_observation(
                    partner_observation, valid_indices, batch_size, chunk_count
                ),
            ),
            "audio_features": (
                self._restore_sequence_field(
                    target_feature, valid_indices, batch_size, chunk_count
                ),
                self._restore_sequence_field(
                    partner_feature, valid_indices, batch_size, chunk_count
                ),
            ),
        }

    @staticmethod
    def _select_observation_chunk(
        observation: EventObservation, chunk_index: int
    ) -> EventObservation:
        modality_mask = (
            None
            if observation.modality_mask is None
            else observation.modality_mask[:, chunk_index]
        )
        return EventObservation(
            aff=observation.aff[:, chunk_index],
            event=observation.event[:, chunk_index],
            action=observation.action[:, chunk_index],
            reliability=observation.reliability[:, chunk_index],
            modality_mask=modality_mask,
        )

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
        observations: Optional[Tuple[EventObservation, EventObservation]] = None,
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
        if observations is not None and "observations" in accepted:
            conditioner_kwargs["observations"] = observations
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
            target_aff = evidence["target_state_aff"] if "target_state_aff" in evidence else evidence["target_aff"]
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
        audio_features: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        audio_target = batch["target_audio"]
        audio_partner = batch["partner_audio"]
        generator_kwargs = {"enable_film": True}
        if audio_features is not None and "audio_features" in inspect.signature(
            self.generator.forward
        ).parameters:
            generator_kwargs["audio_features"] = audio_features
        generated = self.generator(
            audio_target,
            audio_partner,
            batch["partner_blendshape"],
            context,
            **generator_kwargs,
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

        precomputed = self._precompute_sequence_features(batch, chunk_mask)
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
            observations = None
            audio_features = None
            if precomputed is not None:
                observations = (
                    self._select_observation_chunk(
                        precomputed["observations"][0], chunk_index
                    ),
                    self._select_observation_chunk(
                        precomputed["observations"][1], chunk_index
                    ),
                )
                audio_features = (
                    precomputed["audio_features"][0][:, chunk_index],
                    precomputed["audio_features"][1][:, chunk_index],
                )
            context, candidate_state, target_aff = self._condition_chunk(
                chunk, state, observations=observations
            )
            state = self._merge_state(state, candidate_state, valid)
            losses = self._chunk_losses(
                chunk, context, target_aff, audio_features=audio_features
            )
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
        cfg.DUALTALK.AUDIO_FEATURE_BATCH_SIZE,
    ).to(device)


@torch.no_grad()
def validate(
    system,
    loader,
    context,
    bptt_chunks: int,
    max_batches: int = 0,
    log_interval: int = 0,
    epoch: int = 0,
    progress_path: Optional[Path] = None,
) -> Dict[str, float]:
    system.eval()
    totals: Dict[str, float] = {}
    batches = 0
    total_batches = len(loader)
    if max_batches > 0:
        total_batches = min(total_batches, max_batches)
    validation_start = time.perf_counter()
    for index, raw_batch in enumerate(loader):
        if max_batches and index >= max_batches:
            break
        losses = system(move_to_device(raw_batch, context.device), bptt_chunks)
        for name, value in losses.items():
            totals[name] = totals.get(name, 0.0) + float(value)
        batches += 1
        if (
            context.is_main
            and log_interval > 0
            and (batches % log_interval == 0 or batches == total_batches)
        ):
            elapsed = time.perf_counter() - validation_start
            seconds_per_batch = elapsed / max(batches, 1)
            remaining = max(total_batches - batches, 0)
            record = {
                "phase": "validation_progress",
                "epoch": epoch,
                "batch": batches,
                "total_batches": total_batches,
                "percent": 100.0 * batches / max(total_batches, 1),
                "elapsed_sec": elapsed,
                "sec_per_batch": seconds_per_batch,
                "eta_sec": seconds_per_batch * remaining,
            }
            print(json.dumps(record, ensure_ascii=False), flush=True)
            if progress_path is not None:
                append_metrics(progress_path, record)
    metrics = {f"val_{name}": value / max(batches, 1) for name, value in totals.items()}
    metrics = context.reduce_scalars(metrics)
    system.train()
    return metrics


def main() -> None:
    cfg, _ = parse_config_args("Emotion-conditioned DualTalk training")
    if cfg.DUALTALK.PROTOCOL_VERSION == 2:
        from emotion_ssm.train.generation import run_generation
        return run_generation(cfg)
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
        bucket_by_length=True,
        seed=cfg.SEED,
    )
    val_loader, _ = make_loader(
        val_dataset,
        cfg.TRAIN.SEQUENCE_BATCH_SIZE,
        context,
        False,
        cfg.DATA.NUM_WORKERS,
        cfg.DATA.PIN_MEMORY,
        collate_fn=collate_dualtalk_dialogues,
        bucket_by_length=True,
        seed=cfg.SEED,
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
        total_batches = len(train_loader)
        if cfg.TRAIN.DRY_RUN:
            total_batches = min(total_batches, cfg.TRAIN.DRY_RUN_TRAIN_BATCHES)
        epoch_start = time.perf_counter()
        progress_path = run_dir / "progress.jsonl"
        if context.is_main:
            start_record = {
                "phase": "train_start",
                "epoch": epoch + 1,
                "total_batches": total_batches,
            }
            print(json.dumps(start_record, ensure_ascii=False), flush=True)
            append_metrics(progress_path, start_record)
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
            if (
                context.is_main
                and cfg.TRAIN.LOG_INTERVAL > 0
                and (
                    batches % cfg.TRAIN.LOG_INTERVAL == 0
                    or batches == total_batches
                )
            ):
                elapsed = time.perf_counter() - epoch_start
                seconds_per_batch = elapsed / max(batches, 1)
                remaining = max(total_batches - batches, 0)
                progress_record = {
                    "phase": "train_progress",
                    "epoch": epoch + 1,
                    "batch": batches,
                    "total_batches": total_batches,
                    "percent": 100.0 * batches / max(total_batches, 1),
                    "elapsed_sec": elapsed,
                    "sec_per_batch": seconds_per_batch,
                    "eta_sec": seconds_per_batch * remaining,
                    "loss": float(losses["total"].detach()),
                }
                print(json.dumps(progress_record, ensure_ascii=False), flush=True)
                append_metrics(progress_path, progress_record)
        scheduler.step()
        totals = {name: value / max(batches, 1) for name, value in totals.items()}
        totals = context.reduce_scalars(totals)
        if context.is_main:
            validation_start_record = {
                "phase": "validation_start",
                "epoch": epoch + 1,
            }
            print(json.dumps(validation_start_record, ensure_ascii=False), flush=True)
            append_metrics(progress_path, validation_start_record)
        val_metrics = validate(
            system,
            val_loader,
            context,
            cfg.DUALTALK.STATE_BPTT_CHUNKS,
            cfg.TRAIN.DRY_RUN_VAL_BATCHES if cfg.TRAIN.DRY_RUN else 0,
            cfg.TRAIN.LOG_INTERVAL,
            epoch + 1,
            progress_path,
        )
        if context.is_main:
            validation_done_record = {
                "phase": "validation_done",
                "epoch": epoch + 1,
            }
            print(json.dumps(validation_done_record, ensure_ascii=False), flush=True)
            append_metrics(progress_path, validation_done_record)
        metrics = {"epoch": epoch + 1, **totals, **val_metrics}
        if context.is_main:
            append_metrics(run_dir / "metrics.jsonl", metrics)
            print(json.dumps(metrics, ensure_ascii=False), flush=True)
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
