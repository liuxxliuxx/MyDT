from __future__ import annotations

import inspect
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from emotion_ssm.losses import masked_mean, state_prediction_losses, zero_loss
from emotion_ssm.models import AffectDecoder, DyadicEmotionSSM, ObservationEncoder
from emotion_ssm.models.counterfactual import match_counterfactuals
from emotion_ssm.schema import DyadicState, EventObservation, replace_role, select_role


FULL_AVT_MASK = torch.tensor([[1, 1, 1]], dtype=torch.bool)


def _torch_load(path: Path):
    kwargs = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = False
    return torch.load(str(path), **kwargs)


def load_component_state(path: Path, component: str = "encoder") -> Mapping[str, Tensor]:
    """Read exported weights or a complete training checkpoint."""
    payload = _torch_load(path)
    if isinstance(payload, Mapping) and "models" in payload:
        models = payload["models"]
        if component in models:
            return models[component]
        if "bundle" in models:
            prefix = component + "."
            selected = {
                name[len(prefix) :]: value
                for name, value in models["bundle"].items()
                if name.startswith(prefix)
            }
            if selected:
                return selected
    if isinstance(payload, Mapping) and component in payload:
        value = payload[component]
        if isinstance(value, Mapping):
            return value
    if not isinstance(payload, Mapping):
        raise ValueError(f"Unsupported checkpoint payload: {path}")
    return payload


def flatten_sequence_batch(batch: Mapping[str, Tensor]) -> Dict[str, Tensor]:
    """Convert [B,L,...] observation fields to the encoder's [B*L,...] API."""
    batch_size, length = batch["audio"].shape[:2]
    fields = ("audio", "face", "text", "modality_mask", "reliability", "dataset_id")
    return {
        name: batch[name].reshape((batch_size * length,) + batch[name].shape[2:])
        for name in fields
    }


def swap_dialogue_roles(batch: Mapping[str, Tensor], probability: float = 0.5) -> Dict[str, Tensor]:
    """Randomly rename A/B per window without touching the chronological event order."""
    output = dict(batch)
    batch_size = batch["active_role"].shape[0]
    swap = torch.rand(batch_size, device=batch["active_role"].device) < probability
    output["active_role"] = torch.where(
        swap[:, None], 1 - batch["active_role"], batch["active_role"]
    )
    output["speaker_ids"] = torch.where(
        swap[:, None], batch["speaker_ids"].flip(1), batch["speaker_ids"]
    )
    return output


def _reshape_event(output, batch_size: int, length: int) -> EventObservation:
    return EventObservation(
        aff=output.aff[:, 0].reshape(batch_size, length, -1),
        event=output.event[:, 0].reshape(batch_size, length, -1),
        action=output.action[:, 0].reshape(batch_size, length, -1),
        reliability=output.reliability[:, 0].reshape(batch_size, length, -1),
        modality_mask=None,
    )


def event_at(event: EventObservation, turn: int) -> EventObservation:
    return EventObservation(
        aff=event.aff[:, turn],
        event=event.event[:, turn],
        action=event.action[:, turn],
        reliability=event.reliability[:, turn],
        modality_mask=(
            None if event.modality_mask is None else event.modality_mask[:, turn]
        ),
    )


def _flatten_event_range(event: EventObservation, start: int, stop: int) -> EventObservation:
    def flatten(value: Tensor) -> Tensor:
        return value[:, start:stop].reshape(-1, value.shape[-1])

    return EventObservation(
        aff=flatten(event.aff),
        event=flatten(event.event),
        action=flatten(event.action),
        reliability=flatten(event.reliability),
        modality_mask=(
            None if event.modality_mask is None else flatten(event.modality_mask)
        ),
    )


def _where_state(mask: Tensor, new: DyadicState, old: DyadicState) -> DyadicState:
    return DyadicState(
        z=torch.where(mask[:, None, None], new.z, old.z),
        relation=torch.where(mask[:, None], new.relation, old.relation),
        speaker_ids=old.speaker_ids,
    )


class DynamicsTrainingBundle(nn.Module):
    """Observation-to-state training graph used by Phase A and Phase B."""

    def __init__(
        self,
        encoder: ObservationEncoder,
        state_model: DyadicEmotionSSM,
        decoder: AffectDecoder,
        cfg,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.state_model = state_model
        self.decoder = decoder
        self.horizons = tuple(int(value) for value in cfg.DYNAMICS.HORIZONS)
        self.correction_mode = str(cfg.DYNAMICS.CORRECTION_MODE)
        self.fixed_relation = bool(cfg.DYNAMICS.FIXED_RELATION)
        self.symmetric_coupling = bool(cfg.DYNAMICS.SYMMETRIC_COUPLING)
        self.disable_long_timescales = bool(cfg.DYNAMICS.DISABLE_LONG_TIMESCALES)
        self.random_partner = bool(cfg.DYNAMICS.RANDOM_PARTNER)
        self.cf_top_k = int(cfg.COUNTERFACTUAL.TOP_K)
        self.cf_intensity_tolerance = float(cfg.COUNTERFACTUAL.INTENSITY_TOLERANCE)
        self.cf_turn_tolerance = float(cfg.COUNTERFACTUAL.TURN_TOLERANCE)
        self.cf_max_action_cosine = float(cfg.COUNTERFACTUAL.MAX_ACTION_COSINE)
        self.cf_margin = float(cfg.LOSS.COUNTERFACTUAL_MARGIN)
        self.loss_emotion = float(cfg.LOSS.EMOTION)
        self.loss_intensity = float(cfg.LOSS.INTENSITY)
        self.loss_vad = float(cfg.LOSS.VAD)
        self.loss_next = float(cfg.LOSS.NEXT_STATE)
        self.loss_trajectory = float(cfg.LOSS.TRAJECTORY)
        self.loss_correction = float(cfg.LOSS.CORRECTION)
        self.loss_counterfactual = float(cfg.LOSS.COUNTERFACTUAL)
        self.finetune_observation = bool(cfg.TRAIN.FINETUNE_OBSERVATION)

    def train(self, mode: bool = True):
        super().train(mode)
        # The pretrained observer remains a deterministic evidence extractor.
        if not self.finetune_observation:
            self.encoder.eval()
        return self

    def _decode_state(self, state_value: Tensor) -> Dict[str, Tensor]:
        aff = self.state_model.state_to_aff(state_value)
        output = self.decoder(aff)
        output["aff"] = aff
        return output

    def _teacher_forced_states(
        self,
        event: EventObservation,
        batch: Mapping[str, Tensor],
        enable_partner: bool,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        valid = batch["valid_mask"].bool()
        current = self.state_model.initialize(batch["speaker_ids"].long())
        posterior_z = []
        posterior_relation = []
        next_prior_z = []
        correction_errors = []
        for turn in range(valid.shape[1]):
            output = self.state_model.step(
                current,
                event_at(event, turn),
                batch["active_role"][:, turn],
                batch["dt_to_next"][:, turn],
                enable_partner=enable_partner,
                correct=(
                    self.correction_mode == "teacher_forced"
                    or (self.correction_mode == "initial_only" and turn == 0)
                ),
                fixed_relation=self.fixed_relation,
                symmetric_coupling=self.symmetric_coupling,
                disable_long_timescales=self.disable_long_timescales,
            )
            posterior = _where_state(valid[:, turn], output.posterior, current)
            next_state = _where_state(valid[:, turn], output.next_prior, current)
            active_posterior = select_role(posterior.z, batch["active_role"][:, turn])
            reconstructed = self.state_model.state_to_aff(active_posterior)
            correction_errors.append(
                1.0 - F.cosine_similarity(reconstructed, event.aff[:, turn], dim=-1)
            )
            posterior_z.append(posterior.z)
            posterior_relation.append(posterior.relation)
            next_prior_z.append(next_state.z)
            current = next_state
        return (
            torch.stack(posterior_z, dim=1),
            torch.stack(posterior_relation, dim=1),
            torch.stack(next_prior_z, dim=1),
            torch.stack(correction_errors, dim=1),
        )

    def _horizon_loss(
        self,
        horizon: int,
        event: EventObservation,
        posterior_z: Tensor,
        posterior_relation: Tensor,
        batch: Mapping[str, Tensor],
        target_aff: Tensor,
        class_weights: Tensor,
        enable_partner: bool,
    ) -> Tuple[Dict[str, Tensor], Tensor]:
        batch_size, length = batch["valid_mask"].shape
        starts = length - horizon
        if starts <= 0:
            zero = zero_loss(posterior_z)
            return {name: zero for name in ("affect", "emotion", "intensity", "vad")}, zero
        state = DyadicState(
            z=posterior_z[:, :starts].reshape(batch_size * starts, 2, -1),
            relation=posterior_relation[:, :starts].reshape(batch_size * starts, -1),
            speaker_ids=batch["speaker_ids"][:, None, :]
            .expand(-1, starts, -1)
            .reshape(batch_size * starts, 2),
        )
        for offset in range(horizon):
            observation = _flatten_event_range(event, offset, offset + starts)
            role = batch["active_role"][:, offset : offset + starts].reshape(-1)
            dt = batch["dt_to_next"][:, offset : offset + starts].reshape(-1)
            output = self.state_model.step(
                state,
                observation,
                role,
                dt,
                enable_partner=enable_partner,
                correct=False,
                fixed_relation=self.fixed_relation,
                symmetric_coupling=self.symmetric_coupling,
                disable_long_timescales=self.disable_long_timescales,
            )
            valid_step = batch["valid_mask"][:, offset : offset + starts].reshape(-1)
            state = _where_state(valid_step, output.next_prior, state)

        target_role = batch["active_role"][:, horizon:].reshape(-1)
        predicted_state = select_role(state.z, target_role)
        prediction = self._decode_state(predicted_state)
        target_valid = (
            batch["valid_mask"][:, :starts] & batch["valid_mask"][:, horizon:]
        ).reshape(-1)
        target = {
            "aff": target_aff[:, horizon:].reshape(-1, target_aff.shape[-1]),
            "emotion": batch["emotion"][:, horizon:].reshape(-1),
            "intensity": batch["intensity"][:, horizon:].reshape(-1),
            "vad": batch["vad"][:, horizon:].reshape(-1, 3),
            "vad_mask": batch["vad_mask"][:, horizon:].reshape(-1, 3),
        }
        losses = state_prediction_losses(
            prediction,
            target["aff"],
            target["emotion"],
            target["intensity"],
            target["vad"],
            target["vad_mask"],
            target_valid,
            class_weights,
        )
        valid_count = target_valid.sum().to(predicted_state.dtype)
        return losses, valid_count

    def _counterfactual_loss(
        self,
        event: EventObservation,
        posterior_z: Tensor,
        posterior_relation: Tensor,
        batch: Mapping[str, Tensor],
        target_aff: Tensor,
        enable_partner: bool,
    ) -> Tuple[Tensor, Tensor]:
        if not enable_partner:
            zero = zero_loss(posterior_z)
            return zero, zero.detach()
        anchors = []
        batch_size, length = batch["valid_mask"].shape
        # The next actual receiver turn is the causal response target for event t.
        for sample in range(batch_size):
            valid_length = int(batch["valid_mask"][sample].sum().item())
            for turn in range(max(valid_length - 1, 0)):
                receiver = 1 - int(batch["active_role"][sample, turn].item())
                for future in range(turn + 1, valid_length):
                    if int(batch["active_role"][sample, future].item()) == receiver:
                        anchors.append((sample, turn, future, receiver))
                        break
        if len(anchors) < 2:
            zero = zero_loss(posterior_z)
            return zero, zero.detach()

        sample_index = torch.tensor([x[0] for x in anchors], device=posterior_z.device)
        turn_index = torch.tensor([x[1] for x in anchors], device=posterior_z.device)
        future_index = torch.tensor([x[2] for x in anchors], device=posterior_z.device)
        receiver_role = torch.tensor([x[3] for x in anchors], device=posterior_z.device)
        state = DyadicState(
            z=posterior_z[sample_index, turn_index],
            relation=posterior_relation[sample_index, turn_index],
            speaker_ids=batch["speaker_ids"][sample_index],
        )
        observation = EventObservation(
            aff=event.aff[sample_index, turn_index],
            event=event.event[sample_index, turn_index],
            action=event.action[sample_index, turn_index],
            reliability=event.reliability[sample_index, turn_index],
            modality_mask=None,
        )
        matches = match_counterfactuals(
            receiver_event=event.event[sample_index, future_index].detach(),
            sender_action=observation.action.detach(),
            receiver_emotion=batch["emotion"][sample_index, future_index],
            receiver_intensity=batch["intensity"][sample_index, future_index],
            turn_position=batch["turn_position"][sample_index, future_index],
            dialogue_id=batch["dialogue_index"][sample_index],
            top_k=self.cf_top_k,
            intensity_tolerance=self.cf_intensity_tolerance,
            turn_tolerance=self.cf_turn_tolerance,
            max_action_cosine=self.cf_max_action_cosine,
        )
        anchor_valid = matches.valid.any(dim=-1)
        if not anchor_valid.any():
            zero = zero_loss(posterior_z)
            return zero, zero.detach()

        real = self.state_model.step(
            state,
            observation,
            batch["active_role"][sample_index, turn_index],
            batch["dt_to_next"][sample_index, turn_index],
            enable_partner=True,
            correct=False,
            fixed_relation=self.fixed_relation,
            symmetric_coupling=self.symmetric_coupling,
            disable_long_timescales=self.disable_long_timescales,
        )
        real_aff = self.state_model.state_to_aff(select_role(real.next_prior.z, receiver_role))
        response_target = target_aff[sample_index, future_index]
        real_distance = 1.0 - F.cosine_similarity(real_aff, response_target, dim=-1)

        count, candidates = matches.indices.shape
        safe_indices = matches.indices.clamp_min(0)
        candidate_actions = observation.action[safe_indices]
        repeat_state = DyadicState(
            z=state.z[:, None].expand(-1, candidates, -1, -1).reshape(count * candidates, 2, -1),
            relation=state.relation[:, None].expand(-1, candidates, -1).reshape(count * candidates, -1),
            speaker_ids=state.speaker_ids[:, None].expand(-1, candidates, -1).reshape(count * candidates, 2),
        )
        repeated_observation = EventObservation(
            aff=observation.aff[:, None].expand(-1, candidates, -1).reshape(count * candidates, -1),
            event=observation.event[:, None].expand(-1, candidates, -1).reshape(count * candidates, -1),
            action=candidate_actions.reshape(count * candidates, -1),
            reliability=observation.reliability[:, None].expand(-1, candidates, -1).reshape(count * candidates, -1),
            modality_mask=None,
        )
        candidate_output = self.state_model.step(
            repeat_state,
            repeated_observation,
            batch["active_role"][sample_index, turn_index][:, None]
            .expand(-1, candidates)
            .reshape(-1),
            batch["dt_to_next"][sample_index, turn_index][:, None]
            .expand(-1, candidates)
            .reshape(-1),
            enable_partner=True,
            correct=False,
            fixed_relation=self.fixed_relation,
            symmetric_coupling=self.symmetric_coupling,
            disable_long_timescales=self.disable_long_timescales,
        )
        candidate_role = receiver_role[:, None].expand(-1, candidates).reshape(-1)
        candidate_aff = self.state_model.state_to_aff(
            select_role(candidate_output.next_prior.z, candidate_role)
        ).reshape(count, candidates, -1)
        candidate_distance = 1.0 - F.cosine_similarity(
            candidate_aff,
            response_target[:, None].expand_as(candidate_aff),
            dim=-1,
        )
        candidate_distance = candidate_distance.masked_fill(~matches.valid, float("inf"))
        hardest_distance = candidate_distance.min(dim=-1).values
        loss = F.relu(self.cf_margin + real_distance - hardest_distance)
        ranking_accuracy = (real_distance + self.cf_margin < hardest_distance).float()
        return masked_mean(loss, anchor_valid), masked_mean(ranking_accuracy, anchor_valid).detach()

    def forward(
        self,
        batch: Mapping[str, Tensor],
        teacher_aff: Tensor,
        class_weights: Tensor,
        enable_partner: bool,
        use_counterfactual: bool,
    ) -> Dict[str, Tensor]:
        batch_size, length = batch["valid_mask"].shape
        if self.finetune_observation:
            encoder_output = self.encoder(
                flatten_sequence_batch(batch), FULL_AVT_MASK.to(batch["audio"].device)
            )
        else:
            with torch.no_grad():
                encoder_output = self.encoder(
                    flatten_sequence_batch(batch), FULL_AVT_MASK.to(batch["audio"].device)
                )
        event = _reshape_event(encoder_output, batch_size, length)
        event.modality_mask = batch["modality_mask"]
        if enable_partner and self.random_partner and batch_size > 1:
            event.action = event.action.roll(1, dims=0)
        posterior_z, posterior_relation, _, correction_error = self._teacher_forced_states(
            event, batch, enable_partner
        )

        horizon_losses = []
        result: Dict[str, Tensor] = {}
        for horizon in self.horizons:
            losses, count = self._horizon_loss(
                horizon,
                event,
                posterior_z,
                posterior_relation,
                batch,
                teacher_aff,
                class_weights,
                enable_partner,
            )
            weighted = (
                losses["affect"]
                + self.loss_emotion * losses["emotion"]
                + self.loss_intensity * losses["intensity"]
                + self.loss_vad * losses["vad"]
            )
            horizon_losses.append(weighted)
            result[f"h{horizon}"] = weighted.detach()
            result[f"h{horizon}_count"] = count.detach()
        if horizon_losses:
            next_loss = horizon_losses[0]
            trajectory = torch.stack(horizon_losses[1:]).mean() if len(horizon_losses) > 1 else zero_loss(next_loss)
        else:
            next_loss = zero_loss(posterior_z)
            trajectory = zero_loss(posterior_z)
        correction = masked_mean(correction_error, batch["valid_mask"])
        if use_counterfactual:
            counterfactual, ranking_accuracy = self._counterfactual_loss(
                event,
                posterior_z,
                posterior_relation,
                batch,
                teacher_aff,
                enable_partner,
            )
        else:
            counterfactual = zero_loss(posterior_z)
            ranking_accuracy = counterfactual.detach()
        total = (
            self.loss_next * next_loss
            + self.loss_trajectory * trajectory
            + self.loss_correction * correction
            + self.loss_counterfactual * counterfactual
        )
        result.update(
            {
                "total": total,
                "next_state": next_loss.detach(),
                "trajectory": trajectory.detach(),
                "correction": correction.detach(),
                "counterfactual": counterfactual.detach(),
                "cf_ranking_accuracy": ranking_accuracy,
            }
        )
        return result


@torch.no_grad()
def make_teacher_aff(
    teacher: ObservationEncoder,
    batch: Mapping[str, Tensor],
) -> Tensor:
    batch_size, length = batch["valid_mask"].shape
    output = teacher(
        flatten_sequence_batch(batch), FULL_AVT_MASK.to(batch["audio"].device)
    )
    return output.aff[:, 0].reshape(batch_size, length, -1).detach()


def initialize_dynamics_bundle(cfg, num_speakers: int) -> DynamicsTrainingBundle:
    encoder = ObservationEncoder.from_config(cfg)
    observation_path = Path(cfg.TRAIN.OBSERVATION_CHECKPOINT)
    if not observation_path.is_file():
        raise FileNotFoundError(
            "TRAIN.OBSERVATION_CHECKPOINT must point to observation_encoder.pt or best.pt"
        )
    encoder.load_state_dict(load_component_state(observation_path, "encoder"), strict=True)
    encoder.requires_grad_(bool(cfg.TRAIN.FINETUNE_OBSERVATION))
    if not cfg.TRAIN.FINETUNE_OBSERVATION:
        encoder.eval()
    state_model = DyadicEmotionSSM.from_config(cfg, num_speakers)
    decoder = AffectDecoder(cfg.MODEL.OBSERVATION_DIM)
    heads_path = Path(cfg.TRAIN.EMOTION_HEADS_CHECKPOINT)
    if not heads_path.is_file():
        raise FileNotFoundError(
            "TRAIN.EMOTION_HEADS_CHECKPOINT must point to emotion_heads.pt or best.pt"
        )
    heads_state = load_component_state(heads_path, "heads")
    affect_state = {
        name[len("affect.") :]: value
        for name, value in heads_state.items()
        if name.startswith("affect.")
    }
    if not affect_state:
        raise KeyError(f"No affect decoder was found in {heads_path}")
    decoder.load_state_dict(affect_state, strict=True)
    return DynamicsTrainingBundle(encoder, state_model, decoder, cfg)
