from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence

import torch
import torch.nn as nn
from torch import Tensor

from emotion_ssm.schema import (
    DyadicState,
    EventObservation,
    StepOutput,
    replace_role,
    select_role,
)


class PersonalDynamicsParameters(nn.Module):
    """Global baseline/time-scales with small train-speaker residuals."""

    def __init__(
        self,
        state_dim: int,
        num_speakers: int,
        num_timescales: int,
        tau_min: float,
        tau_max: float,
        speaker_delta_scale: float,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.num_timescales = num_timescales
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.speaker_delta_scale = speaker_delta_scale
        self.global_baseline = nn.Parameter(torch.zeros(state_dim))
        initial_tau = torch.logspace(
            torch.log10(torch.tensor(tau_min)),
            torch.log10(torch.tensor(tau_max)),
            steps=num_timescales,
        )
        self.global_log_tau = nn.Parameter(initial_tau.log())
        self.baseline_delta = nn.Embedding(max(num_speakers, 1), state_dim)
        self.tau_delta = nn.Embedding(max(num_speakers, 1), num_timescales)
        nn.init.zeros_(self.baseline_delta.weight)
        nn.init.zeros_(self.tau_delta.weight)

    def forward(self, speaker_ids: Tensor) -> tuple:
        known = speaker_ids >= 0
        safe_ids = speaker_ids.clamp_min(0)
        baseline_delta = self.baseline_delta(safe_ids)
        tau_delta = self.tau_delta(safe_ids)
        mask = known[..., None].to(baseline_delta.dtype)
        baseline = self.global_baseline + self.speaker_delta_scale * torch.tanh(
            baseline_delta
        ) * mask
        log_tau = self.global_log_tau + self.speaker_delta_scale * torch.tanh(
            tau_delta
        ) * mask
        tau_group = log_tau.exp().clamp(self.tau_min, self.tau_max)
        repeat = (self.state_dim + self.num_timescales - 1) // self.num_timescales
        tau = tau_group.repeat_interleave(repeat, dim=-1)[..., : self.state_dim]
        return baseline, tau


class DirectionalInfluence(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        relation_dim: int,
        influence_dim: int,
        channels: int,
    ) -> None:
        super().__init__()
        self.sender = nn.Sequential(
            nn.Linear(state_dim + action_dim, influence_dim),
            nn.GELU(),
            nn.Linear(influence_dim, influence_dim),
        )
        self.q = nn.Linear(influence_dim, channels, bias=False)
        self.gate = nn.Sequential(
            nn.Linear(relation_dim + state_dim, channels), nn.Tanh()
        )
        self.p = nn.Linear(channels, state_dim, bias=False)

    def forward(
        self,
        sender_offset: Tensor,
        action: Tensor,
        receiver_offset: Tensor,
        relation: Tensor,
    ) -> tuple:
        signal = self.sender(torch.cat([sender_offset, action], dim=-1))
        channel_signal = self.q(signal)
        gate = self.gate(torch.cat([relation, receiver_offset], dim=-1))
        influence = self.p(channel_signal * gate)
        return influence, signal, gate


class DyadicEmotionSSM(nn.Module):
    def __init__(
        self,
        state_dim: int = 128,
        observation_dim: int = 128,
        relation_dim: int = 64,
        influence_dim: int = 128,
        influence_channels: int = 32,
        num_speakers: int = 1,
        num_timescales: int = 8,
        tau_min: float = 0.5,
        tau_max: float = 300.0,
        speaker_delta_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.relation_dim = relation_dim
        self.personal = PersonalDynamicsParameters(
            state_dim,
            num_speakers,
            num_timescales,
            tau_min,
            tau_max,
            speaker_delta_scale,
        )
        self.self_input = nn.Sequential(
            nn.Linear(observation_dim * 2, state_dim),
            nn.GELU(),
            nn.Linear(state_dim, state_dim),
        )
        self.direction_ab = DirectionalInfluence(
            state_dim,
            observation_dim,
            relation_dim,
            influence_dim,
            influence_channels,
        )
        self.direction_ba = DirectionalInfluence(
            state_dim,
            observation_dim,
            relation_dim,
            influence_dim,
            influence_channels,
        )
        self.relation_cell = nn.GRUCell(
            state_dim * 2 + observation_dim * 2, relation_dim
        )
        self.state_to_aff = nn.Linear(state_dim, observation_dim)
        self.residual_to_state = nn.Sequential(
            nn.Linear(observation_dim, state_dim),
            nn.Tanh(),
        )
        self.correction_gate = nn.Sequential(
            nn.Linear(state_dim + observation_dim + 6, state_dim),
            nn.Sigmoid(),
        )

    @classmethod
    def from_config(cls, cfg, num_speakers: int) -> "DyadicEmotionSSM":
        return cls(
            state_dim=cfg.MODEL.STATE_DIM,
            observation_dim=cfg.MODEL.OBSERVATION_DIM,
            relation_dim=cfg.MODEL.RELATION_DIM,
            influence_dim=cfg.MODEL.INFLUENCE_DIM,
            influence_channels=cfg.MODEL.INFLUENCE_CHANNELS,
            num_speakers=num_speakers,
            num_timescales=cfg.MODEL.NUM_TIMESCALES,
            tau_min=cfg.MODEL.TAU_MIN,
            tau_max=cfg.MODEL.TAU_MAX,
            speaker_delta_scale=cfg.MODEL.SPEAKER_DELTA_SCALE,
        )

    def initialize(self, speaker_ids: Tensor) -> DyadicState:
        baseline, _ = self.personal(speaker_ids)
        relation = torch.zeros(
            speaker_ids.shape[0],
            self.relation_dim,
            device=speaker_ids.device,
            dtype=baseline.dtype,
        )
        return DyadicState(z=baseline, relation=relation, speaker_ids=speaker_ids)

    def _correct(
        self,
        state: DyadicState,
        observation: EventObservation,
        active_role: Tensor,
        enabled: bool,
    ) -> tuple:
        active_state = select_role(state.z, active_role)
        if not enabled:
            gate = torch.zeros_like(active_state)
            return state, gate
        modality_mask = observation.modality_mask
        if modality_mask is None:
            modality_mask = torch.ones_like(observation.reliability)
        modality_mask = modality_mask.to(observation.reliability.dtype)
        gate = self.correction_gate(
            torch.cat(
                [
                    active_state,
                    observation.aff,
                    observation.reliability,
                    modality_mask,
                ],
                dim=-1,
            )
        )
        evidence_residual = observation.aff - self.state_to_aff(active_state)
        corrected = active_state + gate * self.residual_to_state(evidence_residual)
        return (
            DyadicState(
                z=replace_role(state.z, active_role, corrected),
                relation=state.relation,
                speaker_ids=state.speaker_ids,
            ),
            gate,
        )

    def _directional_influence(
        self,
        offset: Tensor,
        relation: Tensor,
        action: Tensor,
        active_role: Tensor,
        symmetric: bool,
    ) -> tuple:
        receiver_role = 1 - active_role
        sender_offset = select_role(offset, active_role)
        receiver_offset = select_role(offset, receiver_role)
        influence_ab, signal_ab, gate_ab = self.direction_ab(
            sender_offset, action, receiver_offset, relation
        )
        module_ba = self.direction_ab if symmetric else self.direction_ba
        influence_ba, signal_ba, gate_ba = module_ba(
            sender_offset, action, receiver_offset, relation
        )
        choose_ba = active_role.bool()[:, None]
        influence = torch.where(choose_ba, influence_ba, influence_ab)
        signal = torch.where(choose_ba, signal_ba, signal_ab)
        gate = torch.where(choose_ba, gate_ba, gate_ab)
        return influence, signal, gate

    def step(
        self,
        state: DyadicState,
        observation: EventObservation,
        active_role: Tensor,
        dt: Tensor,
        enable_partner: bool = True,
        correct: bool = True,
        fixed_relation: bool = False,
        symmetric_coupling: bool = False,
        disable_long_timescales: bool = False,
    ) -> StepOutput:
        posterior, correction_gate = self._correct(
            state, observation, active_role, correct
        )
        baseline, tau = self.personal(state.speaker_ids)
        if disable_long_timescales:
            tau = tau.clamp_max(tau.median(dim=-1, keepdim=True).values)

        offset = posterior.z - baseline
        decay = torch.exp(-dt[:, None, None].clamp_min(0.0) / tau.clamp_min(1e-6))
        next_offset = offset * decay
        active_offset = select_role(next_offset, active_role)
        stimulus = self.self_input(
            torch.cat([observation.aff, observation.event], dim=-1)
        )
        next_offset = replace_role(
            next_offset, active_role, active_offset + stimulus
        )

        influence = torch.zeros_like(stimulus)
        signal = torch.zeros_like(stimulus)
        influence_gate = torch.zeros(
            len(stimulus),
            self.direction_ab.q.out_features,
            device=stimulus.device,
            dtype=stimulus.dtype,
        )
        if enable_partner:
            influence, signal, influence_gate = self._directional_influence(
                next_offset,
                posterior.relation,
                observation.action,
                active_role,
                symmetric_coupling,
            )
            receiver_role = 1 - active_role
            receiver_offset = select_role(next_offset, receiver_role)
            next_offset = replace_role(
                next_offset, receiver_role, receiver_offset + influence
            )

        relation = posterior.relation
        if not fixed_relation:
            zero_action = torch.zeros_like(observation.action)
            action_a = torch.where(
                (active_role == 0)[:, None], observation.action, zero_action
            )
            action_b = torch.where(
                (active_role == 1)[:, None], observation.action, zero_action
            )
            relation_input = torch.cat(
                [next_offset[:, 0], next_offset[:, 1], action_a, action_b], dim=-1
            )
            relation = self.relation_cell(relation_input, relation)

        next_prior = DyadicState(
            z=baseline + next_offset,
            relation=relation,
            speaker_ids=state.speaker_ids,
        )
        return StepOutput(
            posterior=posterior,
            next_prior=next_prior,
            correction_gate=correction_gate,
            influence=influence,
            baseline=baseline,
            tau=tau,
            auxiliary={
                "stimulus": stimulus,
                "sender_signal": signal,
                "influence_gate": influence_gate,
                "decay": decay,
            },
        )

    def rollout(
        self,
        state: DyadicState,
        observations: Sequence[EventObservation],
        active_roles: Tensor,
        dt: Tensor,
        valid_mask: Optional[Tensor] = None,
        horizons: Optional[Sequence[int]] = None,
        enable_partner: bool = True,
        correction_mode: str = "teacher_forced",
        fixed_relation: bool = False,
        symmetric_coupling: bool = False,
        disable_long_timescales: bool = False,
    ) -> Dict[str, Tensor]:
        if correction_mode not in {"teacher_forced", "initial_only", "none"}:
            raise ValueError("Unknown correction mode")
        if valid_mask is None:
            valid_mask = torch.ones_like(active_roles, dtype=torch.bool)
        posterior_values = []
        posterior_relation_values = []
        prior_values = []
        relation_values = []
        influence_values = []
        current = state
        for turn, observation in enumerate(observations):
            use_correction = correction_mode == "teacher_forced" or (
                correction_mode == "initial_only" and turn == 0
            )
            output = self.step(
                current,
                observation,
                active_roles[:, turn],
                dt[:, turn],
                enable_partner=enable_partner,
                correct=use_correction,
                fixed_relation=fixed_relation,
                symmetric_coupling=symmetric_coupling,
                disable_long_timescales=disable_long_timescales,
            )
            posterior_relation_values.append(output.posterior.relation)
            valid = valid_mask[:, turn, None, None]
            next_z = torch.where(valid, output.next_prior.z, current.z)
            valid_relation = valid_mask[:, turn, None]
            next_relation = torch.where(
                valid_relation, output.next_prior.relation, current.relation
            )
            current = DyadicState(next_z, next_relation, current.speaker_ids)
            posterior_values.append(output.posterior.z)
            prior_values.append(current.z)
            relation_values.append(current.relation)
            influence_values.append(output.influence)
        result = {
            "posterior_z": torch.stack(posterior_values, dim=1),
            "posterior_relation": torch.stack(posterior_relation_values, dim=1),
            "next_prior_z": torch.stack(prior_values, dim=1),
            "relation": torch.stack(relation_values, dim=1),
            "influence": torch.stack(influence_values, dim=1),
            "final_state_z": current.z,
            "final_relation": current.relation,
        }
        if horizons is not None:
            for horizon in horizons:
                if horizon < 1 or horizon > len(prior_values):
                    continue
                result[f"horizon_{horizon}_prior_z"] = result["next_prior_z"][
                    :, horizon - 1
                ]
        return result

    def rollout_candidates(
        self,
        state: DyadicState,
        user_observation: EventObservation,
        candidate_actions: Tensor,
        avatar_role: int = 1,
        dt: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        batch_size, candidates, _ = candidate_actions.shape
        repeat = lambda value: value[:, None].expand(
            (batch_size, candidates) + value.shape[1:]
        ).reshape((batch_size * candidates,) + value.shape[1:])
        repeated_state = DyadicState(
            z=repeat(state.z),
            relation=repeat(state.relation),
            speaker_ids=repeat(state.speaker_ids),
        )
        observation = EventObservation(
            aff=repeat(user_observation.aff),
            event=repeat(user_observation.event),
            action=candidate_actions.reshape(batch_size * candidates, -1),
            reliability=repeat(user_observation.reliability),
            modality_mask=(
                None
                if user_observation.modality_mask is None
                else repeat(user_observation.modality_mask)
            ),
        )
        role = torch.full(
            (batch_size * candidates,), avatar_role, device=state.z.device, dtype=torch.long
        )
        if dt is None:
            dt = torch.ones(batch_size, device=state.z.device)
        output = self.step(
            repeated_state,
            observation,
            role,
            repeat(dt[:, None]).squeeze(-1),
            enable_partner=True,
            correct=False,
        )
        return {
            "z": output.next_prior.z.reshape(batch_size, candidates, 2, -1),
            "relation": output.next_prior.relation.reshape(batch_size, candidates, -1),
            "influence": output.influence.reshape(batch_size, candidates, -1),
        }
