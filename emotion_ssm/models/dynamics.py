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
        self.event_to_delta = nn.Sequential(
            nn.Linear(observation_dim, state_dim),
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
        # The default 128-D state and affect observation therefore share exact
        # coordinates.  Keep learned projections only for small compatibility
        # models that intentionally use different dimensions.
        self.state_to_aff = (
            nn.Identity()
            if state_dim == observation_dim
            else nn.Linear(state_dim, observation_dim)
        )
        self.residual_to_state = (
            nn.Identity()
            if state_dim == observation_dim
            else nn.Sequential(nn.Linear(observation_dim, state_dim), nn.Tanh())
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

    @staticmethod
    def _normalize_dt(dt: Tensor, batch_size: int, device, dtype) -> Tensor:
        """Normalize scalar/[B]/[B,1] intervals to a finite [B] tensor."""
        value = torch.as_tensor(dt, device=device, dtype=dtype)
        if value.ndim == 0:
            value = value.expand(batch_size)
        elif value.ndim == 2 and value.shape[-1] == 1:
            value = value[:, 0]
        elif value.ndim != 1:
            raise ValueError(f"dt must be scalar, [B] or [B,1], got {tuple(value.shape)}")
        if value.shape[0] != batch_size:
            raise ValueError(
                f"dt batch dimension must be {batch_size}, got {value.shape[0]}"
            )
        if not torch.isfinite(value).all():
            raise ValueError("Time intervals must be finite")
        return value

    def decay_only(self, state: DyadicState, dt: Tensor) -> DyadicState:
        """Advance time without correction, self stimulus or partner action."""
        baseline, tau = self.personal(state.speaker_ids)
        interval = self._normalize_dt(
            dt, state.z.shape[0], state.z.device, state.z.dtype
        )
        decay = torch.exp(
            -interval[:, None, None].clamp_min(0.0) / tau.clamp_min(1e-6)
        )
        return DyadicState(
            z=baseline + (state.z - baseline) * decay,
            relation=state.relation,
            speaker_ids=state.speaker_ids,
        )

    def predict_at(self, state, current_time, query_times, future_inputs=None, **options):
        """Predict at explicit seconds; conditional inputs contain no affect correction.

        future_inputs is an optional sequence of (timestamp, two observations).
        Open-loop calls never inspect it. Inputs at the origin are already in
        state and cannot be injected again.
        """
        import math
        queries = [float(t) for t in query_times]
        if any(not math.isfinite(t) or t < current_time for t in queries):
            raise ValueError("Query times must be finite and at/after the state timestamp")
        if queries != sorted(queries):
            raise ValueError("Query times must be sorted")
        inputs = sorted(future_inputs or [], key=lambda item: item[0])
        if any(o.action_duration is None for _, observations in inputs for o in observations):
            raise ValueError("Conditional future actions require explicit durations")
        if any(not math.isfinite(float(t)) or t <= current_time for t, _ in inputs):
            raise ValueError("Conditional inputs must occur strictly after the origin")
        results, cursor, time, current = [], 0, float(current_time), state
        for query in queries:
            while cursor < len(inputs) and inputs[cursor][0] <= query:
                timestamp, observations = inputs[cursor]
                current = self.observe(current, observations, timestamp-time, correct=False, **options)
                time = float(timestamp)
                cursor += 1
            current = self.decay_only(current, query-time)
            time = query
            results.append(current)
        return results

    def correct(
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
        gate = gate * modality_mask.bool().any(-1, keepdim=True).to(gate.dtype)
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

    # Kept for checkpoints and downstream callers that used the private helper.
    def _correct(self, state, observation, active_role, enabled):
        return self.correct(state, observation, active_role, enabled)

    @staticmethod
    def evidence_present(observation):
        if observation.modality_mask is None:
            return torch.ones(observation.aff.shape[0], dtype=torch.bool, device=observation.aff.device)
        return observation.modality_mask.bool().any(-1)

    def event_stimulus(self, observation):
        present = observation.event_present
        if present is None:
            present = (observation.modality_mask[..., 2] if observation.modality_mask is not None
                       else observation.event.abs().sum(-1) > 0)
        if observation.modality_mask is not None:
            present = present.bool() & observation.modality_mask[..., 2].bool()
        return self.event_to_delta(observation.event) * present[..., None].to(observation.event.dtype)

    def observe(self, state, observations, dt, enable_partner=True, correct=True,
                fixed_relation=False, symmetric_coupling=False):
        """Endpoint clock: elapsed decay, current correction, one event injection.

        Both observations refer to the same endpoint. Their action_duration is
        the amount of new observed behavior, not time since an arbitrary call.
        The returned state is at this endpoint, with no future advance.
        """
        if len(observations) != 2:
            raise ValueError("observe requires exactly two role observations")
        dt = self._normalize_dt(dt, len(state.z), state.z.device, state.z.dtype)
        if (dt < 0).any():
            raise ValueError("Observation time cannot run backwards")
        prior = self.decay_only(state, dt)
        baseline, _ = self.personal(state.speaker_ids)
        corrected, available = [], []
        for role, observation in enumerate(observations):
            roles = torch.full((len(state.z),), role, dtype=torch.long, device=state.z.device)
            available.append(self.evidence_present(observation))
            value, _ = self.correct(prior, observation, roles, correct)
            corrected.append(torch.where(available[-1][:, None], value.z[:, role], prior.z[:, role]))
        offset = torch.stack(corrected, 1) - baseline
        offset = offset + torch.stack([self.event_stimulus(o) for o in observations], 1)
        durations = []
        for observation, valid in zip(observations, available):
            duration = dt if observation.action_duration is None else self._normalize_dt(
                observation.action_duration, len(state.z), state.z.device, state.z.dtype)
            if (duration < 0).any():
                raise ValueError("Action duration cannot be negative")
            durations.append(duration * valid.to(dt.dtype))
        to_a, to_b = torch.zeros_like(offset[:, 0]), torch.zeros_like(offset[:, 1])
        if enable_partner:
            to_b = self.direction_ab(offset[:, 0], observations[0].action,
                                     offset[:, 1], prior.relation)[0] * durations[0][:, None]
            direction = self.direction_ab if symmetric_coupling else self.direction_ba
            to_a = direction(offset[:, 1], observations[1].action,
                             offset[:, 0], prior.relation)[0] * durations[1][:, None]
        offset = offset + torch.stack([to_a, to_b], 1)
        relation = prior.relation
        if not fixed_relation and enable_partner:
            actions = [o.action * a[:, None].to(o.action.dtype) for o, a in zip(observations, available)]
            candidate = self.relation_cell(torch.cat([offset[:, 0], offset[:, 1], *actions], -1), relation)
            gate = 1 - torch.exp(-torch.maximum(durations[0], durations[1]))
            relation = relation + gate[:, None] * (candidate - relation)
        return DyadicState(baseline + offset, relation, state.speaker_ids)

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
        # A and B use independent learned maps. For a mixed batch calculate both
        # maps and select by active role so each direction remains causal.
        influence_ab, signal_ab, gate_ab = self.direction_ab(
            sender_offset, action, receiver_offset, relation
        )
        influence_ba, signal_ba, gate_ba = (self.direction_ab if symmetric else self.direction_ba)(
            sender_offset, action, receiver_offset, relation
        )
        choose_ba = active_role.bool()[:, None]
        influence = torch.where(choose_ba, influence_ba, influence_ab)
        signal = torch.where(choose_ba, signal_ba, signal_ab)
        gate = torch.where(choose_ba, gate_ba, gate_ab)
        return influence, signal, gate

    def transition(
        self,
        state: DyadicState,
        observation: EventObservation,
        active_role: Tensor,
        dt: Tensor,
        enable_partner: bool = True,
        fixed_relation: bool = False,
        symmetric_coupling: bool = False,
        disable_long_timescales: bool = False,
    ) -> StepOutput:
        """Advance a prior state without reading affect evidence.

        Affect is deliberately consumed only by :meth:`correct`; this makes an
        open-loop rollout invariant to edits of future affect observations.
        """
        baseline, tau = self.personal(state.speaker_ids)
        dt = self._normalize_dt(dt, state.z.shape[0], state.z.device, state.z.dtype)
        if disable_long_timescales:
            tau = tau.clamp_max(tau.median(dim=-1, keepdim=True).values)
        offset = state.z - baseline
        # Inject the current event and partner action before decay.  The new
        # evidence therefore contributes for the complete interval ``dt``.
        active_offset = select_role(offset, active_role)
        stimulus = self.event_stimulus(observation)
        next_offset = replace_role(offset, active_role, active_offset + stimulus)
        influence = torch.zeros_like(stimulus)
        signal = torch.zeros_like(stimulus)
        influence_gate = torch.zeros(
            len(stimulus), self.direction_ab.q.out_features,
            device=stimulus.device, dtype=stimulus.dtype,
        )
        if enable_partner:
            influence, signal, influence_gate = self._directional_influence(
                next_offset, state.relation, observation.action, active_role, symmetric_coupling
            )
            receiver_role = 1 - active_role
            receiver_offset = select_role(next_offset, receiver_role)
            next_offset = replace_role(next_offset, receiver_role, receiver_offset + influence)
        decay = torch.exp(-dt[:, None, None].clamp_min(0.0) / tau.clamp_min(1e-6))
        next_offset = next_offset * decay
        relation = state.relation
        if not fixed_relation:
            zero_action = torch.zeros_like(observation.action)
            action_a = torch.where((active_role == 0)[:, None], observation.action, zero_action)
            action_b = torch.where((active_role == 1)[:, None], observation.action, zero_action)
            relation_input = torch.cat([next_offset[:, 0], next_offset[:, 1], action_a, action_b], dim=-1)
            relation = self.relation_cell(relation_input, relation)
        next_prior = DyadicState(baseline + next_offset, relation, state.speaker_ids)
        return StepOutput(
            posterior=state,
            next_prior=next_prior,
            correction_gate=torch.zeros_like(stimulus),
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
        posterior, correction_gate = self.correct(
            state, observation, active_role, correct
        )
        output = self.transition(
            posterior, observation, active_role, dt, enable_partner,
            fixed_relation, symmetric_coupling, disable_long_timescales,
        )
        output.posterior = posterior
        output.correction_gate = correction_gate
        return output

    def step_parallel(
        self,
        state: DyadicState,
        observations: Sequence[EventObservation],
        dt: Tensor,
        enable_partner: bool = True,
        correct: bool = True,
        fixed_relation: bool = False,
        symmetric_coupling: bool = False,
        disable_long_timescales: bool = False,
    ) -> StepOutput:
        """Update both roles simultaneously from one common pre-chunk state.

        Both self stimuli and both directional influences are evaluated from
        the same corrected state.  This avoids introducing an artificial A
        then B ordering when a DualTalk chunk contains both audio streams.
        """
        if len(observations) != 2:
            raise ValueError("step_parallel expects observations for roles A and B")
        batch = state.z.shape[0]
        roles = [
            torch.zeros(batch, dtype=torch.long, device=state.z.device),
            torch.ones(batch, dtype=torch.long, device=state.z.device),
        ]
        corrected = []
        gates = []
        for observation, role in zip(observations, roles):
            posterior, gate = self.correct(state, observation, role, correct)
            corrected.append(posterior)
            gates.append(gate)
        corrected_state = DyadicState(
            z=torch.stack([corrected[0].z[:, 0], corrected[1].z[:, 1]], dim=1),
            relation=state.relation,
            speaker_ids=state.speaker_ids,
        )
        baseline, tau = self.personal(corrected_state.speaker_ids)
        dt = self._normalize_dt(
            dt, corrected_state.z.shape[0], corrected_state.z.device, corrected_state.z.dtype
        )
        if disable_long_timescales:
            tau = tau.clamp_max(tau.median(dim=-1, keepdim=True).values)

        offset = corrected_state.z - baseline
        stimulus_a = self.event_stimulus(observations[0])
        stimulus_b = self.event_stimulus(observations[1])
        self_offset = torch.stack(
            [offset[:, 0] + stimulus_a, offset[:, 1] + stimulus_b], dim=1
        )

        influence_to_a = torch.zeros_like(stimulus_a)
        influence_to_b = torch.zeros_like(stimulus_b)
        signal_ab = torch.zeros_like(stimulus_a)
        signal_ba = torch.zeros_like(stimulus_b)
        gate_ab = torch.zeros(
            batch,
            self.direction_ab.q.out_features,
            device=stimulus_a.device,
            dtype=stimulus_a.dtype,
        )
        gate_ba = gate_ab.clone()
        if enable_partner:
            direction_ba = self.direction_ab if symmetric_coupling else self.direction_ba
            influence_to_b, signal_ab, gate_ab = self.direction_ab(
                self_offset[:, 0],
                observations[0].action,
                self_offset[:, 1],
                corrected_state.relation,
            )
            influence_to_a, signal_ba, gate_ba = direction_ba(
                self_offset[:, 1],
                observations[1].action,
                self_offset[:, 0],
                corrected_state.relation,
            )

        next_offset = torch.stack(
            [self_offset[:, 0] + influence_to_a, self_offset[:, 1] + influence_to_b],
            dim=1,
        )
        decay = torch.exp(-dt[:, None, None].clamp_min(0.0) / tau.clamp_min(1e-6))
        next_offset = next_offset * decay
        if fixed_relation:
            relation = corrected_state.relation
        else:
            relation_input = torch.cat(
                [
                    next_offset[:, 0],
                    next_offset[:, 1],
                    observations[0].action,
                    observations[1].action,
                ],
                dim=-1,
            )
            relation = self.relation_cell(relation_input, corrected_state.relation)
        merged = DyadicState(baseline + next_offset, relation, state.speaker_ids)
        influence = (influence_to_a + influence_to_b) * 0.5
        return StepOutput(
            posterior=corrected_state,
            next_prior=merged,
            correction_gate=torch.stack(gates, dim=1).mean(dim=1),
            influence=influence,
            baseline=baseline,
            tau=tau,
            auxiliary={
                "decay": decay,
                "parallel": torch.ones(batch, device=state.z.device),
                "influence_ab": influence_to_b,
                "influence_ba": influence_to_a,
                "sender_signal_ab": signal_ab,
                "sender_signal_ba": signal_ba,
                "influence_gate_ab": gate_ab,
                "influence_gate_ba": gate_ba,
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
            valid = valid_mask[:, turn, None, None]
            posterior = DyadicState(
                z=torch.where(valid, output.posterior.z, current.z),
                relation=torch.where(
                    valid_mask[:, turn, None], output.posterior.relation, current.relation
                ),
                speaker_ids=current.speaker_ids,
            )
            posterior_relation_values.append(posterior.relation)
            next_z = torch.where(valid, output.next_prior.z, current.z)
            valid_relation = valid_mask[:, turn, None]
            next_relation = torch.where(
                valid_relation, output.next_prior.relation, current.relation
            )
            current = DyadicState(next_z, next_relation, current.speaker_ids)
            posterior_values.append(posterior.z)
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
