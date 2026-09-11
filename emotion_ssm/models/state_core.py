"""Version 3: one differentiable clock for memory, coupling and forecasts.

The v2 dynamics module is intentionally untouched.  Rates here are measured in
seconds; A/B are storage positions, not independently parameterised people.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any, Mapping, Optional, Sequence

import torch
from torch import Tensor, nn

from emotion_ssm.schema import EventObservation


@dataclass
class StateObservation(EventObservation):
    """Explicit v3 evidence contract, compatible with the v2 observation fields.

    ``fresh_observation`` is [B,3] (or [B]) and describes newly observed evidence,
    rather than the availability of old context.  ``action_present`` explicitly
    declares new language/nonverbal behaviour.  A zero action vector alone is
    not a missing-data marker.  Event IDs, when supplied, are monotonic per role.
    """

    fresh_observation: Optional[Tensor] = None
    action_present: Optional[Tensor] = None
    event_id: Optional[Tensor] = None

    def index(self, index) -> "StateObservation":
        return type(self)(**{
            field.name: (None if getattr(self, field.name) is None
                         else getattr(self, field.name)[index])
            for field in fields(self)
        })


@dataclass
class EmotionMemory:
    """Fast/slow offsets share the observer's affect coordinates.

    relation[:,0] is A->B and relation[:,1] is B->A.  All floating tensors remain
    attached until the trainer explicitly calls detach at a TBPTT boundary.
    """

    fast: Tensor
    slow: Tensor
    relation: Tensor
    baseline: Tensor
    elapsed: Tensor
    last_event_ids: Tensor

    FORMAT_VERSION = 3

    @property
    def z(self) -> Tensor:
        return self.baseline + self.fast + self.slow

    def _map(self, function) -> "EmotionMemory":
        return type(self)(**{field.name: function(getattr(self, field.name))
                             for field in fields(self)})

    def detach(self) -> "EmotionMemory":
        return self._map(lambda value: value.detach())

    def clone(self) -> "EmotionMemory":
        return self._map(lambda value: value.clone())

    def to(self, *args, **kwargs) -> "EmotionMemory":
        values = {field.name: getattr(self, field.name).to(*args, **kwargs)
                  for field in fields(self) if field.name != "last_event_ids"}
        values["last_event_ids"] = self.last_event_ids.to(device=values["fast"].device)
        return type(self)(**values)

    def role_swap(self) -> "EmotionMemory":
        return type(self)(self.fast.flip(1), self.slow.flip(1), self.relation.flip(1),
                          self.baseline.flip(1), self.elapsed.clone(),
                          self.last_event_ids.flip(1))

    def state_dict(self) -> dict[str, Any]:
        return {"format_version": self.FORMAT_VERSION,
                **{field.name: getattr(self, field.name) for field in fields(self)}}

    serialize = state_dict

    @classmethod
    def from_state_dict(cls, value: Mapping[str, Any]) -> "EmotionMemory":
        if value.get("format_version") != cls.FORMAT_VERSION:
            raise ValueError("An explicit v3 memory checkpoint is required")
        result = cls(**{field.name: value[field.name] for field in fields(cls)})
        if (result.fast.ndim != 3 or result.fast.shape[1] != 2 or
                result.slow.shape != result.fast.shape or
                result.baseline.shape != result.fast.shape or
                result.relation.shape[:2] != result.fast.shape[:2] or
                result.elapsed.shape != result.fast.shape[:1] or
                result.last_event_ids.shape != result.fast.shape[:2]):
            raise ValueError("Invalid v3 memory tensor shapes")
        return result

    deserialize = from_state_dict


@dataclass(frozen=True)
class KnownFutureInput:
    """A conditional input ending ``seconds`` after the forecast origin.

    planned inputs must already be available at the origin.  Reading recorded
    future behaviour is possible only with the explicit oracle_conditional
    protocol.  Neither protocol enables future affect correction.
    """

    seconds: float
    observations: Sequence[EventObservation]
    protocol: str = "planned"
    available_at: float = 0.0


class UnifiedEmotionStateCore(nn.Module):
    """Same-coordinate fast/slow state with explicitly versioned autonomous flow.

    The legacy flow has independent 2x2 blocks with negative symmetric parts.
    The adaptive flow recomputes cross-coordinate mixing, damping and directed
    feedback from both roles' current states; feedback can increase energy.
    The optional affine revision adds a bounded shared continuous drive.
    All revisions keep new event/action inputs separate from autonomous flow.
    """

    FORMAT_VERSION = 3

    def __init__(self, observation_dim: int = 128, relation_dim: int = 32,
                 hidden_dim: int = 128, fast_tau: float = 2.0,
                 slow_tau: float = 120.0, relation_tau: float = 90.0,
                 max_integration_step: float = 0.5,
                 fast_correction_rate: float = 1.0,
                 slow_correction_rate: float = 0.002,
                 max_slow_correction_rate: float = 1.0 / 60.0,
                 max_coupling: float = 1.0, max_event: float = 1.0,
                 max_autonomous_rotation: float = 0.05,
                 initial_autonomous_rotation: float = 0.002,
                 flow_kind: str = "legacy_linear_v1", flow_rank: int = 8,
                 max_cross_rate: float = .05, max_feedback: float = 2.,
                 affine_max_offset: float = .5) -> None:
        super().__init__()
        self.observation_dim = int(observation_dim)
        self.state_dim = self.observation_dim
        self.relation_dim = int(relation_dim)
        self.context_dim = 6 * self.observation_dim + 2 * self.relation_dim
        if min(observation_dim, relation_dim, hidden_dim) < 1:
            raise ValueError("State dimensions must be positive")
        if min(fast_tau, slow_tau, relation_tau, max_integration_step,
               max_slow_correction_rate, max_coupling, max_event,
               max_autonomous_rotation) <= 0:
            raise ValueError("State rates, bounds and time scales must be positive")
        if not 0 < slow_correction_rate < max_slow_correction_rate:
            raise ValueError("Initial slow correction must be below its rate bound")
        if not 0 < fast_correction_rate < 8:
            raise ValueError("Initial fast correction rate must be in (0,8)")
        if not abs(initial_autonomous_rotation) < max_autonomous_rotation:
            raise ValueError("Initial rotation exceeds the autonomous bound")
        self._construction = dict(
            observation_dim=observation_dim, relation_dim=relation_dim,
            hidden_dim=hidden_dim, fast_tau=fast_tau, slow_tau=slow_tau,
            relation_tau=relation_tau, max_integration_step=max_integration_step,
            fast_correction_rate=fast_correction_rate,
            slow_correction_rate=slow_correction_rate,
            max_slow_correction_rate=max_slow_correction_rate,
            max_coupling=max_coupling, max_event=max_event,
            max_autonomous_rotation=max_autonomous_rotation,
            initial_autonomous_rotation=initial_autonomous_rotation)
        self.max_integration_step = float(max_integration_step)
        self.max_slow_correction_rate = float(max_slow_correction_rate)
        self.max_coupling = float(max_coupling)
        self.max_event = float(max_event)
        self.max_autonomous_rotation = float(max_autonomous_rotation)
        self.baseline = nn.Parameter(torch.zeros(observation_dim))
        self.log_fast_tau = nn.Parameter(torch.full((observation_dim,), math.log(fast_tau)))
        self.log_slow_tau = nn.Parameter(torch.full((observation_dim,), math.log(slow_tau)))
        self.log_relation_tau = nn.Parameter(torch.full((relation_dim,), math.log(relation_tau)))
        self.rotation = nn.Parameter(torch.full((observation_dim,), math.atanh(
            initial_autonomous_rotation / max_autonomous_rotation)))
        logit = lambda x: math.log(x / (1.0 - x))
        self.fast_correction_logits = nn.Parameter(torch.full(
            (observation_dim,), logit(fast_correction_rate / 8.0)))
        self.slow_correction_logits = nn.Parameter(torch.full(
            (observation_dim,), logit(slow_correction_rate / max_slow_correction_rate)))
        self.coupling_strength = nn.Parameter(torch.full((2, observation_dim), -2.0))
        self.event_strength = nn.Parameter(torch.full((2, observation_dim), -2.0))
        # Bias-free bounded maps have a genuine zero-input neutral reference.
        ordered_dim = 3 * observation_dim + relation_dim
        self.influence = nn.Sequential(nn.Linear(ordered_dim, hidden_dim, bias=False),
                                       nn.Tanh(), nn.Linear(hidden_dim, observation_dim, bias=False),
                                       nn.Tanh())
        self.relation_drive = nn.Sequential(nn.Linear(ordered_dim, hidden_dim, bias=False),
                                            nn.Tanh(), nn.Linear(hidden_dim, relation_dim, bias=False),
                                            nn.Tanh())
        self.event_projection = nn.Sequential(nn.Linear(observation_dim, hidden_dim, bias=False),
                                              nn.Tanh(), nn.Linear(hidden_dim, observation_dim, bias=False),
                                              nn.Tanh())
        if flow_kind not in ("legacy_linear_v1", "adaptive_dyadic_v1", "adaptive_affine_dyadic_v2"):
            raise ValueError("Unknown autonomous flow; checkpoint construction is authoritative")
        self.flow_kind = flow_kind
        self.adaptive_flow = None
        if flow_kind in ("adaptive_dyadic_v1", "adaptive_affine_dyadic_v2"):
            from emotion_ssm.models.adaptive_flow import AdaptiveDyadicFlow
            if max_integration_step > .5:
                raise ValueError("Adaptive integration steps must not exceed 0.5 seconds")
            self._construction.update(flow_kind=flow_kind, flow_rank=flow_rank,
                                      max_cross_rate=max_cross_rate, max_feedback=max_feedback)
            self.adaptive_flow = AdaptiveDyadicFlow(observation_dim, relation_dim, hidden_dim,
                flow_rank, max_cross_rate, max_feedback,
                affine_max_offset if flow_kind == 'adaptive_affine_dyadic_v2' else None)
            if flow_kind == 'adaptive_affine_dyadic_v2':
                self._construction['affine_max_offset'] = affine_max_offset

    def get_config(self) -> dict[str, Any]:
        return dict(self._construction)

    construction_config = get_config

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "UnifiedEmotionStateCore":
        return cls(**dict(config))

    def initialize(self, batch_size: int, device=None, dtype=None) -> EmotionMemory:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        device = self.baseline.device if device is None else device
        dtype = self.baseline.dtype if dtype is None else dtype
        # AMP may return half/bfloat16 observations. Persistent memories must
        # still resolve small per-second slow updates; half precision can round
        # a 1/1800 decay at unit magnitude back to one indefinitely.
        dtype = torch.float64 if dtype == torch.float64 else torch.float32
        zeros = torch.zeros(batch_size, 2, self.observation_dim, device=device, dtype=dtype)
        baseline = self.baseline.to(device=device, dtype=dtype)[None, None].expand_as(zeros)
        return EmotionMemory(zeros, zeros.clone(),
                             torch.zeros(batch_size, 2, self.relation_dim, device=device, dtype=dtype),
                             baseline, torch.zeros(batch_size, device=device, dtype=dtype),
                             torch.full((batch_size, 2), -1, device=device, dtype=torch.long))

    def rebind_trainable_baseline(self, memory: EmotionMemory) -> EmotionMemory:
        """Attach the current baseline parameter at the start of a training graph.

        Call after detaching TBPTT history or restoring a training memory, with
        gradient recording enabled. Detach removes the baseline's gradient path;
        checkpoint restoration also removes its connection to parameter storage.
        Rebinding gives both cases the current parameter value and gradient path.
        All history tensors retain their values and existing gradient status.

        This is explicit training behavior: inference may carry a persisted or
        caller-supplied baseline, which advance/forecast must not replace.
        """
        baseline = self.baseline.to(device=memory.fast.device, dtype=memory.fast.dtype)
        baseline = baseline[None, None].expand_as(memory.fast)
        return EmotionMemory(memory.fast, memory.slow, memory.relation, baseline,
                             memory.elapsed, memory.last_event_ids)

    @staticmethod
    def _persistent_precision(state: EmotionMemory) -> EmotionMemory:
        """Repair external low-precision memories without detaching their graph."""
        dtype = torch.float64 if state.fast.dtype == torch.float64 else torch.float32
        floating = (state.fast, state.slow, state.relation, state.baseline, state.elapsed)
        if any(value.dtype != dtype for value in floating):
            state = state.to(dtype=dtype)
        return state

    @staticmethod
    def _interval(value, state: EmotionMemory, name="dt") -> Tensor:
        value = torch.as_tensor(value, device=state.fast.device, dtype=state.fast.dtype)
        if value.ndim == 0:
            value = value.expand(len(state.fast))
        elif value.ndim == 2 and value.shape[1] == 1:
            value = value[:, 0]
        if value.shape != (len(state.fast),) or not torch.isfinite(value).all() or (value < 0).any():
            raise ValueError(f"{name} must be finite, non-negative and scalar/[B]/[B,1]")
        return value

    def rates(self) -> tuple[Tensor, Tensor, Tensor]:
        # Distinct ranges prevent the learned slow decay from becoming a second
        # fast channel.  Updating slow memory still receives task gradients.
        return (self.log_fast_tau.exp().clamp(0.25, 16.0).reciprocal(),
                self.log_slow_tau.exp().clamp(30.0, 1800.0).reciprocal(),
                self.log_relation_tau.exp().clamp(10.0, 3600.0).reciprocal())

    def autonomous_matrix(self) -> Tensor:
        if self.adaptive_flow is not None:
            raise ValueError("Adaptive dynamics have a state-dependent field; use autonomous_derivative(state)")
        fast, slow, _ = self.rates()
        omega = self.max_autonomous_rotation * self.rotation.tanh()
        return torch.stack([torch.stack([-fast, omega], -1),
                            torch.stack([-omega, -slow], -1)], -2)

    def affect(self, state: EmotionMemory) -> Tensor:
        return state.z

    def autonomous_derivative(self, state: EmotionMemory, enable_partner=True):
        state = self._persistent_precision(state)
        if self.adaptive_flow is None:
            fast, slow, relation = self.rates()
            omega = self.max_autonomous_rotation*self.rotation.tanh()
            return (-fast*state.fast+omega*state.slow,
                    -slow*state.slow-omega*state.fast, -relation*state.relation)
        with torch.autocast(device_type=state.fast.device.type, enabled=False):
            return self.adaptive_flow.derivative(state, self.rates(),
                self.max_autonomous_rotation*self.rotation.tanh(), enable_partner)

    def decay_only(self, state: EmotionMemory, dt) -> EmotionMemory:
        state = self._persistent_precision(state)
        dt = self._interval(dt, state)
        fast, slow, relation = self.rates()
        return EmotionMemory(state.fast * torch.exp(-dt[:, None, None] * fast),
                             state.slow * torch.exp(-dt[:, None, None] * slow),
                             state.relation * torch.exp(-dt[:, None, None] * relation),
                             state.baseline, state.elapsed + dt, state.last_event_ids)

    def _propagate(self, state: EmotionMemory, dt: Tensor,
                  force: Optional[Tensor] = None,
                  relation_force: Optional[Tensor] = None,
                  enable_partner: bool = True) -> EmotionMemory:
        state = self._persistent_precision(state)
        dt = self._interval(dt, state)
        if self.adaptive_flow is not None:
            with torch.autocast(device_type=state.fast.device.type, enabled=False):
                return self.adaptive_flow.propagate(state, dt, self.rates(),
                    self.max_autonomous_rotation*self.rotation.tanh(), self.max_integration_step,
                    enable_partner=enable_partner, force=force, relation_force=relation_force)
        # Matrix exponential is evaluated in float32 under AMP.  Its 2x2 blocks
        # give an exact autonomous semigroup, including irregular query times.
        work_dtype = torch.float64 if state.fast.dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=state.fast.device.type, enabled=False):
            matrix = self.autonomous_matrix().to(work_dtype)
            transition = torch.matrix_exp(matrix[None] * dt.to(work_dtype)[:, None, None, None])
            old = torch.stack([state.fast, state.slow], -1).to(work_dtype)
            value = torch.einsum("bdij,brdj->brdi", transition, old)
            if force is not None:
                eye = torch.eye(2, device=matrix.device, dtype=work_dtype)
                integral = torch.linalg.solve(matrix[None], transition - eye)
                value = value + torch.einsum("bdij,brdj->brdi", integral, force.to(work_dtype))
            relation_rate = self.rates()[2].to(work_dtype)
            relation_decay = torch.exp(-dt.to(work_dtype)[:, None, None] * relation_rate)
            relation = state.relation.to(work_dtype) * relation_decay
            if relation_force is not None:
                relation = relation + (1 - relation_decay) * relation_force.to(work_dtype)
        return EmotionMemory(value[..., 0].to(state.fast.dtype), value[..., 1].to(state.slow.dtype),
                             relation.to(state.relation.dtype), state.baseline,
                             state.elapsed + dt, state.last_event_ids)

    def _metadata(self, observation: EventObservation, state: EmotionMemory,
                  dt: Tensor, correct: bool) -> dict[str, Tensor]:
        batch = len(state.fast)
        device = state.fast.device
        mask = observation.modality_mask
        if mask is None:
            mask = torch.ones(batch, 3, device=device, dtype=torch.bool)
        if mask.shape != (batch, 3):
            raise ValueError("modality_mask must have shape [B,3]")
        mask = mask.to(device=device, dtype=torch.bool)
        event_present = observation.event_present
        event_present = (torch.zeros(batch, device=device, dtype=torch.bool) if event_present is None
                         else event_present.to(device=device, dtype=torch.bool))
        event_present = event_present & mask[:, 2]
        if event_present.shape != (batch,):
            raise ValueError("event_present must have shape [B]")
        fresh = getattr(observation, "fresh_observation", None)
        if fresh is None:
            # Compatibility is conservative: old stored text is not new evidence.
            fresh = mask.clone()
            fresh[:, 2] &= event_present
        fresh = fresh.to(device=device, dtype=state.fast.dtype)
        if fresh.shape == (batch,):
            fresh = fresh[:, None].expand(-1, 3)
        if fresh.shape != (batch, 3) or not torch.isfinite(fresh).all() or ((fresh < 0) | (fresh > 1)).any():
            raise ValueError("fresh_observation must be [B,3] or [B] fractions in [0,1]")
        reliability = observation.reliability.to(device=device, dtype=state.fast.dtype)
        if reliability.shape != (batch, 3) or not torch.isfinite(reliability).all():
            raise ValueError("reliability must be finite [B,3]")
        evidence = (fresh * mask * reliability.clamp(0, 1)).amax(-1)
        action_present = getattr(observation, "action_present", None)
        # An absent v3 declaration means no external action, never old context.
        action_present = (torch.zeros(batch, device=device, dtype=torch.bool) if action_present is None
                          else action_present.to(device=device, dtype=torch.bool))
        if action_present.shape != (batch,):
            raise ValueError("action_present must have shape [B]")
        action_present = action_present & mask.any(-1)
        duration = (torch.zeros_like(dt) if observation.action_duration is None else
                    self._interval(observation.action_duration, state, "action_duration"))
        duration = duration * action_present
        if (duration > dt + 1e-5).any():
            raise ValueError("New action duration cannot exceed the elapsed packet interval")
        for name, present in [("action", action_present), ("event", event_present)]:
            value = getattr(observation, name)
            if value.shape != (batch, self.observation_dim):
                raise ValueError(f"{name} must have shape [B,observation_dim]")
            if not torch.isfinite(value[present]).all():
                raise ValueError(f"Present {name} features must be finite")
        if correct:
            if observation.aff.shape != (batch, self.observation_dim):
                raise ValueError("aff must have shape [B,observation_dim]")
            if not torch.isfinite(observation.aff[evidence > 0]).all():
                raise ValueError("Fresh affect evidence must be finite")
        event_ids = getattr(observation, "event_id", None)
        if event_ids is not None:
            event_ids = event_ids.to(device=device, dtype=torch.long)
            if event_ids.shape != (batch,) or (event_ids[event_present] < 0).any():
                raise ValueError("Present event IDs must be non-negative [B] integers")
        return dict(mask=mask, event_present=event_present, evidence=evidence,
                    action_present=action_present, duration=duration, event_ids=event_ids)

    def _continuous_forcing(self, state: EmotionMemory, observations, metadata,
                            occupied_fraction: Sequence[Tensor]) -> tuple[Tensor, Tensor]:
        offset = state.fast + state.slow
        force = torch.zeros(*state.fast.shape, 2, device=state.fast.device, dtype=state.fast.dtype)
        directional = []
        fast_rate, slow_rate, _ = self.rates()
        strength = self.max_coupling * self.coupling_strength.sigmoid()
        for sender in (0, 1):
            receiver = 1 - sender
            action = torch.where(metadata[sender]["action_present"][:, None],
                                 observations[sender].action, torch.zeros_like(observations[sender].action))
            inputs = torch.cat([offset[:, sender], offset[:, receiver], action,
                                state.relation[:, sender]], -1).to(self.influence[0].weight.dtype)
            fraction = occupied_fraction[sender][:, None]
            influence = self.influence(inputs) * fraction
            role_force = torch.stack([fast_rate * strength[0] * influence,
                                      slow_rate * strength[1] * influence], -1)
            # Stack rather than modifying state or detaching the sender path.
            force = force + torch.stack([role_force if receiver == r else torch.zeros_like(role_force)
                                         for r in (0, 1)], 1)
            directional.append(self.relation_drive(inputs) * fraction)
        return force, torch.stack(directional, 1)

    def advance(self, state: EmotionMemory, observations: Sequence[EventObservation], dt,
                enable_partner: bool = True, correct: bool = True,
                diagnostics: Optional[dict] = None) -> EmotionMemory:
        """Advance elapsed time, then inject each new event and correct its endpoint.

        Continuous behaviour is integrated only over its explicit new duration.
        Old context can inform a newly encoded action, but does not declare one.
        """
        if len(observations) != 2:
            raise ValueError("advance requires one observation for each role")
        state = self._persistent_precision(state)
        dt = self._interval(dt, state)
        metadata = [self._metadata(o, state, dt, correct) for o in observations]
        if diagnostics is not None and diagnostics.get('include_autonomous_prior', False):
            diagnostics['autonomous_prior'] = self._propagate(state, dt, enable_partner=enable_partner).z
        current = state
        active_actions = enable_partner and any(bool((m["duration"] > 0).any()) for m in metadata)
        if not active_actions:
            current = self._propagate(current, dt, enable_partner=enable_partner)
        else:
            steps = max(1, math.ceil(float(dt.max().detach()) / self.max_integration_step))
            elapsed = torch.zeros_like(dt)
            for _ in range(steps):
                step = (dt - elapsed).clamp(0, self.max_integration_step)
                fractions = []
                for m in metadata:
                    occupied = ((elapsed + step - (dt - m["duration"])).clamp_min(0)
                                - (elapsed - (dt - m["duration"])).clamp_min(0)).clamp_min(0)
                    fractions.append(occupied / step.clamp_min(1e-8))
                force, relation_force = self._continuous_forcing(current, observations, metadata, fractions)
                current = self._propagate(current, step, force, relation_force, enable_partner=enable_partner)
                elapsed = elapsed + step
        fast, slow, ids = [], [], []
        fast_rate = 8.0 * self.fast_correction_logits.sigmoid()
        slow_rate = self.max_slow_correction_rate * self.slow_correction_logits.sigmoid()
        total_rate = fast_rate + slow_rate
        event_strength = self.max_event * self.event_strength.sigmoid()
        before_correction, correction_amounts, evidence_values = [], [], []
        for role, (observation, m) in enumerate(zip(observations, metadata)):
            present = m["event_present"]
            event_ids = m["event_ids"]
            if event_ids is not None:
                present = present & (event_ids > state.last_event_ids[:, role])
            event_value = torch.where(present[:, None], observation.event, torch.zeros_like(observation.event))
            event_delta = self.event_projection(event_value.to(self.event_projection[0].weight.dtype))
            event_delta = event_delta.to(current.fast.dtype) * present[:, None]
            role_fast = current.fast[:, role] + event_strength[0] * event_delta
            role_slow = current.slow[:, role] + event_strength[1] * event_delta
            if diagnostics is not None:
                before_correction.append(current.baseline[:, role] + role_fast + role_slow)
                evidence_values.append(m["evidence"])
            amount = torch.zeros_like(role_fast)
            if correct:
                innovation = torch.where(m["evidence"][:, None] > 0,
                                         observation.aff - current.baseline[:, role] - role_fast - role_slow,
                                         torch.zeros_like(role_fast))
                amount = -torch.expm1(-dt[:, None] * m["evidence"][:, None] * total_rate)
                role_fast = role_fast + amount * (fast_rate / total_rate) * innovation
                role_slow = role_slow + amount * (slow_rate / total_rate) * innovation
            if diagnostics is not None:
                correction_amounts.append(amount)
            fast.append(role_fast)
            slow.append(role_slow)
            ids.append(state.last_event_ids[:, role] if event_ids is None else
                       torch.where(present, event_ids, state.last_event_ids[:, role]))
        if diagnostics is not None:
            diagnostics.update(input_conditioned_prior=torch.stack(before_correction, 1),
                               correction_gain=torch.stack(correction_amounts, 1),
                               evidence=torch.stack(evidence_values, 1))
        return EmotionMemory(torch.stack(fast, 1), torch.stack(slow, 1), current.relation,
                             current.baseline, current.elapsed, torch.stack(ids, 1))

    def forecast(self, state: EmotionMemory, query_seconds: Sequence[float],
                 known_inputs: Optional[Sequence[KnownFutureInput | Mapping[str, Any]]] = None,
                 enable_partner: bool = True) -> list[EmotionMemory]:
        """Query relative seconds without reading future affect evidence.

        Queries do not change the integration path: every result is computed
        from the origin and declared input endpoints, not from other queries.
        """
        state = self._persistent_precision(state)
        queries = [float(t) for t in query_seconds]
        if any(not math.isfinite(t) or t < 0 for t in queries):
            raise ValueError("Forecast query seconds must be finite and non-negative")
        if queries != sorted(queries):
            raise ValueError("Forecast query seconds must be sorted")
        declared = []
        for value in known_inputs or []:
            item = KnownFutureInput(**dict(value)) if isinstance(value, Mapping) else value
            if not isinstance(item, KnownFutureInput):
                raise ValueError("Conditional inputs require an explicit KnownFutureInput protocol")
            if item.protocol not in {"planned", "oracle_conditional"}:
                raise ValueError("Unknown future-input protocol")
            if (not math.isfinite(float(item.seconds)) or item.seconds <= 0 or
                    not math.isfinite(float(item.available_at)) or
                    (item.protocol == "planned" and item.available_at > 0)):
                raise ValueError("Planned inputs must be available at the origin and occur later")
            declared.append(item)
        declared.sort(key=lambda item: item.seconds)
        if len({item.seconds for item in declared}) != len(declared):
            raise ValueError("Combine simultaneous role inputs into one declared packet")
        if self.adaptive_flow is not None and not declared:
            # Intermediate queries must not become integration boundaries. Keep
            # an origin-anchored grid; fractional queries are read-only branches.
            result, anchor, seconds = [], state, 0.
            step = self.max_integration_step
            optimized = getattr(self, '_execution_optimized', False)
            if optimized:
                # The Python query schedule above is already validated. Rates
                # are constant within this forecast (weights do not update).
                rates = self.rates()
                omega = self.max_autonomous_rotation*self.rotation.tanh()
                def scheduled(current, interval):
                    dt = torch.full_like(current.elapsed, interval)
                    return self.adaptive_flow.propagate(current, dt, rates, omega, step,
                        enable_partner=enable_partner, known_steps=math.ceil(interval/step))
            for query in queries:
                while seconds+step <= query+1e-12:
                    anchor = (scheduled(anchor, step) if optimized else
                              self._propagate(anchor, self._interval(step, anchor),
                                              enable_partner=enable_partner))
                    seconds += step
                interval = max(0., query-seconds)
                result.append(scheduled(anchor, interval) if optimized else
                              self._propagate(anchor, self._interval(interval, anchor),
                                              enable_partner=enable_partner))
            return result
        result = []
        current, origin, cursor = state, 0.0, 0
        for query in queries:
            while cursor < len(declared) and declared[cursor].seconds <= query:
                item = declared[cursor]
                current = self.advance(current, item.observations, item.seconds - origin,
                                       enable_partner=enable_partner, correct=False)
                origin = float(item.seconds)
                cursor += 1
            result.append(self._propagate(current, self._interval(query - origin, current),
                                          enable_partner=enable_partner))
        return result

    def configure_execution(self, mode='reference'):
        """Execution-only choice; no model construction or weight keys change."""
        if mode not in {'reference', 'optimized', 'compiled'}:
            raise ValueError('Unknown dynamics execution mode')
        self._execution_optimized = mode != 'reference'
        if mode == 'compiled':
            # Grad/no-grad, scalar/batched states and the four phase parameter
            # partitions are finite, legitimate variants of the same callable.
            torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)
        if self.adaptive_flow is not None:
            self.adaptive_flow.configure_execution(compile_midpoint=mode == 'compiled')

    def context(self, state: EmotionMemory, observations: Sequence[EventObservation],
                variant: str = "dyadic") -> Tensor:
        """Fixed-width FiLM input: both current affects, both fast/slow, directions.

        self and dyadic expose the same observation inputs.  Their difference is
        whether advance enables coupling; self additionally hides relation.
        The trainer must call advance(enable_partner=False) for the self ablation.
        """
        if variant not in {"none", "affect", "self", "dyadic"}:
            raise ValueError("Unknown emotion condition variant")
        if len(observations) != 2:
            raise ValueError("context requires both role observations")
        current = []
        for observation in observations:
            mask = observation.modality_mask
            present = (torch.ones(len(state.fast), device=state.fast.device, dtype=torch.bool)
                       if mask is None else mask.bool().any(-1))
            current.append(torch.where(present[:, None], observation.aff, torch.zeros_like(observation.aff)))
        affect = torch.stack(current, 1)
        fast, slow, relation = state.fast, state.slow, state.relation
        if variant == "none":
            affect = torch.zeros_like(affect)
        if variant in {"none", "affect"}:
            fast, slow = torch.zeros_like(fast), torch.zeros_like(slow)
        if variant != "dyadic":
            relation = torch.zeros_like(relation)
        return torch.cat([affect.flatten(1), fast.flatten(1), slow.flatten(1), relation.flatten(1)], -1)


StateCore = UnifiedEmotionStateCore
Memory = EmotionMemory

__all__ = ["UnifiedEmotionStateCore", "StateCore", "EmotionMemory", "Memory",
           "StateObservation", "KnownFutureInput"]
