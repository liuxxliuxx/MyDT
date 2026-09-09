from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import Tensor


@dataclass
class ObservationOutput:
    """Outputs for K modality subsets.

    aff/event/action: [B, K, D], reliability: [B, K, 3],
    valid_subsets: [B, K]. ``modality_aff`` contains the three per-modality
    embeddings before subset aggregation when the encoder exposes them.
    """

    aff: Tensor
    event: Tensor
    action: Tensor
    reliability: Tensor
    valid_subsets: Tensor
    hidden: Tensor
    reliability_logits: Optional[Tensor] = None
    # Per-modality affect embeddings in the shared emotion space [B, 3, D].
    modality_aff: Optional[Tensor] = None
    # Reliability/presence weights used to aggregate each requested subset [B, K, 3].
    subset_weights: Optional[Tensor] = None
    # Student-only predictor outputs used for EMA self-supervision.  Downstream
    # dynamics always consume ``aff``/``modality_aff``, never these values.
    ssl_aff: Optional[Tensor] = None
    modality_ssl_aff: Optional[Tensor] = None
    # Unnormalised affect coordinates are retained for variance/covariance
    # regularisation. Cosine objectives and dynamics continue to consume aff.
    raw_aff: Optional[Tensor] = None
    raw_modality_aff: Optional[Tensor] = None

    def select(self, subset_index: int) -> "EventObservation":
        return EventObservation(
            aff=self.aff[:, subset_index],
            event=self.event[:, subset_index],
            action=self.action[:, subset_index],
            reliability=self.reliability[:, subset_index],
            modality_mask=None,
        )


@dataclass
class EventObservation:
    """One selected observation per batch element."""

    aff: Tensor
    event: Tensor
    action: Tensor
    reliability: Tensor
    modality_mask: Optional[Tensor]
    event_present: Optional[Tensor] = None
    action_duration: Optional[Tensor] = None

    def index(self, index: int) -> "EventObservation":
        mask = None if self.modality_mask is None else self.modality_mask[index]
        return EventObservation(
            aff=self.aff[index],
            event=self.event[index],
            action=self.action[index],
            reliability=self.reliability[index],
            modality_mask=mask,
            event_present=None if self.event_present is None else self.event_present[index],
            action_duration=None if self.action_duration is None else self.action_duration[index],
        )


@dataclass
class DyadicState:
    """Persistent two-person state.

    z: [B, 2, state_dim], relation: [B, relation_dim],
    speaker_ids: [B, 2], where -1 denotes an unseen speaker.
    """

    z: Tensor
    relation: Tensor
    speaker_ids: Tensor

    def detach(self) -> "DyadicState":
        return DyadicState(
            z=self.z.detach(),
            relation=self.relation.detach(),
            speaker_ids=self.speaker_ids,
        )

    def clone(self) -> "DyadicState":
        return DyadicState(
            z=self.z.clone(),
            relation=self.relation.clone(),
            speaker_ids=self.speaker_ids.clone(),
        )


@dataclass
class StepOutput:
    posterior: DyadicState
    next_prior: DyadicState
    correction_gate: Tensor
    influence: Tensor
    baseline: Tensor
    tau: Tensor
    auxiliary: Dict[str, Tensor]


def select_role(values: Tensor, role: Tensor) -> Tensor:
    """Select values[:, role] from a [B, 2, ...] tensor."""
    batch = torch.arange(values.shape[0], device=values.device)
    return values[batch, role.long()]


def replace_role(values: Tensor, role: Tensor, replacement: Tensor) -> Tensor:
    """Return a copy of [B, 2, ...] with the selected role replaced."""
    output = values.clone()
    batch = torch.arange(values.shape[0], device=values.device)
    output[batch, role.long()] = replacement
    return output
