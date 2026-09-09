from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass
class CounterfactualMatches:
    indices: Tensor
    valid: Tensor
    context_similarity: Tensor


@torch.no_grad()
def match_counterfactuals(
    context_event: Tensor,
    sender_action: Tensor,
    context_emotion: Tensor,
    context_intensity: Tensor,
    turn_position: Tensor,
    dialogue_id: Tensor,
    sender_role: Optional[Tensor] = None,
    top_k: int = 8,
    intensity_tolerance: float = 0.2,
    turn_tolerance: float = 0.15,
    max_action_cosine: float = 0.8,
    dataset_id: Optional[Tensor] = None,
) -> CounterfactualMatches:
    """Find context-near but action-different interventions in another dialogue.

    Every input describes the instant at which the sender acts.  In
    particular, callers must not supply the receiver's later reaction here:
    that reaction is held out as the ranking target after candidate selection.
    """
    count = len(context_event)
    event_similarity = F.normalize(context_event, dim=-1) @ F.normalize(
        context_event, dim=-1
    ).transpose(0, 1)
    action_similarity = F.normalize(sender_action, dim=-1) @ F.normalize(
        sender_action, dim=-1
    ).transpose(0, 1)
    valid = dialogue_id[:, None] != dialogue_id[None, :]
    if dataset_id is not None:
        valid &= dataset_id[:, None] == dataset_id[None, :]
    valid &= context_emotion[:, None] == context_emotion[None, :]
    valid &= context_emotion[:, None] >= 0
    valid &= (
        context_intensity[:, None] - context_intensity[None, :]
    ).abs() <= intensity_tolerance
    valid &= (turn_position[:, None] - turn_position[None, :]).abs() <= turn_tolerance
    valid &= action_similarity <= max_action_cosine
    if sender_role is not None:
        sender_role = sender_role.reshape(-1)
        if len(sender_role) != count:
            raise ValueError("sender_role must have one value per candidate")
        valid &= sender_role[:, None] == sender_role[None, :]
    valid.fill_diagonal_(False)

    k = min(top_k, max(count, 1))
    scores = event_similarity.masked_fill(~valid, float("-inf"))
    values, indices = scores.topk(k, dim=-1)
    selected_valid = torch.isfinite(values)
    indices = indices.masked_fill(~selected_valid, -1)
    return CounterfactualMatches(indices, selected_valid, values)
