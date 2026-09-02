from __future__ import annotations

from dataclasses import dataclass

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
    receiver_event: Tensor,
    sender_action: Tensor,
    receiver_emotion: Tensor,
    receiver_intensity: Tensor,
    turn_position: Tensor,
    dialogue_id: Tensor,
    top_k: int = 8,
    intensity_tolerance: float = 0.2,
    turn_tolerance: float = 0.15,
    max_action_cosine: float = 0.8,
) -> CounterfactualMatches:
    """Find in-batch context-near but action-different partner behavior."""
    count = len(receiver_event)
    event_similarity = F.normalize(receiver_event, dim=-1) @ F.normalize(
        receiver_event, dim=-1
    ).transpose(0, 1)
    action_similarity = F.normalize(sender_action, dim=-1) @ F.normalize(
        sender_action, dim=-1
    ).transpose(0, 1)
    valid = dialogue_id[:, None] != dialogue_id[None, :]
    valid &= receiver_emotion[:, None] == receiver_emotion[None, :]
    valid &= receiver_emotion[:, None] >= 0
    valid &= (
        receiver_intensity[:, None] - receiver_intensity[None, :]
    ).abs() <= intensity_tolerance
    valid &= (turn_position[:, None] - turn_position[None, :]).abs() <= turn_tolerance
    valid &= action_similarity <= max_action_cosine
    valid.fill_diagonal_(False)

    k = min(top_k, max(count, 1))
    scores = event_similarity.masked_fill(~valid, float("-inf"))
    values, indices = scores.topk(k, dim=-1)
    selected_valid = torch.isfinite(values)
    indices = indices.masked_fill(~selected_valid, -1)
    return CounterfactualMatches(indices, selected_valid, values)
