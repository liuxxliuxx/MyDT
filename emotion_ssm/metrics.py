from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import Tensor


def confusion_matrix(
    target: Tensor, prediction: Tensor, num_classes: int = 7, mask: Optional[Tensor] = None
) -> Tensor:
    target = target.reshape(-1).long().cpu()
    prediction = prediction.reshape(-1).long().cpu()
    valid = target >= 0
    if mask is not None:
        valid &= mask.reshape(-1).bool().cpu()
    bins = torch.bincount(
        target[valid] * num_classes + prediction[valid],
        minlength=num_classes * num_classes,
    )
    return bins.reshape(num_classes, num_classes)


def classification_metrics(matrix: Tensor) -> Dict[str, float]:
    matrix = matrix.float()
    true_positive = matrix.diag()
    precision = true_positive / matrix.sum(0).clamp_min(1.0)
    recall = true_positive / matrix.sum(1).clamp_min(1.0)
    f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1e-8)
    return {"macro_f1": f1.mean().item(), "uar": recall.mean().item()}


def ccc_value(prediction: Tensor, target: Tensor, mask: Tensor) -> float:
    values = []
    for dimension in range(prediction.shape[-1]):
        valid = mask[..., dimension].reshape(-1).bool()
        if valid.sum() < 2:
            continue
        x = prediction[..., dimension].reshape(-1)[valid].float()
        y = target[..., dimension].reshape(-1)[valid].float()
        covariance = ((x - x.mean()) * (y - y.mean())).mean()
        denominator = x.var(unbiased=False) + y.var(unbiased=False) + (x.mean() - y.mean()).square()
        values.append((2.0 * covariance / denominator.clamp_min(1e-8)).item())
    return sum(values) / len(values) if values else 0.0
