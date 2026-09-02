from __future__ import annotations

from typing import Dict, Mapping, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from emotion_ssm.models.observation import SUBSET_MASKS
from emotion_ssm.schema import ObservationOutput


def zero_loss(reference: Tensor) -> Tensor:
    return reference.sum() * 0.0


def masked_mean(value: Tensor, mask: Tensor) -> Tensor:
    mask = mask.to(value.dtype)
    while mask.ndim < value.ndim:
        mask = mask.unsqueeze(-1)
    return (value * mask).sum() / mask.expand_as(value).sum().clamp_min(1.0)


def masked_cross_entropy(
    logits: Tensor,
    target: Tensor,
    mask: Tensor,
    class_weights: Optional[Tensor] = None,
) -> Tensor:
    flat_mask = mask.reshape(-1).bool()
    if not flat_mask.any():
        return zero_loss(logits)
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1])[flat_mask],
        target.reshape(-1)[flat_mask],
        weight=class_weights,
    )


def masked_mse(prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    return masked_mean((prediction - target).square(), mask)


def concordance_correlation_coefficient(
    prediction: Tensor, target: Tensor, mask: Tensor
) -> Tensor:
    values = []
    for dimension in range(prediction.shape[-1]):
        valid = mask[..., dimension].reshape(-1).bool()
        if valid.sum() < 2:
            continue
        x = prediction[..., dimension].reshape(-1)[valid]
        y = target[..., dimension].reshape(-1)[valid]
        x_mean = x.mean()
        y_mean = y.mean()
        covariance = ((x - x_mean) * (y - y_mean)).mean()
        denominator = x.var(unbiased=False) + y.var(unbiased=False) + (x_mean - y_mean).square()
        values.append(2.0 * covariance / denominator.clamp_min(1e-8))
    if not values:
        return zero_loss(prediction)
    return torch.stack(values).mean()


def decorrelation_loss(first: Tensor, second: Tensor, valid: Tensor) -> Tensor:
    selected_first = first.reshape(-1, first.shape[-1])[valid.reshape(-1)]
    selected_second = second.reshape(-1, second.shape[-1])[valid.reshape(-1)]
    if len(selected_first) < 2:
        return zero_loss(first)
    selected_first = F.normalize(selected_first - selected_first.mean(0), dim=-1)
    selected_second = F.normalize(selected_second - selected_second.mean(0), dim=-1)
    covariance = selected_first.transpose(0, 1) @ selected_second / len(selected_first)
    return covariance.square().mean()


def observation_losses(
    student: ObservationOutput,
    teacher: ObservationOutput,
    predictions: Mapping[str, Tensor],
    batch: Mapping[str, Tensor],
    class_weights: Tensor,
    cfg,
    subset_masks: Tensor = SUBSET_MASKS,
) -> Dict[str, Tensor]:
    target_aff = teacher.aff[:, :1].expand_as(student.aff)
    target_event = teacher.event[:, :1].expand_as(student.event)
    target_action = teacher.action[:, :1].expand_as(student.action)
    valid = student.valid_subsets

    consistency = zero_loss(student.aff)
    smooth = zero_loss(student.aff)
    for student_value, target_value in (
        (student.aff, target_aff),
        (student.event, target_event),
        (student.action, target_action),
    ):
        cosine = 1.0 - F.cosine_similarity(student_value, target_value, dim=-1)
        consistency = consistency + masked_mean(cosine, valid)
        smooth = smooth + masked_mean(
            F.smooth_l1_loss(student_value, target_value, reduction="none"), valid
        )
    consistency /= 3.0
    smooth /= 3.0

    num_subsets = student.aff.shape[1]
    emotion_target = batch["emotion"][:, None].expand(-1, num_subsets)
    label_valid = valid & (emotion_target >= 0)
    emotion = masked_cross_entropy(
        predictions["emotion"], emotion_target, label_valid, class_weights
    )
    intensity_target = batch["intensity"][:, None].expand(-1, num_subsets)
    intensity = masked_mse(predictions["intensity"], intensity_target, valid)
    vad_target = batch["vad"][:, None, :].expand(-1, num_subsets, -1)
    vad_mask = batch["vad_mask"][:, None, :].expand_as(vad_target)
    vad_mask = vad_mask & valid[:, :, None]
    vad_mse = masked_mse(predictions["vad"], vad_target, vad_mask)
    vad_ccc = concordance_correlation_coefficient(
        predictions["vad"], vad_target, vad_mask
    )
    vad = vad_mse + (1.0 - vad_ccc)

    speaker_target = batch["speaker"][:, None].expand(-1, num_subsets)
    speaker = masked_cross_entropy(
        predictions["speaker"],
        speaker_target,
        valid & (speaker_target >= 0),
    )
    domain_target = batch["dataset_id"][:, None].expand(-1, num_subsets)
    domain = masked_cross_entropy(predictions["domain"], domain_target, valid)

    actual = batch["modality_mask"][:, None, :]
    requested = subset_masks.to(actual.device)[None, :, :]
    reliability_target = (actual & requested).float()
    reliability = masked_mean(
        F.binary_cross_entropy(
            student.reliability, reliability_target.expand_as(student.reliability), reduction="none"
        ),
        valid,
    )
    decorrelation = (
        decorrelation_loss(student.aff, student.event, valid)
        + decorrelation_loss(student.aff, student.action, valid)
        + decorrelation_loss(student.event, student.action, valid)
    ) / 3.0

    total = (
        cfg.LOSS.UNIFY * consistency
        + cfg.LOSS.SMOOTH_L1 * smooth
        + cfg.LOSS.EMOTION * emotion
        + cfg.LOSS.INTENSITY * intensity
        + cfg.LOSS.VAD * vad
        + cfg.LOSS.RELIABILITY * reliability
        + cfg.LOSS.SPEAKER * speaker
        + cfg.LOSS.DOMAIN * domain
        + cfg.LOSS.DECORRELATION * decorrelation
    )
    return {
        "total": total,
        "consistency": consistency,
        "smooth_l1": smooth,
        "emotion": emotion,
        "intensity": intensity,
        "vad": vad,
        "vad_ccc": vad_ccc.detach(),
        "speaker": speaker,
        "domain": domain,
        "reliability": reliability,
        "decorrelation": decorrelation,
    }


def state_prediction_losses(
    predictions: Mapping[str, Tensor],
    target_aff: Tensor,
    target_emotion: Tensor,
    target_intensity: Tensor,
    target_vad: Tensor,
    target_vad_mask: Tensor,
    valid: Tensor,
    class_weights: Optional[Tensor] = None,
) -> Dict[str, Tensor]:
    affect = masked_mean(
        1.0 - F.cosine_similarity(predictions["aff"], target_aff, dim=-1), valid
    ) + masked_mean(
        F.smooth_l1_loss(predictions["aff"], target_aff, reduction="none"), valid
    )
    emotion_valid = valid & (target_emotion >= 0)
    emotion = masked_cross_entropy(
        predictions["emotion"], target_emotion, emotion_valid, class_weights
    )
    intensity = masked_mse(predictions["intensity"], target_intensity, valid)
    vad_valid = target_vad_mask & valid[..., None]
    if vad_valid.any():
        vad_mse = masked_mse(predictions["vad"], target_vad, vad_valid)
        vad_ccc = concordance_correlation_coefficient(
            predictions["vad"], target_vad, vad_valid
        )
        vad_loss = vad_mse + (1.0 - vad_ccc)
    else:
        vad_ccc = zero_loss(predictions["vad"])
        vad_loss = vad_ccc
    return {
        "affect": affect,
        "emotion": emotion,
        "intensity": intensity,
        "vad": vad_loss,
        "vad_ccc": vad_ccc.detach(),
    }
