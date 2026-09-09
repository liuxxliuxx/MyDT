"""Deployment observation with a separate, calibrated FLAME visual route."""
from __future__ import annotations

import torch
from torch import nn

from emotion_ssm.models.observation import TemporalAUEncoder
from emotion_ssm.schema import EventObservation


class MultimodalObserver(nn.Module):
    def __init__(self, encoder, model_dim=256, heads=4, layers=3, dropout=.1, domain_id=2):
        super().__init__()
        self.encoder = encoder
        self.flame = TemporalAUEncoder(56, model_dim, dropout, layers, heads)
        self.domain_id = domain_id

    def prepare_batch(self, features, use_flame=True):
        batch = dict(features)
        audio = batch["audio"]
        batch.setdefault("dataset_id", torch.full((len(audio),), self.domain_id,
                                                  dtype=torch.long, device=audio.device))
        batch.setdefault("face", audio.new_zeros(len(audio), 1, 35))
        batch.setdefault("face_frame_mask", torch.zeros(len(audio), 1, dtype=torch.bool, device=audio.device))
        batch.setdefault("face_confidence", audio.new_zeros(len(audio), 1))
        if use_flame and "flame" in batch:
            batch["visual_token"] = self.flame(batch["flame"], batch.get("flame_mask"),
                                               batch.get("flame_confidence"))
        return batch

    def forward(self, features, use_flame=True):
        batch = self.prepare_batch(features, use_flame)
        audio = batch["audio"]
        output = self.encoder(batch, torch.ones(1, 3, dtype=torch.bool, device=audio.device))
        observation = output.select(0)
        observation.modality_mask = batch["modality_mask"].bool()
        observation.event_present = batch.get("event_present", observation.modality_mask[:, 2])
        observation.action_duration = batch.get("action_duration")
        return observation

    @torch.no_grad()
    def diagnose(self, features, teacher_domain, use_flame=True):
        batch = self.prepare_batch(features, use_flame)
        output = self.encoder(batch)
        teacher_batch = dict(batch)
        teacher_batch["dataset_id"] = torch.full_like(batch["dataset_id"], teacher_domain)
        teacher_batch["modality_mask"] = batch["modality_mask"].clone()
        teacher_batch["modality_mask"][:, 1] = False
        teacher_batch.pop("visual_token", None)
        teacher = self.encoder(teacher_batch, batch["audio"].new_tensor([[1, 0, 1]], dtype=torch.bool))
        return output, teacher.aff[:, 0], teacher.valid_subsets[:, 0]

    def freeze(self):
        self.requires_grad_(False)
        self.eval()


@torch.no_grad()
def representation_diagnostics(affect, teacher=None):
    value = torch.nn.functional.normalize(affect.float(), dim=-1)
    count = len(value)
    if count < 2:
        return {"samples": count, "mean_std": None, "pairwise_cosine": None}
    mean_cosine = (value.sum(0).square().sum() - value.square().sum()) / (count * (count - 1))
    result = {"samples": count, "mean_std": float(value.std(0, unbiased=False).mean()),
              "pairwise_cosine": float(mean_cosine)}
    if teacher is not None:
        result["teacher_cosine_error"] = float((1 - torch.nn.functional.cosine_similarity(value, teacher.float())).mean())
    return result
