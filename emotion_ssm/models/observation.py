from __future__ import annotations

import copy
from typing import Dict, Mapping, Optional

import torch
import torch.nn as nn
from torch import Tensor

from emotion_ssm.schema import ObservationOutput


SUBSET_NAMES = ("A", "V", "T", "AV", "AT", "VT", "AVT")
SUBSET_MASKS = torch.tensor(
    [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
    ],
    dtype=torch.bool,
)


class Projector(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, value: Tensor) -> Tensor:
        return self.layers(value)


class DomainProjector(nn.Module):
    """Select one adapter per sample while keeping a shared output space."""

    def __init__(
        self, input_dim: int, output_dim: int, num_domains: int, dropout: float
    ) -> None:
        super().__init__()
        self.adapters = nn.ModuleList(
            [Projector(input_dim, output_dim, dropout) for _ in range(num_domains)]
        )

    def forward(self, value: Tensor, domain: Tensor) -> Tensor:
        if domain.min().item() < 0 or domain.max().item() >= len(self.adapters):
            raise ValueError("dataset_id is outside the configured domain range")
        candidates = torch.stack([adapter(value) for adapter in self.adapters], dim=1)
        gather_index = domain[:, None, None].expand(-1, 1, candidates.shape[-1])
        return candidates.gather(1, gather_index).squeeze(1)


class ObservationEncoder(nn.Module):
    """Fuse A/V/T subsets and factor the result into affect/event/action."""

    def __init__(
        self,
        audio_dim: int = 768,
        face_dim: int = 35,
        text_dim: int = 768,
        model_dim: int = 256,
        observation_dim: int = 128,
        num_domains: int = 2,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.audio_adapter = DomainProjector(
            audio_dim, model_dim, num_domains, dropout
        )
        self.face_adapter = DomainProjector(face_dim, model_dim, num_domains, dropout)
        self.text_adapter = DomainProjector(text_dim, model_dim, num_domains, dropout)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, model_dim))
        self.modality_embedding = nn.Parameter(torch.empty(1, 3, model_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.fusion = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.aff_head = nn.Sequential(nn.Linear(model_dim, observation_dim), nn.LayerNorm(observation_dim))
        self.event_head = nn.Sequential(nn.Linear(model_dim, observation_dim), nn.LayerNorm(observation_dim))
        self.action_head = nn.Sequential(nn.Linear(model_dim, observation_dim), nn.LayerNorm(observation_dim))
        self.reliability_head = nn.Sequential(nn.Linear(model_dim, 3), nn.Sigmoid())
        self.register_buffer("subset_masks", SUBSET_MASKS.clone(), persistent=False)
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.modality_embedding, std=0.02)

    @classmethod
    def from_config(cls, cfg) -> "ObservationEncoder":
        return cls(
            audio_dim=cfg.MODEL.AUDIO_DIM,
            face_dim=cfg.MODEL.FACE_DIM,
            text_dim=cfg.MODEL.TEXT_DIM,
            model_dim=cfg.MODEL.MODEL_DIM,
            observation_dim=cfg.MODEL.OBSERVATION_DIM,
            num_domains=cfg.MODEL.NUM_DOMAINS,
            num_layers=cfg.MODEL.NUM_LAYERS,
            num_heads=cfg.MODEL.NUM_HEADS,
            dropout=cfg.MODEL.DROPOUT,
        )

    def forward(
        self,
        batch: Mapping[str, Tensor],
        subset_masks: Optional[Tensor] = None,
    ) -> ObservationOutput:
        audio = batch["audio"]
        face = batch["face"]
        text = batch["text"]
        domain = batch["dataset_id"].long()
        actual_mask = batch["modality_mask"].bool()
        if subset_masks is None:
            subset_masks = self.subset_masks
        subset_masks = subset_masks.to(audio.device, dtype=torch.bool)
        if subset_masks.ndim == 1:
            subset_masks = subset_masks[None]
        if subset_masks.ndim != 2 or subset_masks.shape[-1] != 3:
            raise ValueError("subset_masks must be [3] or [K, 3]")

        projected = torch.stack(
            [
                self.audio_adapter(audio, domain),
                self.face_adapter(face, domain),
                self.text_adapter(text, domain),
            ],
            dim=1,
        )
        projected = projected + self.modality_embedding
        batch_size = len(audio)
        num_subsets = len(subset_masks)
        tokens = projected[:, None].expand(-1, num_subsets, -1, -1)
        tokens = tokens.reshape(batch_size * num_subsets, 3, -1)

        present = actual_mask[:, None, :] & subset_masks[None, :, :]
        valid_subsets = present.any(dim=-1)
        flat_present = present.reshape(batch_size * num_subsets, 3)
        cls = self.cls_token.expand(batch_size * num_subsets, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        padding = torch.cat(
            [
                torch.zeros(
                    batch_size * num_subsets,
                    1,
                    dtype=torch.bool,
                    device=tokens.device,
                ),
                ~flat_present,
            ],
            dim=1,
        )
        hidden = self.fusion(tokens, src_key_padding_mask=padding)[:, 0]
        hidden = hidden.reshape(batch_size, num_subsets, -1)
        return ObservationOutput(
            aff=self.aff_head(hidden),
            event=self.event_head(hidden),
            action=self.action_head(hidden),
            reliability=self.reliability_head(hidden),
            valid_subsets=valid_subsets,
            hidden=hidden,
        )


class _GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value: Tensor, alpha: float) -> Tensor:
        ctx.alpha = alpha
        return value.view_as(value)

    @staticmethod
    def backward(ctx, gradient: Tensor):
        return -ctx.alpha * gradient, None


def grad_reverse(value: Tensor, alpha: float = 1.0) -> Tensor:
    return _GradientReversal.apply(value, float(alpha))


class AffectDecoder(nn.Module):
    """Decode either observation-affect or persistent state into labels."""

    def __init__(self, input_dim: int = 128, num_emotions: int = 7) -> None:
        super().__init__()
        self.emotion = nn.Linear(input_dim, num_emotions)
        self.intensity = nn.Sequential(
            nn.Linear(input_dim, 64), nn.GELU(), nn.Linear(64, 1), nn.Softplus()
        )
        self.vad = nn.Sequential(nn.Linear(input_dim, 64), nn.GELU(), nn.Linear(64, 3), nn.Tanh())

    def forward(self, value: Tensor) -> Dict[str, Tensor]:
        return {
            "emotion": self.emotion(value),
            "intensity": self.intensity(value).squeeze(-1),
            "vad": self.vad(value),
        }


class ObservationSupervisionHeads(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        num_speakers: int,
        num_domains: int,
        num_emotions: int = 7,
    ) -> None:
        super().__init__()
        self.affect = AffectDecoder(observation_dim, num_emotions)
        self.speaker = nn.Sequential(
            nn.Linear(observation_dim, 64), nn.GELU(), nn.Linear(64, max(num_speakers, 1))
        )
        self.domain = nn.Sequential(
            nn.Linear(observation_dim, 64), nn.GELU(), nn.Linear(64, num_domains)
        )

    def forward(self, aff: Tensor, grl_alpha: float) -> Dict[str, Tensor]:
        output = self.affect(aff)
        reversed_aff = grad_reverse(aff, grl_alpha)
        output["speaker"] = self.speaker(reversed_aff)
        output["domain"] = self.domain(reversed_aff)
        return output


def build_ema_teacher(student: ObservationEncoder) -> ObservationEncoder:
    teacher = copy.deepcopy(student)
    teacher.requires_grad_(False)
    teacher.eval()
    return teacher


@torch.no_grad()
def update_ema(
    teacher: ObservationEncoder,
    student: ObservationEncoder,
    decay: float,
) -> None:
    for teacher_parameter, student_parameter in zip(
        teacher.parameters(), student.parameters()
    ):
        teacher_parameter.mul_(decay).add_(student_parameter, alpha=1.0 - decay)
    for teacher_buffer, student_buffer in zip(teacher.buffers(), student.buffers()):
        teacher_buffer.copy_(student_buffer)
    teacher.eval()
