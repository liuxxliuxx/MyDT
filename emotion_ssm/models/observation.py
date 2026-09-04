from __future__ import annotations

import copy
import math
from typing import Dict, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
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


class TemporalAUEncoder(nn.Module):
    """Encode AU frames with positional information and mask-safe pooling."""

    def __init__(
        self,
        face_dim: int = 35,
        model_dim: int = 256,
        dropout: float = 0.1,
        layers: int = 3,
        heads: int = 4,
        input_dim: Optional[int] = None,
        num_layers: Optional[int] = None,
        num_heads: Optional[int] = None,
    ):
        super().__init__()
        # Accept the descriptive names used by the design document while
        # retaining the original constructor names for checkpoint scripts.
        if input_dim is not None:
            face_dim = input_dim
        if num_layers is not None:
            layers = num_layers
        if num_heads is not None:
            heads = num_heads
        self.input = nn.Sequential(
            nn.Linear(face_dim, model_dim), nn.LayerNorm(model_dim), nn.GELU()
        )
        block = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=heads if model_dim % heads == 0 else 1,
            dim_feedforward=model_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(block, num_layers=layers)
        self.attention_pool = nn.Linear(model_dim, 1)
        self.output = nn.Sequential(
            nn.LayerNorm(model_dim), nn.Linear(model_dim, model_dim)
        )

    @property
    def input_projection(self) -> nn.Module:
        """Descriptive alias retained without duplicating state-dict entries."""
        return self.input

    @property
    def temporal_encoder(self) -> nn.Module:
        return self.encoder

    @staticmethod
    def _position(length: int, dim: int, device, dtype) -> Tensor:
        position = torch.arange(length, device=device, dtype=dtype)[:, None]
        scale = torch.exp(
            torch.arange(0, dim, 2, device=device, dtype=dtype)
            * (-math.log(10000.0) / max(dim, 1))
        )
        encoding = torch.zeros(length, dim, device=device, dtype=dtype)
        encoding[:, 0::2] = torch.sin(position * scale)
        if dim > 1:
            encoding[:, 1::2] = torch.cos(
                position * scale[: encoding[:, 1::2].shape[1]]
            )
        return encoding

    def forward(
        self,
        face: Tensor,
        frame_mask: Optional[Tensor] = None,
        confidence: Optional[Tensor] = None,
    ) -> Tensor:
        if face.ndim == 2:
            face = face[:, None, :]
        if face.ndim != 3:
            raise ValueError(f"face must be [B,T,F] or [B,F], got {tuple(face.shape)}")
        batch, length, _ = face.shape
        if frame_mask is None:
            frame_mask = torch.ones(batch, length, dtype=torch.bool, device=face.device)
        elif frame_mask.ndim == 1:
            frame_mask = frame_mask[:, None]
        if frame_mask.shape != (batch, length):
            raise ValueError(
                f"frame_mask must have shape {(batch, length)}, got {tuple(frame_mask.shape)}"
            )
        frame_mask = frame_mask.bool()
        if confidence is None:
            confidence = torch.ones(batch, length, device=face.device, dtype=face.dtype)
        elif confidence.ndim == 1:
            confidence = confidence[:, None]
        if confidence.shape != (batch, length):
            raise ValueError(
                f"confidence must have shape {(batch, length)}, got {tuple(confidence.shape)}"
            )
        confidence = torch.nan_to_num(
            confidence.to(face.dtype), nan=0.0, posinf=1.0, neginf=0.0
        ).clamp(0.0, 1.0)
        frame_mask = frame_mask & (confidence > 0) & torch.isfinite(face).all(dim=-1)
        # Transformer attention cannot softmax an all-masked row. Keep one
        # zero-valued sentinel unmasked for absent modalities, then zero output.
        safe_mask = frame_mask.clone()
        empty = ~safe_mask.any(dim=1)
        if empty.any():
            safe_mask[empty, 0] = True
        clean = torch.nan_to_num(face) * confidence[..., None]
        if empty.any():
            clean = clean.clone()
            clean[empty, 0] = 0
        hidden = self.input(clean)
        hidden = hidden + self._position(
            length, hidden.shape[-1], hidden.device, hidden.dtype
        )[None]
        hidden = self.encoder(hidden, src_key_padding_mask=~safe_mask)
        score = self.attention_pool(hidden).squeeze(-1)
        score = score + torch.log(confidence.clamp_min(1e-4))
        score = score.masked_fill(~safe_mask, -1e4)
        weights = torch.softmax(score, dim=-1) * frame_mask.to(hidden.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        pooled = (hidden * weights[..., None]).sum(dim=1)
        pooled = self.output(pooled)
        return pooled * frame_mask.any(dim=1, keepdim=True).to(pooled.dtype)


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
        au_num_layers: int = 3,
        au_num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.audio_adapter = DomainProjector(
            audio_dim, model_dim, num_domains, dropout
        )
        # The face adapter consumes the temporal encoder output.  A 2-D legacy
        # face vector is treated as a one-frame sequence in ``forward``.
        self.face_adapter = DomainProjector(model_dim, model_dim, num_domains, dropout)
        self.text_adapter = DomainProjector(text_dim, model_dim, num_domains, dropout)
        self.face_temporal = TemporalAUEncoder(
            face_dim, model_dim, dropout, au_num_layers, au_num_heads
        )
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
        # All three modalities use this same map.  The resulting vectors are
        # the actual common affect coordinates; subset fusion only averages
        # these vectors and never rotates them with a second modality-specific
        # head.
        self.shared_affect_projector = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, observation_dim),
            nn.LayerNorm(observation_dim),
        )
        # A predictor makes the EMA target asymmetric.  It is used only by the
        # self-supervised objective, so inference stays in the raw affect space.
        self.ssl_predictor = nn.Sequential(
            nn.Linear(observation_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, observation_dim),
        )
        self.affect_weight = nn.Linear(observation_dim, 1)
        self.event_head = nn.Sequential(
            nn.Linear(model_dim, observation_dim), nn.LayerNorm(observation_dim)
        )
        self.action_head = nn.Sequential(
            nn.Linear(model_dim, observation_dim), nn.LayerNorm(observation_dim)
        )
        # Keep logits for the AMP-safe BCE-with-logits training objective.  The
        # sigmoid probability is still exposed to the dynamics model as before.
        self.reliability_head = nn.Sequential(nn.Linear(model_dim, 3))
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
            au_num_layers=cfg.MODEL.AU_NUM_LAYERS,
            au_num_heads=cfg.MODEL.AU_NUM_HEADS,
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

        audio_token = self.audio_adapter(audio, domain)
        text_token = self.text_adapter(text, domain)
        face_token = self.face_temporal(
            face, batch.get("face_frame_mask"), batch.get("face_confidence")
        )
        face_token = self.face_adapter(face_token, domain)
        projected = torch.stack([audio_token, face_token, text_token], dim=1)

        # Per-modality affect lives in a shared 128-D space.  Normalize before
        # aggregation so reliability changes the evidence mixture, not its
        # scale.
        modality_aff = F.normalize(
            self.shared_affect_projector(projected), dim=-1
        )
        batch_size = len(audio)
        num_subsets = len(subset_masks)
        present = actual_mask[:, None, :] & subset_masks[None, :, :]
        valid_subsets = present.any(dim=-1)
        quality = batch.get("reliability")
        if quality is None:
            quality = torch.ones_like(actual_mask, dtype=audio.dtype)
        quality = quality.to(audio.dtype).clamp(0.0, 1.0)
        scores = self.affect_weight(modality_aff).squeeze(-1)[:, None, :]
        scores = scores + torch.log(quality[:, None, :].clamp_min(1e-4))
        scores = scores.masked_fill(~present, -1e4)
        subset_weights = torch.softmax(scores, dim=-1) * present.to(scores.dtype)
        subset_weights = subset_weights / subset_weights.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-6)
        aff = torch.einsum("bkm,bmd->bkd", subset_weights, modality_aff)
        aff = F.normalize(aff, dim=-1) * valid_subsets[..., None].to(aff.dtype)
        ssl_aff = self.ssl_predictor(aff)
        modality_ssl_aff = self.ssl_predictor(modality_aff)

        flat_present = present.reshape(batch_size * num_subsets, 3)
        projected_for_fusion = projected + self.modality_embedding
        tokens = projected_for_fusion[:, None].expand(-1, num_subsets, -1, -1)
        tokens = tokens.reshape(batch_size * num_subsets, 3, -1)
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
        # Event is a semantic signal, so an audio/video-only subset must not
        # manufacture one from masked text.  Action remains a multimodal
        # behavior representation and is learned later from partner response.
        event = self.event_head(text_token)[:, None].expand(-1, num_subsets, -1)
        event = event * present[:, :, 2, None].to(event.dtype)
        reliability_logits = self.reliability_head(hidden)
        return ObservationOutput(
            aff=aff,
            event=event,
            action=self.action_head(hidden),
            reliability=torch.sigmoid(reliability_logits),
            valid_subsets=valid_subsets,
            hidden=hidden,
            reliability_logits=reliability_logits,
            modality_aff=modality_aff,
            subset_weights=subset_weights,
            ssl_aff=ssl_aff,
            modality_ssl_aff=modality_ssl_aff,
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
