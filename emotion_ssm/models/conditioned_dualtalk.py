from __future__ import annotations

from types import SimpleNamespace
from typing import Dict, Mapping, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from DualTalk import DualTalkModel
from emotion_ssm.schema import DyadicState, EventObservation


class StateFiLM(nn.Module):
    """Feature-wise conditioning initialized as the exact identity transform."""

    def __init__(self, state_context_dim: int, feature_dim: int, scale: float = 0.1) -> None:
        super().__init__()
        self.scale = scale
        self.affine = nn.Linear(state_context_dim, feature_dim * 2)
        nn.init.zeros_(self.affine.weight)
        nn.init.zeros_(self.affine.bias)

    def forward(self, features: Tensor, state_context: Tensor) -> Tensor:
        gamma, beta = self.affine(state_context).chunk(2, dim=-1)
        gamma = self.scale * torch.tanh(gamma)
        beta = self.scale * beta
        return features * (1.0 + gamma[:, None]) + beta[:, None]


class EmotionConditionedDualTalk(nn.Module):
    """DualTalk baseline with optional dyadic-state FiLM conditioning."""

    def __init__(
        self,
        blendshape_dim: int = 56,
        feature_dim: int = 256,
        state_dim: int = 128,
        relation_dim: int = 64,
        film_scale: float = 0.1,
    ) -> None:
        super().__init__()
        args = SimpleNamespace(blendshape_dim=blendshape_dim, feature_dim=feature_dim)
        self.baseline = DualTalkModel(args)
        self.context_dim = state_dim * 2 + relation_dim
        self.film = StateFiLM(self.context_dim, feature_dim * 2, film_scale)

    @classmethod
    def from_config(cls, cfg) -> "EmotionConditionedDualTalk":
        return cls(
            blendshape_dim=cfg.DUALTALK.BLENDSHAPE_DIM,
            feature_dim=cfg.DUALTALK.FEATURE_DIM,
            state_dim=cfg.MODEL.STATE_DIM,
            relation_dim=cfg.MODEL.RELATION_DIM,
            film_scale=cfg.DUALTALK.FILM_SCALE,
        )

    def forward(
        self,
        audio_target: Tensor,
        audio_partner: Tensor,
        partner_blendshape: Tensor,
        state_context: Optional[Tensor] = None,
        enable_film: bool = True,
    ) -> Tensor:
        audio_target_feature, audio_partner_feature, blendshape_feature = (
            self.baseline.joint_encoder(
                audio_target, audio_partner, partner_blendshape
            )
        )
        temporal_feature = self.baseline.temporal_enhancer(
            audio_partner_feature, blendshape_feature
        )
        interaction_feature = self.baseline.interaction_module(
            audio_target_feature, temporal_feature
        )
        if enable_film and state_context is not None:
            interaction_feature = self.film(interaction_feature, state_context)
        return self.baseline.synthesis_module(interaction_feature)

    def load_baseline_state_dict(self, state: Mapping[str, Tensor], strict: bool = True):
        if "models" in state and isinstance(state["models"], Mapping):
            models = state["models"]
            if "baseline" in models and isinstance(models["baseline"], Mapping):
                state = models["baseline"]
        if "state_dict" in state:
            state = state["state_dict"]
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        if "model" in state and isinstance(state["model"], Mapping):
            state = state["model"]
        clean = {}
        for name, value in state.items():
            for prefix in ("module.", "baseline."):
                if name.startswith(prefix):
                    name = name[len(prefix) :]
            clean[name] = value
        return self.baseline.load_state_dict(clean, strict=strict)


class AudioOnlyStateObserver(nn.Module):
    """Extract Wav2Vec2-Base features and invoke the A-only observation path."""

    def __init__(self, observation_encoder, model_name: str, local_files_only: bool) -> None:
        super().__init__()
        try:
            from transformers import Wav2Vec2Model
        except ImportError as error:
            raise ImportError("transformers is required for DualTalk state conditioning") from error
        self.backbone = Wav2Vec2Model.from_pretrained(
            model_name, local_files_only=local_files_only
        )
        self.backbone.feature_extractor._freeze_parameters()
        self.observation_encoder = observation_encoder

    def forward(self, waveform: Tensor, dataset_id: int = 1) -> EventObservation:
        attention_mask = torch.ones_like(waveform, dtype=torch.long)
        hidden = self.backbone(
            waveform, attention_mask=attention_mask
        ).last_hidden_state
        audio = hidden.mean(dim=1)
        batch_size = len(audio)
        batch = {
            "audio": audio,
            "face": audio.new_zeros(batch_size, 35),
            "text": audio.new_zeros(batch_size, 768),
            "dataset_id": torch.full(
                (batch_size,), dataset_id, dtype=torch.long, device=audio.device
            ),
            "modality_mask": torch.tensor(
                [1, 0, 0], dtype=torch.bool, device=audio.device
            )[None].expand(batch_size, -1),
        }
        output = self.observation_encoder(
            batch, torch.tensor([[1, 0, 0]], dtype=torch.bool, device=audio.device)
        )
        observation = output.select(0)
        observation.modality_mask = batch["modality_mask"]
        return observation


class DyadicAudioConditioner(nn.Module):
    """Turn a paired audio chunk into persistent [z_target,z_partner,r]."""

    def __init__(self, audio_observer: AudioOnlyStateObserver, state_model) -> None:
        super().__init__()
        self.audio_observer = audio_observer
        self.state_model = state_model

    def forward(
        self,
        audio_target: Tensor,
        audio_partner: Tensor,
        dt: Tensor,
        state: Optional[DyadicState] = None,
        correct: bool = True,
        enable_partner: bool = True,
    ) -> Tuple[Tensor, DyadicState, Dict[str, Tensor]]:
        batch_size = len(audio_target)
        if state is None:
            speaker_ids = torch.full(
                (batch_size, 2), -1, dtype=torch.long, device=audio_target.device
            )
            state = self.state_model.initialize(speaker_ids)
        target_observation = self.audio_observer(audio_target)
        partner_observation = self.audio_observer(audio_partner)
        half_dt = dt.reshape(-1).to(audio_target.dtype) * 0.5
        target_step = self.state_model.step(
            state,
            target_observation,
            torch.zeros(batch_size, dtype=torch.long, device=audio_target.device),
            half_dt,
            enable_partner=enable_partner,
            correct=correct,
        )
        partner_step = self.state_model.step(
            target_step.next_prior,
            partner_observation,
            torch.ones(batch_size, dtype=torch.long, device=audio_target.device),
            half_dt,
            enable_partner=enable_partner,
            correct=correct,
        )
        final_state = partner_step.next_prior
        context = torch.cat(
            [final_state.z[:, 0], final_state.z[:, 1], final_state.relation], dim=-1
        )
        return context, final_state, {
            "target_aff": target_observation.aff,
            "partner_aff": partner_observation.aff,
        }


class BlendshapeAffectProjector(nn.Module):
    """Small auxiliary head for generated-expression/state consistency."""

    def __init__(self, blendshape_dim: int, observation_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(blendshape_dim, 128),
            nn.GELU(),
            nn.Linear(128, observation_dim),
        )

    def forward(self, blendshape: Tensor) -> Tensor:
        return self.layers(blendshape.mean(dim=1))


def expression_state_consistency(
    projector: BlendshapeAffectProjector,
    generated: Tensor,
    target_affect: Tensor,
) -> Tensor:
    predicted = projector(generated)
    return (1.0 - F.cosine_similarity(predicted, target_affect, dim=-1)).mean()
