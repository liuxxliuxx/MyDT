"""V3.1 role-conditioned observation with clean unit-affect self-supervision.

Cached tokens must have independent dependency units, such as nonoverlapping
500ms acoustic atoms. Slicing full-utterance hidden states does not make them
local. Mask replacement happens before contextual attention in this model.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import math
from typing import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from emotion_ssm.schema import EventObservation

SUBSETS = {"A": (1, 0, 0), "V": (0, 1, 0), "T": (0, 0, 1),
           "AV": (1, 1, 0), "AT": (1, 0, 1), "VT": (0, 1, 1), "AVT": (1, 1, 1)}
MODES = ("audio", "prosody", "au", "flame", "text")
TOKEN_PROTOCOL = "emotion-token-packets-v3.1"


@dataclass
class TokenObserverConfig:
    audio_dim: int = 768
    text_dim: int = 768
    prosody_dim: int = 8
    au_dim: int = 35
    flame_dim: int = 56
    model_dim: int = 256
    affect_dim: int = 128
    num_layers: int = 2
    num_heads: int = 4
    num_domains: int = 3
    dropout: float = .1
    token_protocol: str = TOKEN_PROTOCOL
    summary_dim: int = 32
    variance_scale: float = .5
    variance_weight: float = 1.
    covariance_weight: float = .04
    uniformity_weight: float = .1
    mean_weight: float = .1
    uniformity_temperature: float = 2.
    token_reconstruction_weight: float = 1.
    affect_summary_weight: float = 1.
    affect_distillation_weight: float = .5


class TokenObserver(nn.Module):
    """One shared affect coordinate system, queried relative to SELF/PARTNER."""

    def __init__(self, config: TokenObserverConfig | Mapping | None = None):
        super().__init__()
        self.config = config if isinstance(config, TokenObserverConfig) else TokenObserverConfig(**dict(config or {}))
        c = self.config
        if c.token_protocol != TOKEN_PROTOCOL:
            raise ValueError("V3.1 observer requires its explicit token protocol; legacy weights cannot resume")
        if c.model_dim % c.num_heads:
            raise ValueError("model_dim must divide num_heads")
        if c.summary_dim < 1 or not 0 < c.variance_scale <= 1 or c.uniformity_temperature <= 0:
            raise ValueError("Invalid affect summary or unit-sphere regularization configuration")
        self.adapters = nn.ModuleDict({m: nn.ModuleList([
            nn.Sequential(nn.Linear(getattr(c, m + "_dim"), c.model_dim),
                          nn.LayerNorm(c.model_dim), nn.GELU()) for _ in range(c.num_domains)
        ]) for m in MODES})
        self.mode_embedding = nn.Embedding(len(MODES), c.model_dim)
        self.relative_role = nn.Embedding(2, c.model_dim)
        self.query = nn.Parameter(torch.randn(1, 1, c.model_dim) * .02)
        self.mask_token = nn.Parameter(torch.randn(len(MODES), c.model_dim) * .02)
        self.time_projection = nn.Linear(8, c.model_dim, bias=False)
        self.text_position_projection = nn.Linear(16, c.model_dim, bias=False)
        layer = nn.TransformerEncoderLayer(c.model_dim, c.num_heads, c.model_dim * 4,
                                           c.dropout, activation="gelu", batch_first=True,
                                           norm_first=True)
        self.fusion = nn.TransformerEncoder(layer, c.num_layers, enable_nested_tensor=False)
        self.affect_head = nn.Sequential(nn.LayerNorm(c.model_dim), nn.Linear(c.model_dim, c.affect_dim))
        self.event_head = nn.Sequential(nn.LayerNorm(c.model_dim), nn.Linear(c.model_dim, c.affect_dim))
        self.action_head = nn.Sequential(nn.LayerNorm(c.model_dim), nn.Linear(c.model_dim, c.affect_dim))
        self.reliability_head = nn.Linear(c.model_dim, 3)
        self.reconstruct = nn.ModuleDict({m: nn.Linear(c.model_dim, getattr(c, m + "_dim")) for m in MODES})
        # Coarse masked prediction has an explicit affect bottleneck. Fixed
        # target projections cannot rotate/collapse together with their decoder.
        self.affect_summary_heads = nn.ModuleDict({m: nn.Sequential(
            nn.Linear(c.affect_dim, c.model_dim), nn.GELU(), nn.Linear(c.model_dim, c.summary_dim)
        ) for m in MODES})
        projection_rng = torch.Generator().manual_seed(310908)
        for mode in MODES:
            dimension = getattr(c, mode + "_dim")
            projection = torch.randn(dimension, c.summary_dim, generator=projection_rng) / math.sqrt(dimension)
            self.register_buffer("summary_projection_" + mode, projection)
        self.emotion_head = nn.Linear(c.affect_dim, 7)
        self.intensity_head = nn.Linear(c.affect_dim, 1)
        self.vad_head = nn.Linear(c.affect_dim, 3)

    def construction(self):
        return asdict(self.config)

    @staticmethod
    def _flag(features, name, default, device):
        value = features.get(name, default)
        return torch.as_tensor(value, device=device).bool()

    def _mode_features(self, features, mode):
        if mode + "_tokens" in features:
            return features[mode + "_tokens"], features[mode + "_mask"].bool()
        if mode != "prosody":
            raise KeyError(mode + "_tokens")
        # Missing prosody is explicit absence, not a fabricated acoustic feature.
        audio = features["audio_tokens"]
        return audio.new_zeros(len(audio), 1, self.config.prosody_dim), torch.zeros(
            len(audio), 1, device=audio.device, dtype=torch.bool)

    def synchronize_corruption(self, features, corruption):
        """Mask audio and its same-atom prosody together, including caller masks."""
        output = {}
        for mode in MODES:
            _, valid = self._mode_features(features, mode)
            chosen = (torch.zeros_like(valid) if corruption is None or mode not in corruption
                      else corruption[mode].to(device=valid.device, dtype=torch.bool))
            if chosen.shape != valid.shape:
                raise ValueError(f"Invalid {mode} corruption mask")
            output[mode] = chosen & valid
        _, prosody = self._mode_features(features, "prosody")
        _, audio = self._mode_features(features, "audio")
        if prosody.any():
            if prosody.shape != audio.shape:
                raise ValueError("Prosody must align with its acoustic dependency atoms")
            audio_time, prosody_time = features.get("audio_times"), features.get("prosody_times")
            if audio_time is not None and prosody_time is not None:
                if not torch.allclose(audio_time[prosody], prosody_time[prosody], rtol=0, atol=1e-6):
                    raise ValueError("Prosody and audio dependency timestamps disagree")
            shared = output["audio"] | output["prosody"]
            output["audio"] = shared & audio
            output["prosody"] = shared & prosody
        return output

    @contextmanager
    def _clean_view(self):
        """Disable attention/residual dropout while retaining parameter gradients."""
        modes = [(module, module.training) for module in self.fusion.modules()]
        self.fusion.eval()
        try:
            yield
        finally:
            for module, training in modes:
                module.training = training

    def encode_clean(self, features, subset=None):
        with self._clean_view():
            return self.encode(features, subset=subset)

    def _tokens(self, features, subset, corruption):
        ref = features["audio_tokens"]
        batch, device = ref.shape[0], ref.device
        domain = features.get("domain_id", torch.zeros(batch, device=device, dtype=torch.long)).long()
        if ((domain < 0) | (domain >= self.config.num_domains)).any():
            raise ValueError("Unknown token source domain")
        selected = torch.tensor(SUBSETS[subset or "AVT"], device=device, dtype=torch.bool)
        now = features.get("now", torch.zeros(batch, device=device))
        pieces, valid, spans, original_masks, masked = [], [], {}, {}, {}
        offset = 1
        corruption = self.synchronize_corruption(features, corruption)
        for index, mode in enumerate(MODES):
            value, mask = self._mode_features(features, mode)
            mask = mask & torch.isfinite(value).all(-1)
            if value.ndim != 3 or value.shape[-1] != getattr(self.config, mode + "_dim"):
                raise ValueError(f"Invalid {mode} token dimensions")
            mode_index = 0 if mode in ("audio", "prosody") else 2 if mode == "text" else 1
            mask = mask & selected[mode_index]
            candidates = torch.stack([adapter(torch.nan_to_num(value).to(self.query.dtype)) for adapter in self.adapters[mode]], 1)
            hidden = candidates[torch.arange(batch, device=device), domain]
            corrupt = corruption[mode]
            corrupt = corrupt & mask
            # Mask before positional/role tags and, critically, before fusion.
            hidden = torch.where(corrupt[..., None], self.mask_token[index], hidden)
            times = features.get(mode + "_times")
            if times is None:
                times = torch.linspace(0, 1, value.shape[1], device=device)[None].expand(batch, -1)
            age = (times - now[:, None]).clamp(-3600, 0)
            freq = age[..., None] / torch.tensor([1., 4., 16., 64.], device=device)
            hidden = hidden + self.time_projection(torch.cat((freq.sin(), freq.cos()), -1).to(self.query.dtype))
            hidden = hidden + self.mode_embedding.weight[index]
            roles = features.get("text_roles", torch.zeros_like(mask, dtype=torch.long)) if mode == "text" else torch.zeros_like(mask, dtype=torch.long)
            if ((roles < 0) | (roles > 1)).any():
                raise ValueError("text_roles must be relative SELF=0/PARTNER=1")
            hidden = hidden + self.relative_role(roles.long())
            if mode == "text":
                positions = features.get("text_positions")
                if positions is None or positions.shape != mask.shape:
                    # Lightweight callers may omit cached positions; array order
                    # is then the declared text order. Dataset validation checks
                    # explicit cached position shapes at the source boundary.
                    positions = torch.arange(value.shape[1], device=device)[None].expand(batch, -1)
                if (positions[mask] < 0).any():
                    raise ValueError("Text token positions must be non-negative")
                frequency = positions.float()[..., None] / torch.logspace(0, 4, 8, device=device)
                position_code = torch.cat([frequency.sin(), frequency.cos()], -1)
                hidden = hidden + self.text_position_projection(position_code.to(self.query.dtype))
            pieces.append(hidden)
            valid.append(mask)
            spans[mode] = slice(offset, offset + value.shape[1])
            original_masks[mode], masked[mode] = mask, corrupt
            offset += value.shape[1]
        query = self.query.expand(batch, -1, -1) + self.relative_role.weight[0]
        return torch.cat([query] + pieces, 1), torch.cat([
            torch.ones(batch, 1, device=device, dtype=torch.bool)] + valid, 1), spans, original_masks, masked

    def encode(self, features, subset=None, corruption=None):
        hidden, mask, spans, masks, masked = self._tokens(features, subset, corruption)
        hidden = self.fusion(hidden, src_key_padding_mask=~mask)
        pooled = hidden[:, 0]
        raw_aff = self.affect_head(pooled)
        modality_mask = torch.stack((masks["audio"].any(1) | masks["prosody"].any(1),
                                     masks["au"].any(1) | masks["flame"].any(1), masks["text"].any(1)), 1)
        valid = modality_mask.any(1)
        aff = F.normalize(raw_aff.float(), dim=-1, eps=1e-6) * valid[:, None]
        event_mask = features.get("text_fresh_mask", torch.zeros_like(masks["text"])).bool() & masks["text"]
        # Only own newly arrived text is a sender's event; shared history is context.
        event_mask = event_mask & (features["text_roles"] == 0)
        event_pooled = (hidden[:, spans["text"]] * event_mask[..., None]).sum(1) / event_mask.sum(1, keepdim=True).clamp_min(1)
        event_present = event_mask.any(1) & self._flag(features, "event_present", event_mask.any(1), pooled.device)
        event = self.event_head(event_pooled) * event_present[:, None]
        fresh = features.get("fresh_observation", modality_mask).bool() & modality_mask
        action_present = self._flag(features, "action_present", fresh.any(1), pooled.device) & fresh.any(1)
        action = self.action_head(pooled) * action_present[:, None]
        reliability = torch.sigmoid(self.reliability_head(pooled)) * modality_mask
        obs = EventObservation(aff=aff, event=event, action=action, reliability=reliability,
                               modality_mask=modality_mask, event_present=event_present,
                               action_duration=features.get("action_duration", features.get("dt", pooled.new_ones(len(pooled)))) * action_present)
        # V2 dataclasses remain loadable; V3 core consumes these additional contracts.
        obs.fresh_observation = fresh
        obs.context_available = masks["text"].any(1)
        obs.action_present = action_present
        return {"observation": obs, "raw_aff": raw_aff, "valid": valid,
                "hidden": hidden, "spans": spans, "masks": masks, "masked": masked,
                "reconstruction": {m: self.reconstruct[m](hidden[:, spans[m]]) for m in MODES}}

    def forward(self, features, subset=None, corruption=None):
        return self.encode(features, subset, corruption)["observation"]

    def sample_corruption(self, features, mask_ratio=.4, generator=None):
        if not 0 < mask_ratio < 1:
            raise ValueError("mask_ratio must lie between zero and one")
        output = {}
        for mode in MODES:
            _, valid = self._mode_features(features, mode)
            output[mode] = torch.zeros_like(valid)
            if mode == "prosody":
                # One acoustic atom is sampled once. Independently sampling its
                # audio/prosody copies would inflate the requested mask ratio.
                continue
            # Consecutive time/word spans, not hidden-vector dimensions.
            for row in range(len(valid)):
                indices = valid[row].nonzero().flatten()
                if not len(indices):
                    continue
                count = max(1, round(len(indices) * mask_ratio))
                start = int(torch.randint(max(1, len(indices) - count + 1), (), generator=generator))
                output[mode][row, indices[start:start + count]] = True
        return self.synchronize_corruption(features, output)

    @staticmethod
    def _global_affect(affect, valid, domains, distributed):
        """Differentiable variable-size gather with unconditional collectives.

        Every rank participates, even when its modality subset has no valid rows.
        The loss must be called once on ALL ranks at the same training boundary.
        Per-packet generation calls therefore default to distributed=False.
        """
        import torch.distributed as dist
        if not distributed or not dist.is_available() or not dist.is_initialized():
            return affect, valid, domains
        from torch.distributed.nn.functional import all_gather
        world = dist.get_world_size()
        size = torch.tensor([len(affect)], device=affect.device, dtype=torch.long)
        sizes = [torch.empty_like(size) for _ in range(world)]
        dist.all_gather(sizes, size)
        width = max(1, max(int(item.item()) for item in sizes))
        padded = F.pad(affect, (0, 0, 0, width-len(affect)))
        metadata = torch.stack([valid.long(), domains.long()], -1)
        metadata = F.pad(metadata, (0, 0, 0, width-len(affect)), value=0)
        all_metadata = [torch.empty_like(metadata) for _ in range(world)]
        dist.all_gather(all_metadata, metadata)
        # Returning all rows keeps a graph dependency on this collective even
        # when the globally valid set is empty or has just one sample.
        gathered = torch.cat(all_gather(padded), 0)
        info = torch.cat(all_metadata, 0)
        return gathered, info[:, 0].bool(), info[:, 1]

    def _unit_statistics(self, values):
        # Autocast would otherwise quantize the covariance matrix product even
        # though the input representations have already been promoted to FP32.
        with torch.autocast(device_type=values.device.type, enabled=False):
            return self._unit_statistics_float(values.float())

    def _unit_statistics_float(self, values):
        zero = values.sum() * 0.
        samples = len(values)
        if samples < 2:
            return {"variance": zero, "covariance": zero, "uniformity": zero,
                    "mean": zero, "total": zero, "unit_std": zero,
                    "samples": zero.detach().new_tensor(samples)}
        dimension = values.shape[-1]
        center = values.mean(0)
        centered = values - center
        target = self.config.variance_scale / math.sqrt(dimension)
        std = (centered.square().mean(0) + (target * 1e-3)**2).sqrt()
        # Relative deficiency is approximately one at collapse, independent of
        # affect dimension. It cannot be reduced by increasing raw-affect norm.
        variance = F.relu(1. - std / target).mean()
        covariance = dimension * (centered.T @ centered) / (samples-1)
        off_diagonal = covariance - torch.diag_embed(covariance.diag())
        covariance_loss = off_diagonal.square().sum() / dimension
        pair_distance = torch.pdist(values).square()
        uniformity = torch.exp(-self.config.uniformity_temperature * pair_distance).mean()
        mean_loss = center.square().sum()
        total = (self.config.variance_weight * variance + self.config.covariance_weight * covariance_loss
                 + self.config.uniformity_weight * uniformity + self.config.mean_weight * mean_loss)
        return {"variance": variance, "covariance": covariance_loss, "uniformity": uniformity,
                "mean": mean_loss, "total": total, "unit_std": std.mean(),
                "samples": zero.detach().new_tensor(samples)}

    def representation_from_affect(self, affect, valid=None, domain_id=None, distributed=False):
        """Regularize clean final affect; callers can batch generation records.

        Pass representations from encode_clean, not a dropout or masked view.
        Domain-local constraints prevent domain identity alone from satisfying
        the pooled diversity requirement. No collective is used by default.
        """
        if affect.ndim != 2 or affect.shape[-1] != self.config.affect_dim:
            raise ValueError("Clean affect must be [N,affect_dim]")
        if valid is None:
            valid = torch.ones(len(affect), dtype=torch.bool, device=affect.device)
        if domain_id is None:
            domain_id = torch.zeros(len(affect), dtype=torch.long, device=affect.device)
        if valid.shape != (len(affect),) or domain_id.shape != (len(affect),):
            raise ValueError("Affect validity/domain metadata must be [N]")
        # Explicit float32 avoids AMP rounding a small but real spread to zero.
        valid = valid.bool() & torch.isfinite(affect).all(-1)
        safe_affect = torch.where(valid[:, None], affect.float(), torch.zeros_like(affect, dtype=torch.float32))
        unit = F.normalize(safe_affect, dim=-1, eps=1e-6)
        unit, valid, domains = self._global_affect(unit, valid.bool(), domain_id, distributed)
        values = unit[valid]
        result = self._unit_statistics(values)
        # Keep invalid rows connected to the differentiable gather as well.
        total = result["total"] + unit.sum() * 0.
        domain_losses = []
        for domain in range(self.config.num_domains):
            selected = values[domains[valid] == domain]
            result[f"domain{domain}_samples"] = total.detach().new_tensor(len(selected))
            if len(selected) >= 2:
                statistics = self._unit_statistics(selected)
                result[f"domain{domain}_variance"] = statistics["variance"]
                domain_losses.append(statistics["total"])
        if domain_losses:
            total = .5 * total + .5 * torch.stack(domain_losses).mean()
        result["total"] = total
        return result

    def representation_loss(self, features, subset=None, distributed=False):
        clean = self.encode_clean(features, subset)
        domain = features.get("domain_id", torch.zeros(len(clean["valid"]),
                              device=clean["valid"].device, dtype=torch.long))
        return self.representation_from_affect(clean["observation"].aff, clean["valid"], domain, distributed)

    def _summary_targets(self, features):
        result = {}
        for mode in MODES:
            values, mask = self._mode_features(features, mode)
            mask = mask & torch.isfinite(values).all(-1)
            values = torch.nan_to_num(values.detach().float())
            # Prosody already has versioned per-feature units. Normalizing each
            # row across its dimensions would erase absolute energy/amplitude cues.
            values = values.clamp(-10, 10) if mode == "prosody" else F.layer_norm(values, (values.shape[-1],))
            pooled = (values * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
            projection = getattr(self, "summary_projection_" + mode).float()
            result[mode] = (pooled @ projection, mask.any(-1))
        return result

    def masked_loss(self, features, mask_ratio=.4, corruption=None, subset=None,
                    teacher=None, distributed=False, teacher_subset="AVT", masking=None):
        """Masked token auxiliary + affect-bottleneck summaries + clean diversity.

        DDP A0 must explicitly pass distributed=True once per synchronized step.
        Default False is safe for independently sized generation/TBPTT loops.
        """
        student, mask_diagnostics = features, {}
        if masking:
            if corruption is not None or subset not in (None, "AVT"):
                raise ValueError("Mixed masking owns the student modality selection; do not stack independent dropout")
            from emotion_ssm.models.context_masking import sample_context_mask
            student, corruption, mask_diagnostics = sample_context_mask(self, features, masking)
        else:
            corruption = self.sample_corruption(features, mask_ratio) if corruption is None else corruption
        encoded = self.encode(student, subset, corruption)
        zero = encoded["raw_aff"].sum() * 0
        losses, token_losses, summary_losses = {}, [], []
        targets = self._summary_targets(features)
        for mode in MODES:
            chosen = encoded["masked"][mode]
            if chosen.any():
                target = self._mode_features(features, mode)[0].detach().float()
                target = target.clamp(-10, 10) if mode == "prosody" else F.layer_norm(target, (target.shape[-1],))
                losses["mask_" + mode] = F.smooth_l1_loss(encoded["reconstruction"][mode][chosen].float(), target[chosen])
                token_losses.append(losses["mask_" + mode])
            else:
                losses["mask_" + mode] = zero
            target, target_valid = targets[mode]
            valid = target_valid & encoded["valid"]
            if valid.any():
                prediction = self.affect_summary_heads[mode](encoded["observation"].aff.to(self.query.dtype))
                losses["summary_" + mode] = F.smooth_l1_loss(prediction[valid].float(), target[valid])
                summary_losses.append(losses["summary_" + mode])
            else:
                losses["summary_" + mode] = zero
        with torch.no_grad():
            source = self if teacher is None else teacher
            target = source.encode_clean(features, teacher_subset)
        valid = target["valid"] & encoded["valid"]
        losses["affect_distillation"] = ((1 - F.cosine_similarity(
            encoded["observation"].aff[valid], target["observation"].aff[valid].detach())).mean()
            if valid.any() else zero)
        # This is the ONLY view used for variance/covariance/uniformity. It has
        # no corruption or dropout, so copied inputs cannot manufacture spread.
        regularization = self.representation_loss(features, subset, distributed)
        losses.update({key: value for key, value in regularization.items() if key != "total"})
        losses["representation_total"] = regularization["total"]
        losses["reconstruction"] = torch.stack(token_losses).mean() if token_losses else zero
        losses["affect_summary"] = torch.stack(summary_losses).mean() if summary_losses else zero
        losses["total"] = (self.config.token_reconstruction_weight * losses["reconstruction"]
                           + self.config.affect_summary_weight * losses["affect_summary"]
                           + self.config.affect_distillation_weight * losses["affect_distillation"]
                           + losses["representation_total"])
        losses.update({"masking_"+key: value for key, value in mask_diagnostics.items()})
        return losses

    def decode_affect(self, affect):
        return {"emotion_logits": self.emotion_head(affect),
                "intensity": F.softplus(self.intensity_head(affect)).squeeze(-1),
                "vad": self.vad_head(affect).tanh()}

    def supervised_loss(self, features, labels, subset=None):
        encoded = self.encode(features, subset)
        out = self.decode_affect(encoded["observation"].aff)
        zero = encoded["raw_aff"].sum() * 0
        losses = {}
        valid = encoded["valid"] & labels.get("endpoint_mask", encoded["valid"]).bool()
        emotion = labels.get("emotion", torch.full_like(valid, -1, dtype=torch.long))
        mask = valid & (emotion >= 0) & (emotion < 7)
        losses["emotion"] = F.cross_entropy(out["emotion_logits"][mask], emotion[mask].long()) if mask.any() else zero
        intensity = labels.get("intensity", zero.expand(len(valid)))
        mask = valid & labels.get("intensity_mask", torch.zeros_like(valid)).bool()
        losses["intensity"] = F.mse_loss(out["intensity"][mask], intensity[mask]) if mask.any() else zero
        mask = labels.get("vad_mask", torch.zeros(len(valid), 3, device=valid.device, dtype=torch.bool)).bool() & valid[:, None]
        losses["vad"] = F.mse_loss(out["vad"][mask], labels["vad"][mask]) if mask.any() else zero
        losses["total"] = sum(losses.values())
        return losses

    @torch.no_grad()
    def diagnostics(self, features):
        result = {}
        for subset in SUBSETS:
            encoded = self.encode_clean(features, subset)
            aff = encoded["observation"].aff[encoded["valid"]]
            result[subset] = {"samples": len(aff), "std": float(aff.std(0, unbiased=False).mean()) if len(aff) else 0.,
                              "cosine": float(((aff @ aff.T).sum() - len(aff)) / (len(aff) * (len(aff)-1))) if len(aff) > 1 else None}
        return result
