"""One-second generation; persistent state owns only already available inputs."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import nn

from emotion_ssm.schema import DyadicState


@dataclass
class StreamState:
    session_id: str
    roles: tuple
    time: float
    emotion: Optional[DyadicState] = None
    history: list = field(default_factory=list)
    words: dict = field(default_factory=dict)

    def detach(self):
        return StreamState(self.session_id, self.roles, self.time,
                           None if self.emotion is None else self.emotion.detach(),
                           [{k: v.detach() if torch.is_tensor(v) else v for k, v in item.items()}
                            for item in self.history], dict(self.words))


def normalize_audio(value):
    return (value-value.mean(-1, keepdim=True)) / value.std(-1, keepdim=True, unbiased=False).clamp_min(1e-6)


class StreamingAvatar(nn.Module):
    variants = ("none", "affect", "self", "dyadic")

    def __init__(self, generator, observer, state_model, features=None, variant="dyadic",
                 fps=25, history_seconds=3, projector=None, state_loss_weight=0., speech_rms_threshold=1e-4):
        super().__init__()
        if variant not in self.variants:
            raise ValueError(f"Unknown conditioning variant {variant}")
        self.generator, self.observer, self.state_model, self.features = generator, observer, state_model, features
        self.variant, self.fps = variant, fps
        self.history_frames = int(round(history_seconds * fps))
        self.projector, self.state_loss_weight = projector, state_loss_weight
        self.speech_rms_threshold = speech_rms_threshold
        if state_loss_weight and projector is None:
            raise ValueError("State consistency requires a calibrated fixed projector")
        self.configure_trainable()

    def configure_trainable(self):
        # All comparisons expose exactly the same baseline parameters.
        self.generator.baseline.requires_grad_(True)
        for name in ("audio_encoder1", "audio_encoder2"):
            encoder = getattr(self.generator.baseline.joint_encoder, name, None)
            if encoder is not None:
                encoder.feature_extractor.requires_grad_(False)
        self.generator.film.requires_grad_(self.variant != "none")
        for module in (self.observer, self.state_model, self.features, self.projector):
            if module is not None:
                module.requires_grad_(False)
                module.eval()

    def train(self, mode=True):
        super().train(mode)
        for module in (self.observer, self.state_model, self.features, self.projector):
            if module is not None:
                module.eval()
        return self

    def initial_state(self, session_id, roles, time=0.):
        if len(roles) != 2 or roles[0] == roles[1]:
            raise ValueError("A stream requires two distinct ordered roles")
        return StreamState(str(session_id), tuple(roles), float(time))

    def _observation(self, packet, state, role_index):
        role = state.roles[role_index]
        prefix = "target" if role_index == 0 else "partner"
        cached = packet.get(prefix + "_features")
        if cached is None:
            if self.features is None:
                raise ValueError("Provide timed features or a complete feature extractor")
            wave = packet[prefix + "_audio"]
            active = packet.get(prefix + "_speech_active")
            if active is None:
                active = bool(wave.float().square().mean().sqrt() > self.speech_rms_threshold)
            cached = self.features(wave, list(state.words.values()), float(packet["time"]), state.time,
                                   role, bool(active),
                                   packet.get(prefix + "_audio_length"))
        values = {k: v.clone() if torch.is_tensor(v) else v for k, v in cached.items()}
        if "available_at" in values and float(values.pop("available_at")) > float(packet["time"]):
            raise ValueError("Cached features contain future evidence")
        values["modality_mask"] = values["modality_mask"].clone()
        # Hard boundary: target/current ground truth and supplied target visual
        # features cannot enter the observer, even if a caller supplies them.
        if role_index == 0:
            for key in ("flame", "visual_token", "face", "face_frame_mask", "face_confidence"):
                values.pop(key, None)
            values["modality_mask"][:, 1] = False
        elif "partner_au" in packet:
            values["face"] = packet["partner_au"]
            values["face_frame_mask"] = packet["partner_au_mask"]
            values["face_confidence"] = packet.get("partner_au_confidence", values["face_frame_mask"].float())
            values["modality_mask"][:, 1] = values["face_frame_mask"].any(-1)
        else:
            values["flame"] = packet["partner_blendshape"]
            values["flame_mask"] = packet.get("partner_visual_mask", torch.ones(
                values["flame"].shape[:2], dtype=torch.bool, device=values["flame"].device))
            values["modality_mask"][:, 1] = values["flame_mask"].any(-1)
        values["action_duration"] = values["audio"].new_tensor([float(packet["time"]) - state.time])
        return self.observer(values, use_flame="partner_au" not in packet), values

    def forward(self, packet, state=None):
        session_id, roles = str(packet["session_id"]), tuple(packet["roles"])
        if state is None:
            state = self.initial_state(session_id, roles, packet.get("start_time", 0.))
        if state.session_id != session_id or state.roles != roles:
            raise ValueError("Session/role change requires an explicit stream reset")
        now = float(packet["time"])
        dt = now - state.time
        if not 0 < dt <= 1.000001:
            raise ValueError("Packets must advance by at most one second; represent gaps as missing packets")
        count = packet["partner_blendshape"].shape[1]
        if count != round(dt * self.fps):
            raise ValueError("Packet frame count and elapsed time disagree")
        for name in ("target_audio", "partner_audio"):
            if packet[name].shape != (1, round(dt * 16000)):
                raise ValueError("A packet contains one session and exactly its elapsed audio samples")
        packet = dict(packet)
        for prefix in ("target", "partner"):
            wave = packet[prefix+"_audio"]
            length = int(packet.get(prefix+"_audio_length", wave.shape[1]))
            if not 0 <= length <= wave.shape[1]:
                raise ValueError("Invalid audio valid length")
            valid_audio = torch.arange(wave.shape[1], device=wave.device)[None] < length
            if not torch.isfinite(wave[valid_audio]).all():
                raise ValueError("Valid audio contains non-finite samples")
            packet[prefix+"_audio"] = torch.where(valid_audio, wave, 0.)
        visual = packet["partner_blendshape"]
        mask = packet.get("partner_visual_mask", torch.ones(visual.shape[:2], dtype=torch.bool, device=visual.device))
        if mask.shape != visual.shape[:2]:
            raise ValueError("Visual mask and frame dimensions disagree")
        mask = mask.bool() & torch.isfinite(visual).all(-1)
        packet["partner_visual_mask"] = mask
        packet["partner_blendshape"] = torch.where(mask[..., None], visual, 0.)
        # Copy state metadata: callers can replay a prefix without mutation.
        words = dict(state.words)
        from emotion_ssm.data.protocol import visible_words
        for word in visible_words(packet.get("words", []), now):
            if str(word["role"]) not in roles:
                raise ValueError("Text role does not belong to this session")
            key = str(word.get("id", (word["role"], word["start"], word["end"], word["text"])))
            if key not in words:
                word = dict(word)
                word["available_at"] = max(float(word["available_at"]), now)
                words[key] = word
        working = StreamState(state.session_id, state.roles, state.time, state.emotion, state.history, words)
        with torch.no_grad():
            first, first_values = self._observation(packet, working, 0)
            second, second_values = self._observation(packet, working, 1)
            emotion = state.emotion
            if emotion is None:
                ids = torch.full((len(first.aff), 2), -1, dtype=torch.long, device=first.aff.device)
                emotion = self.state_model.initialize(ids)
            emotion = self.state_model.observe(emotion, (first, second), first.aff.new_tensor([dt]),
                                                enable_partner=self.variant == "dyadic")
            slow = torch.cat([emotion.z[:, 0], emotion.z[:, 1], emotion.relation], -1)
            if self.variant in ("none", "affect"):
                slow = torch.zeros_like(slow)
            elif self.variant == "self":
                slow = torch.cat([emotion.z[:, 0], torch.zeros_like(emotion.z[:, 1]),
                                  torch.zeros_like(emotion.relation)], -1)
            context = torch.cat([first.aff, second.aff, slow], -1)
        item = {"target_audio": packet["target_audio"], "partner_audio": packet["partner_audio"],
                "partner_blendshape": packet["partner_blendshape"],
                "context": context[:, None].expand(-1, count, -1)}
        pieces = state.history + [item]
        joined = {k: torch.cat([part[k] for part in pieces], dim=1) for k in item}
        generated = self.generator(normalize_audio(joined["target_audio"]), normalize_audio(joined["partner_audio"]),
                                   joined["partner_blendshape"], joined["context"], self.variant != "none")[:, -count:]
        keep_frames = min(self.history_frames, joined["partner_blendshape"].shape[1])
        history = []
        if keep_frames:
            keep_samples = round(keep_frames / self.fps * 16000)
            history = [{k: v[:, -(keep_samples if k.endswith("audio") else keep_frames):].detach()
                        for k, v in joined.items()}]
        next_state = StreamState(session_id, roles, now, emotion.detach(), history, words)
        diagnostics = {"target_aff": first.aff, "partner_aff": second.aff,
                       "target_modalities": first.modality_mask, "partner_modalities": second.modality_mask,
                       "context": context, "timestamp": now}
        if packet.get("collect_diagnostics", False):
            domain = self.construction_info["adapter_source"]
            diagnostics["subsets"] = {role: self.observer.diagnose(values, domain, "partner_au" not in packet)
                for role, values in (("target", first_values), ("partner", second_values))}
        return generated, next_state, diagnostics


class StreamSessions:
    """Own persistent state for independent sessions; reset is explicit."""
    def __init__(self, model):
        self.model, self.states = model, {}

    def reset(self, session_id=None):
        if session_id is None:
            self.states.clear()
        else:
            self.states.pop(str(session_id), None)

    def __call__(self, packet):
        key = str(packet["session_id"])
        result = self.model(packet, self.states.get(key))
        self.states[key] = result[1]
        return result
