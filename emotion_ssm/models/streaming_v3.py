"""Differentiable, one-second avatar streaming with explicit TBPTT boundaries.

The module owns causal input/state handling only. Training permissions, freezing
and graph truncation belong to the trainer; inference callers use no_grad.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass, replace
from contextlib import contextmanager
from typing import Any

import torch
from torch import nn

from emotion_ssm.data.protocol import visible_words
from emotion_ssm.data.packets_v3 import AUDIO_HISTORY_SECONDS, AUDIO_SEQUENCE_FIELDS, merge_audio_history, collate_role_features
from emotion_ssm.models.streaming import normalize_audio
from emotion_ssm.schema import EventObservation


def checkpoint_rng_contexts():
    """Replay the NumPy SpecAugment RNG as well as checkpoint's Torch RNG.

    The repository's wav2vec implementation draws masks with NumPy. Torch's
    preserve_rng_state does not cover those masks. Every forward invocation owns
    separate state, and recomputation restores the caller's current RNG on exit.
    """
    import numpy as np
    saved = {}

    @contextmanager
    def forward_context():
        saved["numpy"] = np.random.get_state()
        yield

    @contextmanager
    def recompute_context():
        current = np.random.get_state()
        np.random.set_state(saved["numpy"])
        try:
            yield
        finally:
            np.random.set_state(current)

    return forward_context(), recompute_context()


def map_tensors(value, function):
    """Move/detach nested persistent state without losing its dataclass type."""
    if torch.is_tensor(value):
        return function(value)
    if is_dataclass(value) and not isinstance(value, type):
        return type(value)(**{f.name: map_tensors(getattr(value, f.name), function) for f in fields(value)})
    if isinstance(value, dict):
        return {key: map_tensors(item, function) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(map_tensors(item, function) for item in value)
    if isinstance(value, list):
        return [map_tensors(item, function) for item in value]
    return value


@dataclass
class StreamStateV3:
    session_id: str
    roles: tuple
    time: float
    emotion: Any = None
    history: list = field(default_factory=list)
    words: dict = field(default_factory=dict)
    event_ids: frozenset = field(default_factory=frozenset)
    audio_history: dict = field(default_factory=dict)
    boundary_for_loss: object = None

    def detach(self):
        return map_tensors(self, lambda value: value.detach())

    def to(self, device):
        return map_tensors(self, lambda value: value.to(device))


class StreamingAvatarV3(nn.Module):
    variants = ("none", "affect", "self", "dyadic")

    def __init__(self, generator, observer, state_model, features=None, variant="dyadic",
                 fps=25, history_seconds=3, max_text_words=256, speech_rms_threshold=1e-4,
                 condition_config=None):
        super().__init__()
        if variant not in self.variants:
            raise ValueError(f"Unknown conditioning variant: {variant}")
        if fps != 25 or history_seconds < 0 or history_seconds > 3:
            raise ValueError("v3 requires 25 FPS and at most three seconds of generator history")
        self.generator, self.observer = generator, observer
        self.state_model, self.features = state_model, features
        self.variant, self.fps = variant, fps
        self.history_frames = round(history_seconds * fps)
        self.max_text_words = int(max_text_words)
        self.speech_rms_threshold = float(speech_rms_threshold)
        self.generator_checkpointing = False
        self.condition_router = None
        if condition_config is not None:
            from emotion_ssm.models.condition_router import ConditionRouter
            self.condition_router = ConditionRouter(state_model.context_dim, state_model.context_layout(), condition_config)
            if self.condition_router.mode == 'actual_semantic' and variant not in ('self','dyadic'):
                raise ValueError('Actual state semantics require self or dyadic state formation')
        expected = getattr(state_model, "context_dim", None)
        actual = getattr(generator, "context_dim", expected)
        if expected is not None and actual != expected:
            raise ValueError(f"Generator context dimension {actual} does not match state core {expected}")

    def initial_state(self, session_id, roles, time=0.):
        if len(roles) != 2 or roles[0] == roles[1]:
            raise ValueError("A session needs two distinct ordered roles")
        return StreamStateV3(str(session_id), tuple(roles), float(time))

    def _validate_packet(self, packet, state):
        now = float(packet["time"])
        dt = now - state.time
        if not 0 < dt <= 1.000001:
            raise ValueError("Packets advance by at most one second; emit missing packets for longer gaps")
        count = round(dt * self.fps)
        if count < 1 or abs(count / self.fps - dt) > 1e-5:
            raise ValueError("Elapsed time must correspond to a whole number of frames")
        packet = dict(packet)
        for prefix in ("target", "partner"):
            wave = packet[prefix + "_audio"]
            if wave.shape != (1, round(dt * 16000)):
                raise ValueError("A packet contains one session and only its newly elapsed audio samples")
            length = int(packet.get(prefix + "_audio_length", wave.shape[1]))
            if not 0 <= length <= wave.shape[1]:
                raise ValueError("Audio valid length exceeds the supplied block")
            valid = torch.arange(wave.shape[1], device=wave.device)[None] < length
            if not torch.isfinite(wave[valid]).all():
                raise ValueError("Valid audio contains nonfinite values")
            packet[prefix + "_audio"] = torch.where(valid, wave, 0.)
        visual = packet["partner_blendshape"]
        if visual.shape != (1, count, 56):
            raise ValueError("Partner FLAME must be [1,new_frames,56]")
        valid = packet.get("partner_visual_mask", torch.ones(visual.shape[:2], dtype=torch.bool, device=visual.device))
        if valid.shape != visual.shape[:2]:
            raise ValueError("Visual mask and FLAME frames disagree")
        valid = valid.bool() & torch.isfinite(visual).all(-1)
        packet["partner_visual_mask"] = valid
        packet["partner_blendshape"] = torch.where(valid[..., None], visual, 0.)
        return packet, now, dt, count

    def _words(self, packet, state):
        words = dict(state.words)
        for original in visible_words(packet.get("words", []), float(packet["time"])):
            if str(original["role"]) not in tuple(str(role) for role in state.roles):
                raise ValueError("Text role does not belong to this session")
            key = str(original.get("id", (original["role"], original["start"], original["end"], original["text"])))
            if key not in words:
                word = dict(original)
                # Preserve within-block ASR arrival times. Only a message whose
                # declared arrival predates an already emitted block is late;
                # that message first becomes usable now and never rewrites it.
                available=float(word["available_at"])
                word["available_at"] = float(packet["time"]) if available<=state.time else available
                words[key] = word
        if len(words) > self.max_text_words:
            ordered = sorted(words, key=lambda key: (float(words[key]["available_at"]), key))
            words = {key: words[key] for key in ordered[-self.max_text_words:]}
        return words

    def _observation(self, packet, state, role_index, dt, consumed):
        prefix = "target" if role_index == 0 else "partner"
        cached = packet.get(prefix + "_features")
        if cached is None:
            if self.features is None:
                raise ValueError("Provide causal token features or a complete streaming feature extractor")
            waveform = packet[prefix + "_audio"]
            active = packet.get(prefix + "_speech_active")
            if active is None:
                length=int(packet.get(prefix+"_audio_length",waveform.shape[1]))
                active = bool(length>0 and waveform[:,:length].float().square().mean().sqrt() > self.speech_rms_threshold)
            cached = self.features(waveform, list(state.words.values()), float(packet["time"]), state.time,
                                   state.roles[role_index], bool(active), packet.get(prefix + "_audio_length"))
            local={key:value[0] if torch.is_tensor(value) else value for key,value in cached.items()}
            previous=state.audio_history.get(str(state.roles[role_index]))
            seconds=float(getattr(self.features,"source",{}).get("audio_history_seconds",AUDIO_HISTORY_SECONDS))
            local=merge_audio_history(previous,local,seconds)
            cached=collate_role_features([local],waveform.device)
        if packet.get('continuous_timeline',False) and not bool(cached.get('audio_history_complete',False)):
            local={key:value[0] for key,value in cached.items() if torch.is_tensor(value)}
            previous=state.audio_history.get(str(state.roles[role_index]))
            local=merge_audio_history(previous,local,AUDIO_HISTORY_SECONDS)
            cached={**cached,**{key:value[None] for key,value in local.items()}}
        values = dict(cached)
        available = values.pop("available_at", values.get("now", packet["time"]))
        if float(available) > float(packet["time"]) + 1e-7:
            raise ValueError("Cached features contain future evidence")
        for mode in ("audio", "prosody", "au", "flame", "text"):
            times, valid = values.get(mode + "_times"), values.get(mode + "_mask")
            if times is not None and valid is not None and (times[valid.bool()] > float(packet["time"]) + 1e-6).any():
                raise ValueError("A token has a future availability timestamp")
        # The observer sees modality availability, but state updates below receive
        # a separate fresh-evidence mask. Old context is useful without becoming
        # a fresh event or one extra second of behaviour.
        mask = values["modality_mask"].clone().bool()
        if mask.shape != (1, 3):
            raise ValueError("Modality mask must be [1,3] in audio/visual/text order")
        device = mask.device
        audio_present = packet.get(prefix + "_speech_active", True)
        if not bool(torch.as_tensor(audio_present).any()) or int(packet.get(prefix + "_audio_length", packet[prefix + "_audio"].shape[1])) == 0:
            if "audio_mask" in values:
                # Silence suppresses new sound, while already observed acoustic
                # history remains readable. It must not count as fresh action.
                old=values["audio_times"].double()<=state.time+1e-9
                values["audio_mask"] = values["audio_mask"].bool() & old
                if "prosody_mask" in values and values["prosody_mask"].shape==old.shape:
                    values["prosody_mask"] = values["prosody_mask"].bool() & old
                mask[:,0]=values["audio_mask"].any(-1)
            else:
                mask[:, 0] = False
        values["query_role"] = torch.full((1,), role_index, dtype=torch.long, device=device)
        values["target_role"] = values["query_role"]
        if role_index == 0:
            for mode in ("flame", "au"):
                if mode + "_tokens" in values:
                    values[mode + "_tokens"] = torch.zeros_like(values[mode + "_tokens"])
                    values[mode + "_mask"] = torch.zeros_like(values[mode + "_mask"], dtype=torch.bool)
            for key in ("flame", "visual_token", "visual_tokens", "face", "face_tokens",
                        "face_frame_mask", "face_confidence", "au", "visual_mask"):
                values.pop(key, None)
            mask[:, 1] = False
        elif "partner_au" in packet:
            au = packet["partner_au"]
            valid = packet.get("partner_au_mask", torch.isfinite(au).all(-1))
            valid = valid.bool() & torch.isfinite(au).all(-1)
            values["face"] = values["au"] = torch.where(valid[..., None], au, 0.)
            values["au_tokens"] = values["au"]
            values["face_frame_mask"] = values["au_mask"] = valid
            values["au_times"] = torch.linspace(state.time + dt / au.shape[1], float(packet["time"]),
                                                 au.shape[1], device=au.device)[None]
            if "flame_mask" in values:
                values["flame_mask"] = torch.zeros_like(values["flame_mask"], dtype=torch.bool)
            values["face_confidence"] = packet.get("partner_au_confidence", valid.float())
            mask[:, 1] = valid.any(-1)
        else:
            values["flame"] = packet["partner_blendshape"]
            values["flame_tokens"] = values["flame"]
            values["flame_mask"] = packet["partner_visual_mask"]
            values["flame_times"] = torch.linspace(state.time + dt / values["flame"].shape[1], float(packet["time"]),
                                                    values["flame"].shape[1], device=values["flame"].device)[None]
            if "au_mask" in values:
                values["au_mask"] = torch.zeros_like(values["au_mask"], dtype=torch.bool)
            mask[:, 1] = values["flame_mask"].any(-1)
        values["modality_mask"] = mask
        event_present = torch.as_tensor(values.get("event_present", False), device=device).reshape(1).bool() & mask[:, 2]
        event_ids = values.get("event_ids", packet.get(prefix + "_event_ids"))
        if event_ids is None and "event_id" in values:
            event_ids = [values["event_id"]]
        if event_ids is not None:
            if not isinstance(event_ids, (list, tuple, set)):
                event_ids = [event_ids]
            keys = {f"{state.roles[role_index]}:{key}" for key in event_ids}
            unseen = keys - consumed
            event_present = event_present & bool(unseen)
            if bool(event_present.any()):
                consumed.update(unseen)
        # For cached one-second features without IDs, availability timestamp is
        # the identity of that new text fragment; it cannot be injected twice.
        elif bool(event_present.any()):
            key = f"{state.roles[role_index]}:cache:{float(available):.9f}"
            event_present = event_present & (key not in consumed)
            consumed.add(key)
        values["event_present"] = event_present
        fresh = values.get("fresh_observation", mask).clone().bool() & mask
        if "fresh_observation" not in values:
            fresh[:, 2] = event_present
        # Current raw vision replaces any cached visual freshness declaration.
        fresh[:, 1] = mask[:, 1]
        for index, key in ((0, prefix + "_speech_active"),):
            if key in packet:
                fresh[:, index] &= torch.as_tensor(packet[key], device=device).reshape(1).bool()
        if int(packet.get(prefix + "_audio_length", packet[prefix + "_audio"].shape[1])) == 0:
            fresh[:, 0] = False
        values["fresh_mask"] = fresh
        values["fresh_observation"] = fresh
        active_action = fresh[:, :2].any(-1)
        declared_duration = torch.as_tensor(values.get("action_duration", dt), device=device,
                                            dtype=packet[prefix + "_audio"].dtype).reshape(1)
        if (declared_duration < 0).any() or (declared_duration > dt + 1e-5).any():
            raise ValueError("Declared behaviour duration exceeds the actually observed interval")
        audio_duration = declared_duration.clamp(0, dt) * fresh[:, 0]
        visual_mask = values.get("au_mask" if "partner_au" in packet else "flame_mask")
        visual_duration = (visual_mask.float().mean(-1) * dt if role_index == 1 and visual_mask is not None
                           else audio_duration.new_zeros(1))
        values["action_present"] = active_action
        values["action_duration"] = torch.maximum(audio_duration, visual_duration) * active_action
        dtype = next(self.observer.parameters()).dtype
        for key, value in list(values.items()):
            if key.endswith("_tokens") and torch.is_tensor(value):
                values[key] = value.to(dtype=dtype)
        output = self.observer(values)
        if not isinstance(output, EventObservation) and hasattr(output, "select"):
            output = output.select(0)
        if not isinstance(output, EventObservation):
            raise TypeError("The v3 observer must return EventObservation or an output with select(0)")
        output = replace(output, modality_mask=mask, event_present=event_present,
                         event=output.event * event_present[:, None], action_duration=values["action_duration"])
        output.fresh_observation = fresh
        output.action_present = values["action_present"]
        output.context_available = mask[:, 2]
        return output, values

    def forward(self, packet, state=None, *, condition_override=None, diagnostic=False, observe_only=False,
                return_generator_inputs=False):
        session_id, roles = str(packet["session_id"]), tuple(packet["roles"])
        if state is None:
            state = self.initial_state(session_id, roles, packet.get("start_time", 0.))
        if state.session_id != session_id or state.roles != roles:
            raise ValueError("A session or role change requires an explicit reset")
        packet, now, dt, count = self._validate_packet(packet, state)
        words = self._words(packet, state)
        working = replace(state, words=words)
        consumed = set(state.event_ids)
        first, first_values = self._observation(packet, working, 0, dt, consumed)
        second, second_values = self._observation(packet, working, 1, dt, consumed)
        pair = (first, second)
        emotion = state.emotion
        if emotion is None:
            emotion = self.state_model.initialize(len(first.aff), first.aff.device, first.aff.dtype)
        if self.variant in ("self", "dyadic"):
            updated = self.state_model.advance(emotion, pair, dt,
                                               enable_partner=self.variant == "dyadic")
            emotion = updated.posterior if hasattr(updated, "posterior") else updated
        context = self.state_model.context(emotion, pair, variant=self.variant)
        film_enabled = self.variant != "none"
        if self.condition_router is not None:
            semantic = None
            if self.condition_router.mode == "actual_semantic":
                from emotion_ssm.models.condition_router import semantic_code
                from torch.nn import functional as F
                semantic = semantic_code(self.observer.decode_affect(F.normalize(emotion.z[:, 0].float(), dim=-1)),
                                         self.condition_router.heads)
            routed_context, film_enabled = self.condition_router(context, semantic, condition_override, diagnostic)
        else:
            if condition_override is not None:
                raise ValueError("Ordinary deployment cannot accept target condition overrides")
            routed_context = context
        if routed_context.ndim == 2:
            frame_context = routed_context[:, None].expand(-1, count, -1)
        elif routed_context.ndim == 3 and routed_context.shape[1] == count:
            frame_context = routed_context
        else:
            raise ValueError("State context must be [B,D] or time-aligned [B,new_frames,D]")
        item = {"target_audio": packet["target_audio"], "partner_audio": packet["partner_audio"],
                "partner_blendshape": packet["partner_blendshape"], "context": frame_context}
        pieces = state.history + [item]
        joined = {key: torch.cat([part[key] for part in pieces], dim=1) for key in item}
        arguments = (normalize_audio(joined["target_audio"]), normalize_audio(joined["partner_audio"]),
                     joined["partner_blendshape"], joined["context"])
        if observe_only:
            generated = None
        elif self.generator_checkpointing and self.training and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint
            generated = checkpoint(self.generator, *arguments, film_enabled, use_reentrant=False,
                                   preserve_rng_state=True, context_fn=checkpoint_rng_contexts)[:, -count:]
        else:
            generated = self.generator(*arguments, film_enabled)[:, -count:]
        if generated is not None and generated.shape != (1, count, 56):
            raise ValueError("Generator must return the aligned new FLAME frames")
        keep_frames = min(self.history_frames, joined["partner_blendshape"].shape[1])
        history = []
        if keep_frames:
            keep_samples = round(keep_frames / self.fps * 16000)
            history = [{key: value[:, -(keep_samples if key.endswith("audio") else keep_frames):]
                        for key, value in joined.items()}]
        acoustic_history={}
        for role,value in zip(roles,(first_values,second_values)):
            if bool(value.get("audio_history_complete",False)):
                acoustic_history[str(role)]={key:value[key][0] for key in AUDIO_SEQUENCE_FIELDS}
                acoustic_history[str(role)]["now"]=torch.tensor(now,dtype=torch.float64,
                                                                  device=packet["target_audio"].device)
        next_state = StreamStateV3(session_id, roles, now, emotion, history, words, frozenset(consumed), acoustic_history)
        diagnostics = {"target_aff": first.aff, "partner_aff": second.aff, "observations": pair,
                       "observation_inputs": (first_values, second_values),
                       "target_modalities": first_values["modality_mask"], "partner_modalities": second_values["modality_mask"],
                       "target_fresh": first.fresh_observation, "partner_fresh": second.fresh_observation,
                       "context": context, "generator_context": routed_context, "film_enabled": film_enabled,
                       "timestamp": now, "state_affect": self.state_model.affect(emotion)}
        if return_generator_inputs:
            diagnostics['generator_inputs']=arguments
        return generated, next_state, diagnostics


class StreamSessionsV3:
    """Explicit session isolation; inference use is enclosed in no_grad by caller."""
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
