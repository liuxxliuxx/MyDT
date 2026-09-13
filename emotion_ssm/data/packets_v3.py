"""Versioned second-clock dyadic packets, explicit freshness and endpoint labels."""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from torch.utils.data import Dataset

TOKEN_PROTOCOL = "emotion-token-packets-v3.1"
SUPERVISION_REVISION = "endpoint-intensity-mask-v3.1.1"
TOKEN_MODES = {"audio": 768, "prosody": 8, "au": 35, "flame": 56, "text": 768}
AUDIO_HISTORY_SECONDS = 16.
AUDIO_SEQUENCE_FIELDS = ("audio_tokens", "audio_mask", "audio_times", "audio_starts",
                         "audio_fresh_mask", "prosody_tokens", "prosody_mask", "prosody_times")


def validate_supervision_manifest(manifest):
    """Require repaired supervision for caches declaring an IEMOCAP source."""
    has_iemocap = ("iemocap" in manifest.get("feature_sources", {})
                   or any(item.get("dataset") == "iemocap"
                          for item in manifest.get("dialogues", {}).values()))
    if has_iemocap and manifest.get("supervision_revision") != SUPERVISION_REVISION:
        raise ValueError("IEMOCAP token cache requires supervision_revision="
                         f"{SUPERVISION_REVISION}; repair or rebuild its endpoint labels")


def empty_role_features(audio_dim=768, text_dim=768, now=0., dt=1., domain_id=0):
    output = {"now": torch.tensor(float(now),dtype=torch.float64), "dt": torch.tensor(float(dt),dtype=torch.float64),
              "domain_id": torch.tensor(domain_id, dtype=torch.long)}
    for mode, dimension in dict(TOKEN_MODES, audio=audio_dim, text=text_dim).items():
        output[mode + "_tokens"] = torch.zeros(1, dimension)
        output[mode + "_mask"] = torch.zeros(1, dtype=torch.bool)
        output[mode + "_times"] = torch.full((1,), float(now),dtype=torch.float64)
    output.update(audio_starts=torch.full((1,), float(now),dtype=torch.float64),
                  audio_fresh_mask=torch.zeros(1, dtype=torch.bool),
                  audio_history_complete=torch.tensor(False),
                  text_positions=torch.zeros(1, dtype=torch.long),
                  text_roles=torch.zeros(1, dtype=torch.long), text_fresh_mask=torch.zeros(1, dtype=torch.bool),
                  context_available=torch.tensor(False), fresh_observation=torch.zeros(3, dtype=torch.bool),
                  event_present=torch.tensor(False), action_present=torch.tensor(False), action_duration=torch.tensor(0.),
                  modality_mask=torch.zeros(3, dtype=torch.bool))
    return output


def collate_role_features(samples, device=None):
    if not samples:
        raise ValueError("Cannot collate an empty token batch")
    output = {}
    sequence_names = {m + s for m in TOKEN_MODES for s in ("_tokens", "_mask", "_times")} | {
        "text_roles", "text_positions", "text_fresh_mask", "audio_starts", "audio_fresh_mask"}
    for key in samples[0]:
        values = [torch.as_tensor(sample[key]) for sample in samples]
        if key in sequence_names:
            output[key] = torch.nn.utils.rnn.pad_sequence(values, batch_first=True)
        else:
            output[key] = torch.stack(values)
        if device is not None:
            output[key] = output[key].to(device)
    return output


def merge_audio_history(previous, current, history_seconds=AUDIO_HISTORY_SECONDS):
    """Retain past causal acoustic atoms, without repeating fresh evidence.

    ``current`` is one unbatched role feature dictionary. This exact operation
    is used by offline extraction and raw streaming. Only cached token tensors
    are retained, so the generator's raw-waveform context stays independent.
    """
    if history_seconds <= 0:
        raise ValueError("Observer audio history must be positive")
    now, since = float(current["now"]), float(current["now"])-float(current["dt"])
    result = dict(current)
    if bool(current.get("audio_history_complete", False)):
        # Source-bound cached packets already contain precisely this history.
        return result
    if previous is not None:
        if float(previous["now"]) > since+1e-8:
            raise ValueError("Audio history overlaps the new observation interval")
        for key in AUDIO_SEQUENCE_FIELDS:
            result[key] = torch.cat((previous[key].to(current[key].device), current[key]), dim=0)
    keep = (result["audio_starts"].double() >= now-float(history_seconds)-1e-9)
    keep &= result["audio_times"].double() <= now+1e-9
    if not keep.any():
        raise ValueError("Current audio tokens must contain at least one causal placeholder")
    for key in AUDIO_SEQUENCE_FIELDS:
        result[key] = result[key][keep]
    result["audio_fresh_mask"] = result["audio_mask"].bool() & (result["audio_times"].double() > since+1e-9)
    result["modality_mask"] = current["modality_mask"].clone()
    result["modality_mask"][0] = result["audio_mask"].any()
    result["fresh_observation"] = current["fresh_observation"].clone()
    result["fresh_observation"][0] = result["audio_fresh_mask"].any()
    result["audio_history_complete"] = torch.tensor(True,device=current["now"].device)
    return result


def validate_packet(packet):
    if packet.get("protocol") != TOKEN_PROTOCOL:
        raise ValueError("Pooled or legacy features cannot masquerade as V3 token packets")
    start, end = float(packet["start"]), float(packet["end"])
    if not math.isfinite(start + end) or start < 0 or not 0 < end-start <= 1.000001:
        raise ValueError("V3 packets require a positive duration <= one second")
    if abs(float(packet["dt"]) - (end-start)) > 1e-5:
        raise ValueError("Packet clock and dt disagree")
    if len(packet["roles"]) != 2:
        raise ValueError("Exactly two role observations are required")
    for role, features in enumerate(packet["roles"]):
        for mode in TOKEN_MODES:
            values, mask = features[mode + "_tokens"], features[mode + "_mask"].bool()
            if values.ndim != 2 or len(mask) != len(values):
                raise ValueError(f"Invalid {mode} token shape")
            if (features[mode + "_times"][mask].double() > end + 1e-9).any():
                raise ValueError("Future token leaked into a packet")
        if features["prosody_tokens"].shape != (len(features["audio_tokens"]), 8):
            raise ValueError("Prosody and acoustic atoms must be aligned")
        if not torch.equal(features["prosody_mask"], features["audio_mask"]) or not torch.equal(
                features["prosody_times"], features["audio_times"]):
            raise ValueError("Prosody must have the same mask and dependency endpoint as audio")
        if features["audio_starts"].shape != features["audio_times"].shape:
            raise ValueError("Every audio atom must declare its dependency start")
        if (features["audio_starts"] > features["audio_times"]).any():
            raise ValueError("Audio dependency interval runs backwards")
        if float(features["action_duration"]) < 0 or float(features["action_duration"]) > end-start+1e-5:
            raise ValueError("Action duration exceeds the actually observed interval")
        if not bool(features["action_present"]) and float(features["action_duration"]) != 0:
            raise ValueError("Absent action has nonzero duration")
        for label in packet.get("targets", [[], []])[role]:
            if not start < float(label["end"]) <= end + 1e-6:
                raise ValueError("Utterance supervision must be applied once at its true endpoint")
    target_role = packet.get("avatar_role")
    if target_role is not None and packet["roles"][int(target_role)]["flame_mask"].any():
        raise ValueError("Current Avatar target FLAME is forbidden in observation")


def endpoint_label(item, dataset):
    from emotion_ssm.data.feature_store import resolve_emotion_id
    if dataset == "emotiontalk":
        vad = [float(item.get("sentiment_score", 0.)) / 2., 0., 0.]
        vad_mask = ["sentiment_score" in item, False, False]
        intensity = float(item.get("intensity_abs", 0.)) / 2.
        intensity_mask = "intensity_abs" in item
    else:
        vad = [float(v) for v in item.get("vad", [0., 0., 0.])]
        vad_mask = list(item.get("vad_mask", ["vad" in item] * 3))
        intensity = float(item.get("intensity", 0.))
        intensity_mask = "intensity" in item
    # A supplied proxy value is not an independently annotated intensity label.
    intensity_source = item.get("intensity_source")
    intensity_mask = (intensity_mask and bool(item.get("intensity_mask", True))
                      and bool(item.get("intensity_valid", True))
                      and intensity_source != "arousal_proxy_for_candidate_matching_only")
    if dataset == "iemocap":
        # Legacy IEMOCAP exports also stored (arousal + 1) / 2 without metadata.
        # Independent additions must explicitly declare this source and a mask.
        intensity_mask = (intensity_mask and intensity_source == "independent_annotation"
                          and bool(item.get("intensity_mask", False)))
        if intensity_source is None and "intensity" in item:
            intensity_source = "arousal_proxy_for_candidate_matching_only"
    return {"start": float(item["start_time"]), "end": float(item["end_time"]),
            "emotion": resolve_emotion_id(item), "intensity": intensity,
            "raw_emotion": str(item.get("emotion",item.get("emotion_label",item.get("label","")))),
            "intensity_mask": intensity_mask, "intensity_source": intensity_source,
            "vad": vad, "vad_mask": vad_mask,
            "utterance_id": str(item.get("utterance_id", item.get("id", "")))}


def collate_endpoint_labels(labels, device=None):
    return {"emotion": torch.tensor([i["emotion"] for i in labels], dtype=torch.long, device=device),
            "intensity": torch.tensor([i["intensity"] for i in labels], device=device),
            "intensity_mask": torch.tensor([i["intensity_mask"] for i in labels], device=device),
            "vad": torch.tensor([i["vad"] for i in labels], device=device),
            "vad_mask": torch.tensor([i["vad_mask"] for i in labels], device=device),
            "endpoint_mask": torch.ones(len(labels), dtype=torch.bool, device=device)}


def prefix_role_features(features, endpoint):
    """Read an annotation at its true endpoint inside a one-second packet.

    Cached audio windows are independent. Excluding tokens whose end is later
    than this query also excludes their raw samples from the encoder's context.
    """
    start=float(features["now"])-float(features["dt"])
    if not start < endpoint <= float(features["now"])+1e-6:
        raise ValueError("Annotation endpoint falls outside the packet")
    result=dict(features)
    for mode in TOKEN_MODES:
        result[mode+"_mask"]=features[mode+"_mask"] & (features[mode+"_times"].double()<=endpoint+1e-9)
    available=torch.tensor([bool(result["audio_mask"].any()),bool(result["au_mask"].any() | result["flame_mask"].any()),
                            bool(result["text_mask"].any())])
    fresh=features["fresh_observation"].clone() & available
    current_audio=result["audio_mask"] & (features["audio_times"].double()>start+1e-9)
    result["audio_fresh_mask"]=current_audio
    fresh[0]=bool(current_audio.any())
    fresh[2]=bool((features["text_fresh_mask"] & result["text_mask"]).any())
    own=features["text_fresh_mask"] & result["text_mask"] & (features["text_roles"]==0)
    fresh[1]=bool(((result["au_mask"] & (features["au_times"]>start)) .any()) |
                  ((result["flame_mask"] & (features["flame_times"]>start)).any()))
    action=bool(fresh[:2].any()) and bool(features["action_present"])
    audio_duration=((features["audio_times"].double().clamp(max=endpoint)-
                     features["audio_starts"].double().clamp(min=start)).clamp_min(0)*current_audio).sum()
    visual_duration=(endpoint-start) if fresh[1] else 0.
    result.update(now=torch.tensor(float(endpoint),dtype=torch.float64),dt=torch.tensor(float(endpoint-start),dtype=torch.float64),modality_mask=available,
                  fresh_observation=fresh,context_available=available[2],event_present=own.any(),
                  action_present=torch.tensor(action),action_duration=torch.tensor(
                      min(endpoint-start,max(float(audio_duration),visual_duration)) if action else 0.))
    return result


class TokenPacketDataset(Dataset):
    """A full dialogue per item, with manifest-defined source disjoint splits."""
    def __init__(self, root, split="train"):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest["protocol"] != TOKEN_PROTOCOL:
            raise ValueError("Unsupported token cache protocol")
        if "digest" in self.manifest:
            from emotion_ssm.data.protocol import fingerprint
            if self.manifest["digest"] != fingerprint({k:v for k,v in self.manifest.items() if k != "digest"}):
                raise ValueError("Token manifest digest mismatch")
        validate_supervision_manifest(self.manifest)
        self.ids = list(self.manifest["splits"][split])
        self.feature_sources = self.manifest["feature_sources"]
        self.lengths = [int(self.manifest["dialogues"][name]["packets"]) for name in self.ids]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        name = self.ids[index]
        path = self.root / self.manifest["dialogues"][name]["path"]
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload["protocol"] != TOKEN_PROTOCOL or payload["cache_id"] != self.manifest["dialogues"][name]["cache_id"]:
            raise ValueError("Token cache provenance mismatch")
        for packet in payload["packets"]:
            validate_packet(packet)
        return payload


class PacketFrameDataset(Dataset):
    """Training A0 on packets; keeps only one recently read dialogue in memory."""
    def __init__(self, root, split="train"):
        self.dialogues = TokenPacketDataset(root, split)
        self.index = [(i, j, r) for i, length in enumerate(self.dialogues.lengths)
                      for j in range(length) for r in (0, 1)]
        self._cached_index, self._cached = None, None

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        dialogue, packet, role = self.index[index]
        if self._cached_index != dialogue:
            self._cached = self.dialogues[dialogue]
            self._cached_index = dialogue
        item = self._cached["packets"][packet]
        return {"features": item["roles"][role], "targets": item["targets"][role],
                "dialogue_id": item["dialogue_id"], "packet_index": packet, "role": role}


class TokenDualTalk:
    """Generator adapter using the same V3 packets as observation/dynamics.

    Only raw audio and the two FLAME targets are read here. Tokens remain
    source-bound and current Avatar visual input is independently rejected.
    """
    # packets() only loads/validates fixed CPU data; stochastic audio augmentation
    # remains in the generator's main training thread.
    prefetch_rng_neutral = True

    def __init__(self, raw_root, token_root, split="train", fps=25, threshold=1e-4):
        if fps != 25:
            raise ValueError("V3 DualTalk clock requires 25 FPS")
        self.root, self.split = Path(raw_root), split
        self.dataset = TokenPacketDataset(token_root, split)
        self.indices = [i for i,name in enumerate(self.dataset.ids) if name.startswith("dualtalk:")]
        self.names = [self.dataset.ids[i].partition(":")[2] for i in self.indices]
        self.lengths = [self.dataset.lengths[i] for i in self.indices]
        self.feature_sources = self.dataset.feature_sources
        self.manifest_digest = self.dataset.manifest.get("digest", "")
        self.feature_digest = self.manifest_digest
        self.threshold, self.fps = threshold, fps
        self.folder = self.root / ("train" if split in ("train", "val") else split)

    def __len__(self):
        return len(self.indices)

    def packets(self, index):
        import librosa
        from emotion_ssm.data.dualtalk import partner_stem
        payload = self.dataset[self.indices[index]]
        name = self.names[index]
        stems = (name, partner_stem(name))
        roles = ("speaker1" if name.endswith("speaker1") else "speaker2",
                 "speaker2" if name.endswith("speaker1") else "speaker1")
        for filename,size,modified in payload.get("audit",{}).get("raw_signature",[]):
            path = Path(filename)
            if not path.exists() or (path.stat().st_size,path.stat().st_mtime_ns) != (size,modified):
                raise ValueError(f"Raw input changed after V3 preprocessing: {path}")
        waves = [torch.from_numpy(librosa.load(str(self.folder/(stem+".wav")),sr=16000)[0]) for stem in stems]
        for stored in payload["packets"]:
            a,b = round(stored["start"]*16000),round(stored["end"]*16000)
            raw = [wave[a:b][None] for wave in waves]
            feature_pair = [collate_role_features([f]) for f in stored["roles"]]
            for features in feature_pair:
                features["available_at"] = stored["end"]
            truth = stored["target_flame"]
            mask = stored["target_frame_mask"]
            packet = {"session_id": name,"roles":roles,"start_time":0.,"time":stored["end"],
                      "target_audio":raw[0],"partner_audio":raw[1],
                      "target_speech_active":bool(raw[0].square().mean().sqrt()>=self.threshold),
                      "partner_speech_active":bool(raw[1].square().mean().sqrt()>=self.threshold),
                      "partner_blendshape":truth[1:2],"partner_visual_mask":mask[1:2],
                      "target_features":feature_pair[0],"partner_features":feature_pair[1],
                      "text_protocol":stored["text_protocol"],"source_domain":2,
                      "feature_source":payload["feature_source"]}
            yield packet,truth[0:1],mask[0:1]
