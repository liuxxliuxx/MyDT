"""Lazy, one-second packets. Validation selections never move source files."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

from emotion_ssm.data.dualtalk import partner_stem, speech_is_active
from emotion_ssm.data.protocol import PROTOCOL_VERSION, require_compatible, fingerprint


def load_flame(path):
    with np.load(path) as value:
        return torch.from_numpy(np.concatenate([value["exp"], value["pose"][:, 3:], value["pose"][:, :3]], -1)).float()


class TimedDualTalk:
    def __init__(self, root, feature_root, manifest, split="train", fps=25, threshold=1e-4, expected_source=None):
        self.root, self.feature_root = Path(root), Path(feature_root)
        self.fps, self.threshold = fps, threshold
        manifest = json.loads(Path(manifest).read_text(encoding="utf-8")) if not isinstance(manifest, dict) else manifest
        if manifest.get("version") != PROTOCOL_VERSION:
            raise ValueError("Regenerate the v2 split manifest")
        if manifest["digest"] != fingerprint({k: v for k, v in manifest.items() if k != "digest"}):
            raise ValueError("Split manifest digest mismatch")
        self.manifest_digest = manifest["digest"]
        self.split = split
        self.folder = self.root / ("train" if split in ("train", "val") else split)
        self.names = manifest[split] if split in ("train", "val") else sorted(p.stem for p in self.folder.glob("*.npz"))
        self.expected_source = expected_source
        self.text_protocol = None
        feature_manifest = json.loads((self.feature_root / "dataset_manifest.json").read_text(encoding="utf-8"))
        self.feature_ids = feature_manifest["cache_ids"]
        self.feature_digest = feature_manifest["digest"]
        if self.feature_digest != fingerprint(self.feature_ids):
            raise ValueError("Feature dataset manifest digest mismatch")

    def __len__(self):
        return len(self.names)

    def packets(self, index):
        import librosa
        name = self.names[index]
        partner = partner_stem(name)
        role = "speaker1" if name.endswith("speaker1") else "speaker2"
        other_role = "speaker2" if role == "speaker1" else "speaker1"
        target = load_flame(self.folder / (name + ".npz"))
        visual = load_flame(self.folder / (partner + ".npz"))
        waves = [torch.from_numpy(librosa.load(str(self.folder / (stem + ".wav")), sr=16000)[0])
                 for stem in (name, partner)]
        frames = min(len(target), len(visual), int(min(len(w) for w in waves) / 16000 * self.fps))
        cache_path = self.feature_root / self.folder.name / (name + ".pt")
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        if cache["cache_id"] != self.feature_ids.get(f"{self.folder.name}/{name}"):
            raise ValueError("Timed cache is not registered in the feature dataset manifest")
        if self.text_protocol is not None and cache["text_protocol"] != self.text_protocol:
            raise ValueError("A split cannot mix aligned transcript and external ASR protocols")
        self.text_protocol = cache["text_protocol"]
        if cache.get("version") != PROTOCOL_VERSION or cache.get("fps") != self.fps:
            raise ValueError(f"Stale timed features: {cache_path}")
        if cache.get("speech_rms_threshold") != self.threshold:
            raise ValueError("Speech presence policy changed; rebuild timed features")
        if self.expected_source is not None:
            require_compatible(self.expected_source, cache["feature_source"])
        for filename, size, timestamp in cache.get("raw_signature", []):
            path = self.folder / filename
            if not path.exists() or (path.stat().st_size, path.stat().st_mtime_ns) != (size, timestamp):
                raise ValueError(f"Raw input changed since preprocessing: {path}")
        if len(cache["chunks"]) != math.ceil(frames / self.fps):
            raise ValueError(f"Timed cache and raw valid lengths disagree: {name}")
        for chunk, start in enumerate(range(0, frames, self.fps)):
            stop = min(start + self.fps, frames)
            a, b = round(start / self.fps * 16000), round(stop / self.fps * 16000)
            raw = [w[a:b] for w in waves]
            feature_pair = cache["chunks"][chunk]
            for values in feature_pair.values():
                if abs(float(values["available_at"]) - stop/self.fps) > 1e-6:
                    raise ValueError(f"Timed feature timestamp mismatch: {name}, block {chunk}")
            packet = {"session_id": name, "roles": (role, other_role), "start_time": 0.,
                      "time": stop / self.fps,
                      "target_audio": raw[0][None], "partner_audio": raw[1][None],
                      "target_speech_active": speech_is_active(raw[0].numpy(), self.threshold),
                      "partner_speech_active": speech_is_active(raw[1].numpy(), self.threshold),
                      "partner_blendshape": visual[start:stop][None],
                      "partner_visual_mask": torch.isfinite(visual[start:stop]).all(-1)[None],
                      "target_features": feature_pair["target"], "partner_features": feature_pair["partner"],
                      "text_protocol": cache["text_protocol"]}
            packet["partner_blendshape"] = torch.nan_to_num(packet["partner_blendshape"])
            truth = target[start:stop][None]
            mask = torch.isfinite(truth).all(-1)
            yield packet, torch.nan_to_num(truth), mask


def to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {k: to_device(v, device) for k, v in value.items()}
    return value
