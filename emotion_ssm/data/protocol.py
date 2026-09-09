"""Shared provenance and availability rules. No implicit cross-clip history."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

PROTOCOL_VERSION = 2
PREPROCESS_VERSION = "emotion-v2-timed-role-masked"
DOMAINS = {"emotiontalk": 0, "iemocap": 1, "dualtalk": 2}


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def model_revision(model_id, config):
    """Bind local checkpoints to file content, and hosted models to their commit."""
    from pathlib import Path
    root = Path(model_id)
    if not root.is_dir():
        return getattr(config, "_commit_hash", None) or ""
    digest = hashlib.sha256()
    suffixes = {".json", ".txt", ".model", ".bin", ".safetensors"}
    files = sorted(p for p in root.iterdir() if p.is_file() and p.suffix in suffixes)
    if not files:
        raise ValueError(f"Local model directory has no identifiable weights/configuration: {model_id}")
    for path in files:
        digest.update(path.name.encode())
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024*1024), b""):
                digest.update(chunk)
    return "local-sha256:" + digest.hexdigest()


@dataclass(frozen=True)
class FeatureSource:
    audio_model: str
    text_model: str
    normalization: str = "valid-waveform-zscore-v1"
    preprocessing: str = PREPROCESS_VERSION
    audio_dim: int = 768
    text_dim: int = 768
    audio_revision: str = ""
    text_revision: str = ""

    def to_dict(self):
        return asdict(self)


def select_adapter_source(sources: Sequence[str], requested: int = -1) -> int:
    trained = {DOMAINS[name] for name in sources if name in ("emotiontalk", "iemocap")}
    if not trained:
        raise ValueError("No trained emotion domain is declared")
    selected = requested if requested >= 0 else (1 if 1 in trained else 0)
    if selected not in trained:
        raise ValueError(f"Adapter {selected} was not trained on {list(sources)}")
    return selected


def require_compatible(expected: Mapping, actual: Mapping) -> None:
    for key in ("audio_model", "text_model", "normalization", "preprocessing",
                "audio_dim", "text_dim", "audio_revision", "text_revision"):
        if key not in expected or expected.get(key) != actual.get(key):
            raise ValueError(f"Feature provenance mismatch for {key}: "
                             f"{expected.get(key)!r} != {actual.get(key)!r}")


def source_video_id(name: str) -> str:
    return name.partition("_sub_video_")[0]


def make_split_manifest(names: Sequence[str], seed: int = 6666, fraction: float = .1):
    groups = sorted({source_video_id(name) for name in names},
                    key=lambda x: fingerprint([seed, x]))
    if len(groups) < 2:
        raise ValueError("At least two independent video sources are required")
    count = min(len(groups) - 1, max(1, round(len(groups) * fraction)))
    validation = set(groups[:count])
    result = {"version": PROTOCOL_VERSION, "seed": seed, "val_fraction": fraction,
              "train": sorted(n for n in names if source_video_id(n) not in validation),
              "val": sorted(n for n in names if source_video_id(n) in validation)}
    result["digest"] = fingerprint(result)
    return result


def visible_words(words: Sequence[Mapping], now: float, since: float = -float("inf")):
    """Arrival time, not linguistic end time, controls online availability."""
    selected = []
    for item in words:
        available = float(item["available_at"])
        if available < float(item["end"]):
            raise ValueError("A word cannot be available before its acoustic end")
        if since < available <= now:
            selected.append(item)
    return sorted(selected, key=lambda item: (float(item["available_at"]),
                                              float(item["end"]), str(item["role"])))


def role_text(words: Sequence[Mapping]) -> str:
    return " ".join(f"[{word['role']}] {word['text']}" for word in words)
