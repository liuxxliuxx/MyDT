from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import torch
from torch import Tensor


EMOTION_NAMES = (
    "angry",
    "disgusted",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
)


def load_pt(path: Path):
    kwargs = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        # 数据由本项目预处理生成，包含 Tensor 列表和少量字符串元数据。
        kwargs["weights_only"] = False
    return torch.load(str(path), **kwargs)


def read_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def pool_face_sequence(au: Tensor, confidence: Tensor, valid: Tensor) -> Tuple[Tensor, float]:
    if au.ndim != 2 or au.shape[-1] != 35:
        raise ValueError(f"Expected face AU [frames, 35], got {tuple(au.shape)}")
    if len(au) != len(confidence) or len(au) != len(valid):
        raise ValueError("AU, confidence and frame-valid lengths do not match")
    weight = confidence.float().clamp_min(0.0) * valid.bool().float()
    denominator = weight.sum()
    if denominator.item() <= 0:
        return torch.zeros(35, dtype=torch.float32), 0.0
    pooled = (au.float() * weight[:, None]).sum(dim=0) / denominator
    reliability = float((confidence.float() * valid.float()).sum() / valid.sum().clamp_min(1))
    return pooled, max(0.0, min(reliability, 1.0))


class SpeakerVocabulary:
    """Training-only speaker mapping. Unseen speakers deliberately map to -1."""

    def __init__(self, speakers: Iterable[str]) -> None:
        unique = sorted(set(speakers))
        self.speaker_to_id = {speaker: i for i, speaker in enumerate(unique)}

    def __len__(self) -> int:
        return len(self.speaker_to_id)

    def encode(self, speaker: str) -> int:
        return self.speaker_to_id.get(speaker, -1)

    def state_dict(self) -> Dict[str, int]:
        return dict(self.speaker_to_id)

    @classmethod
    def from_state_dict(cls, mapping: Mapping[str, int]) -> "SpeakerVocabulary":
        vocab = cls([])
        vocab.speaker_to_id = dict(mapping)
        return vocab


@dataclass
class DialogueRecord:
    dialogue_id: str
    dataset_name: str
    dataset_id: int
    audio: Tensor
    face: Tensor
    text: Tensor
    modality_mask: Tensor
    reliability: Tensor
    emotion: Tensor
    intensity: Tensor
    vad: Tensor
    vad_mask: Tensor
    active_role: Tensor
    speaker_ids: Tensor
    turn_position: Tensor
    start_time: Tensor
    end_time: Tensor
    dt_to_next: Tensor
    utterance_ids: List[str]

    def __len__(self) -> int:
        return len(self.emotion)


def validate_record(record: DialogueRecord) -> None:
    count = len(record)
    tensors = {
        "audio": record.audio,
        "face": record.face,
        "text": record.text,
        "modality_mask": record.modality_mask,
        "reliability": record.reliability,
        "emotion": record.emotion,
        "intensity": record.intensity,
        "vad": record.vad,
        "vad_mask": record.vad_mask,
        "active_role": record.active_role,
        "turn_position": record.turn_position,
        "start_time": record.start_time,
        "end_time": record.end_time,
        "dt_to_next": record.dt_to_next,
    }
    bad = [name for name, value in tensors.items() if len(value) != count]
    if bad:
        raise ValueError(f"{record.dialogue_id}: mismatched fields: {bad}")
    if record.audio.shape[-1] != 768 or record.text.shape[-1] != 768:
        raise ValueError(f"{record.dialogue_id}: audio/text feature dimension must be 768")
    if record.face.shape[-1] != 35:
        raise ValueError(f"{record.dialogue_id}: face feature dimension must be 35")
    for name in ("audio", "face", "text", "intensity", "vad"):
        if not torch.isfinite(tensors[name]).all():
            raise ValueError(f"{record.dialogue_id}: {name} contains NaN or Inf")
