from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import Tensor

from .common import (
    EMOTION_NAMES,
    FACE_MAX_FRAMES,
    DialogueRecord,
    SpeakerVocabulary,
    load_pt,
    pack_face_sequence,
    pool_face_sequence,
    read_json,
    validate_record,
)


EMOTION_ALIASES = {
    "ang": 0,
    "anger": 0,
    "angry": 0,
    "愤怒": 0,
    "dis": 1,
    "disgust": 1,
    "disgusted": 1,
    "厌恶": 1,
    "fea": 2,
    "fear": 2,
    "fearful": 2,
    "恐惧": 2,
    "hap": 3,
    "happy": 3,
    "exc": 3,
    "excited": 3,
    "高兴": 3,
    "开心": 3,
    "neu": 4,
    "neutral": 4,
    "中性": 4,
    "sad": 5,
    "sadness": 5,
    "悲伤": 5,
    "sur": 6,
    "surprise": 6,
    "surprised": 6,
    "惊讶": 6,
}


def resolve_emotion_id(item: Dict) -> int:
    for key in ("emotion_id", "emotion_label", "emotion"):
        if key not in item:
            continue
        value = item[key]
        if isinstance(value, (int, float)):
            value = int(value)
            return value if 0 <= value < len(EMOTION_NAMES) else -1
        text = str(value).strip().lower()
        if text.lstrip("-").isdigit():
            value = int(text)
            return value if 0 <= value < len(EMOTION_NAMES) else -1
        return EMOTION_ALIASES.get(text, -1)
    return -1


def resolve_feature_tensor(payload, name: str) -> Tensor:
    if torch.is_tensor(payload):
        return payload
    if isinstance(payload, dict):
        for key in ("features", "embeddings", "data", f"{name}_features", name):
            if key in payload and torch.is_tensor(payload[key]):
                return payload[key]
    raise ValueError(f"Cannot find a feature tensor in {name}_features.pt")


def resolve_utterances(labels) -> List[Dict]:
    if isinstance(labels, list):
        return labels
    if isinstance(labels, dict) and "utterances" in labels:
        return list(labels["utterances"])
    if isinstance(labels, dict) and all(
        isinstance(value, dict) for value in labels.values()
    ):
        utterances = []
        for utterance_id, value in labels.items():
            item = dict(value)
            item.setdefault("utterance_id", utterance_id)
            utterances.append(item)
        return utterances
    raise ValueError("labels.json must contain an utterance list or mapping")


def resolve_speaker_id(item: Dict, dialogue_id: str) -> str:
    for key in ("speaker_id", "actor_id", "person_id"):
        if key in item:
            return str(item[key])
    if "speaker" in item:
        return f"{dialogue_id}:{item['speaker']}"
    raise KeyError(f"{dialogue_id}: utterance has no speaker identity")


def resolve_role(value: object) -> Optional[int]:
    text = str(value).strip().lower()
    if text in {"a", "f", "0", "speaker1", "user"}:
        return 0
    if text in {"b", "m", "1", "speaker2", "avatar"}:
        return 1
    return None


def declared_role_speakers(labels: object) -> Dict[int, str]:
    """Read both participants even when only one speaks in a short segment."""
    if not isinstance(labels, dict):
        return {}
    speaker_map = labels.get("speaker_map", {})
    if not isinstance(speaker_map, dict):
        return {}
    output: Dict[int, str] = {}
    for speaker_id, role_name in speaker_map.items():
        role = resolve_role(role_name)
        if role is not None:
            output[role] = str(speaker_id)
    return output


def declared_speaker_ids(labels: object) -> List[str]:
    if not isinstance(labels, dict):
        return []
    output = list(declared_role_speakers(labels).values())
    values = labels.get("speaker_ids", [])
    if isinstance(values, (list, tuple)):
        output.extend(str(value) for value in values)
    return output


def unpack_face_data(face_data, utterance_ids: Sequence[str]):
    if isinstance(face_data, dict) and "au_sequences" in face_data:
        stored_ids = list(face_data.get("utterance_ids", utterance_ids))
        if stored_ids != list(utterance_ids):
            raise ValueError("Face and label utterance order differs")
        au_sequences = list(face_data["au_sequences"])
        confidence_sequences = list(face_data.get("confidence_sequences", []))
        valid_sequences = list(face_data.get("frame_valid_masks", []))
        if not confidence_sequences:
            confidence_sequences = [torch.ones(len(value)) for value in au_sequences]
        if not valid_sequences:
            valid_sequences = [
                torch.isfinite(torch.as_tensor(value)).all(dim=-1)
                for value in au_sequences
            ]
        return (
            au_sequences,
            confidence_sequences,
            valid_sequences,
            bool(face_data.get("normalized", True)),
        )
    if isinstance(face_data, dict):
        values = []
        for index, utterance_id in enumerate(utterance_ids):
            key = utterance_id
            if key not in face_data and index in face_data:
                key = index
            if key not in face_data and str(index) in face_data:
                key = str(index)
            if key not in face_data:
                raise KeyError(f"Missing face AU for utterance {utterance_id}")
            values.append(face_data[key])
    elif torch.is_tensor(face_data) or isinstance(face_data, (list, tuple)):
        values = list(face_data)
    else:
        raise ValueError("Unsupported face_au_features.pt layout")
    au_sequences = []
    confidence_sequences = []
    valid_sequences = []
    for value in values:
        if isinstance(value, dict):
            au = value.get("au", value.get("features"))
            confidence = value.get("confidence")
            valid = value.get("valid_mask")
        else:
            au = value
            confidence = None
            valid = None
        au = torch.as_tensor(au).float()
        if au.ndim == 1:
            au = au[None]
        if confidence is None:
            confidence = torch.ones(len(au))
        if valid is None:
            valid = torch.isfinite(au).all(dim=-1)
        au_sequences.append(torch.nan_to_num(au))
        confidence_sequences.append(torch.as_tensor(confidence).float())
        valid_sequences.append(torch.as_tensor(valid).bool())
    return au_sequences, confidence_sequences, valid_sequences, True


class FeatureDialogueStore:
    """Read the shared per-dialogue feature layout without modifying it."""

    def __init__(
        self,
        root: Path,
        dataset_name: str,
        dataset_id: int,
        fold: Optional[int] = None,
    ) -> None:
        self.root = Path(root)
        self.dataset_name = dataset_name
        self.dataset_id = dataset_id
        self.fold = fold
        self.dialogues_root = self.root / "dialogues"
        if not self.dialogues_root.is_dir() and any(
            path.is_file() for path in self.root.glob("*/labels.json")
        ):
            self.dialogues_root = self.root
        self.splits_root = self.root / "splits"
        if fold is not None:
            self.splits_root = self.splits_root / f"fold_{fold}"
        if not self.dialogues_root.is_dir() or not self.splits_root.is_dir():
            raise FileNotFoundError(
                f"Invalid {dataset_name} feature layout at {self.root}"
            )

    def dialogue_ids(self, split: str) -> List[str]:
        path = self.splits_root / f"{split}_dialogues.txt"
        if not path.is_file():
            alternate = self.splits_root / f"{split}.txt"
            if alternate.is_file():
                path = alternate
            else:
                raise FileNotFoundError(path)
        values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
        values = [value for value in values if value]
        if not values:
            raise ValueError(f"Empty split file: {path}")
        return values

    def collect_speakers(self, split: str) -> List[str]:
        speakers = []
        for dialogue_id in self.dialogue_ids(split):
            labels = read_json(self.dialogues_root / dialogue_id / "labels.json")
            speakers.extend(
                self._speaker_key(speaker_id)
                for speaker_id in declared_speaker_ids(labels)
            )
            utterances = resolve_utterances(labels)
            for item in utterances:
                speakers.append(
                    self._speaker_key(resolve_speaker_id(item, dialogue_id))
                )
        return speakers

    def _speaker_key(self, speaker_id: str) -> str:
        return f"{self.dataset_name}:{speaker_id}"

    def load_dialogue(
        self, dialogue_id: str, speaker_vocab: SpeakerVocabulary
    ) -> DialogueRecord:
        root = self.dialogues_root / dialogue_id
        audio = resolve_feature_tensor(
            load_pt(root / "audio_features.pt"), "audio"
        ).float()
        text = resolve_feature_tensor(
            load_pt(root / "text_features.pt"), "text"
        ).float()
        face_data = load_pt(root / "face_au_features.pt")
        labels = read_json(root / "labels.json")
        utterances = resolve_utterances(labels)
        utterance_ids = [
            str(item.get("utterance_id", item.get("id", index)))
            for index, item in enumerate(utterances)
        ]

        if len(audio) != len(text) or len(audio) != len(utterances):
            raise ValueError(f"{dialogue_id}: A/T/label counts differ")
        au_sequences, confidence_sequences, frame_valid_masks, normalized = (
            unpack_face_data(face_data, utterance_ids)
        )

        face_vectors = []
        face_masks = []
        face_confidences = []
        face_reliability = []
        face_mean = None
        face_std = None
        if not normalized:
            stats_path = self.root / "metadata" / f"fold_{self.fold}_face_stats.pt"
            if self.fold is None or not stats_path.exists():
                raise FileNotFoundError(
                    f"Raw face AU requires fold-specific stats: {stats_path}"
                )
            stats = load_pt(stats_path)
            face_mean = stats["mean"].float()
            face_std = stats["std"].float().clamp_min(1e-6)
        for au, confidence, valid in zip(
            au_sequences,
            confidence_sequences,
            frame_valid_masks,
        ):
            if face_mean is not None:
                au = (au.float() - face_mean) / face_std
            # Keep the frame axis. TemporalAUEncoder uses this mask to ignore
            # padded/failed OpenFace frames while retaining order information.
            au = torch.nan_to_num(au.float())
            confidence = confidence.float().flatten()
            valid = valid.bool().flatten()
            if len(confidence) != len(au) or len(valid) != len(au):
                raise ValueError(f"{dialogue_id}: AU frame metadata length mismatch")
            vector, reliability = pool_face_sequence(au, confidence, valid)
            face_vectors.append(au)
            face_masks.append(valid)
            face_confidences.append(confidence.clamp(0.0, 1.0))
            face_reliability.append(reliability)
        if len(face_vectors) != len(utterances):
            raise ValueError(f"{dialogue_id}: face sequence count differs from labels")
        packed = [
            pack_face_sequence(au, confidence, valid, FACE_MAX_FRAMES)
            for au, valid, confidence in zip(
                face_vectors, face_masks, face_confidences
            )
        ]
        face = torch.stack([value[0] for value in packed], dim=0)
        face_confidence = torch.stack([value[1] for value in packed], dim=0)
        face_frame_mask = torch.stack([value[2] for value in packed], dim=0)

        modality_mask = torch.ones(len(utterances), 3, dtype=torch.bool)
        reliability = torch.ones(len(utterances), 3, dtype=torch.float32)
        reliability[:, 1] = torch.tensor(face_reliability)
        modality_mask[:, 1] = reliability[:, 1] > 0

        emotion = torch.tensor(
            [resolve_emotion_id(item) for item in utterances], dtype=torch.long
        )
        intensity_values = []
        vad_values = []
        vad_masks = []
        for item in utterances:
            if "vad" in item:
                vad = [float(value) for value in item["vad"]]
                vad_mask = [bool(value) for value in item.get("vad_mask", [1, 1, 1])]
                intensity = float(item.get("intensity", (vad[1] + 1.0) / 2.0))
            else:
                # EmotionTalk 的 intensity_abs/sentiment_score 范围均为 [-2, 2] 的子集。
                valence = float(item.get("sentiment_score", 0.0)) / 2.0
                arousal = float(item.get("intensity_abs", 0.0)) / 2.0
                vad = [valence, arousal, 0.0]
                vad_mask = [True, True, False]
                intensity = float(item.get("intensity_abs", 0.0)) / 2.0
            intensity_values.append(intensity)
            vad_values.append(vad)
            vad_masks.append(vad_mask)

        role_values = []
        role_to_speaker = declared_role_speakers(labels)
        inferred_roles: Dict[str, int] = {}
        for item in utterances:
            speaker_id = resolve_speaker_id(item, dialogue_id)
            if "active_role" in item:
                role = int(item["active_role"])
            else:
                role = resolve_role(item.get("speaker", ""))
                if role is None:
                    role = next(
                        (
                            known_role
                            for known_role, known_speaker in role_to_speaker.items()
                            if known_speaker == speaker_id
                        ),
                        None,
                    )
                if role is None:
                    if speaker_id not in inferred_roles:
                        used_roles = set(role_to_speaker) | set(inferred_roles.values())
                        inferred_roles[speaker_id] = next(
                            (candidate for candidate in (0, 1) if candidate not in used_roles),
                            len(used_roles),
                        )
                    role = inferred_roles[speaker_id]
            if role not in {0, 1}:
                raise ValueError(f"{dialogue_id}: active_role must be 0 or 1")
            declared_speaker = role_to_speaker.get(role)
            if declared_speaker is not None and declared_speaker != speaker_id:
                raise ValueError(
                    f"{dialogue_id}: role {role} maps to both "
                    f"{declared_speaker} and {speaker_id}"
                )
            role_values.append(role)
            role_to_speaker.setdefault(role, speaker_id)
        # Some EmotionTalk clips contain only one active speaker. Keep the
        # dyadic schema by representing the silent partner as an unseen role.
        for missing_role in (0, 1):
            role_to_speaker.setdefault(
                missing_role, f"{dialogue_id}:__missing_role_{missing_role}"
            )
        speaker_ids = torch.tensor(
            [
                speaker_vocab.encode(self._speaker_key(role_to_speaker[0])),
                speaker_vocab.encode(self._speaker_key(role_to_speaker[1])),
            ],
            dtype=torch.long,
        )

        start_time = torch.tensor(
            [float(item.get("start_time", i)) for i, item in enumerate(utterances)],
            dtype=torch.float32,
        )
        end_time = torch.tensor(
            [float(item.get("end_time", start_time[i])) for i, item in enumerate(utterances)],
            dtype=torch.float32,
        )
        dt_to_next = torch.zeros(len(utterances), dtype=torch.float32)
        if len(utterances) > 1:
            dt_to_next[:-1] = (start_time[1:] - start_time[:-1]).clamp(0.0, 60.0)
        turn_position = torch.tensor(
            [
                float(item.get("turn_position", i / max(len(utterances) - 1, 1)))
                for i, item in enumerate(utterances)
            ],
            dtype=torch.float32,
        )

        record = DialogueRecord(
            dialogue_id=dialogue_id,
            dataset_name=self.dataset_name,
            dataset_id=self.dataset_id,
            audio=audio,
            face=face,
            face_frame_mask=face_frame_mask,
            face_confidence=face_confidence,
            text=text,
            modality_mask=modality_mask,
            reliability=reliability,
            emotion=emotion,
            intensity=torch.tensor(intensity_values, dtype=torch.float32),
            vad=torch.tensor(vad_values, dtype=torch.float32),
            vad_mask=torch.tensor(vad_masks, dtype=torch.bool),
            active_role=torch.tensor(role_values, dtype=torch.long),
            speaker_ids=speaker_ids,
            turn_position=turn_position,
            start_time=start_time,
            end_time=end_time,
            dt_to_next=dt_to_next,
            utterance_ids=utterance_ids,
        )
        validate_record(record)
        return record


def build_speaker_vocabulary(stores: Sequence[FeatureDialogueStore]) -> SpeakerVocabulary:
    speakers: List[str] = []
    for store in stores:
        speakers.extend(store.collect_speakers("train"))
    return SpeakerVocabulary(speakers)
