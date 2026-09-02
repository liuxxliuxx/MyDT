from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset

from .common import DialogueRecord, SpeakerVocabulary
from .feature_store import FeatureDialogueStore


UTTERANCE_FIELDS = (
    "audio",
    "face",
    "text",
    "modality_mask",
    "reliability",
    "emotion",
    "intensity",
    "vad",
    "vad_mask",
    "active_role",
    "turn_position",
    "start_time",
    "end_time",
    "dt_to_next",
)


class UnifiedUtteranceDataset(Dataset):
    def __init__(
        self,
        stores: Sequence[FeatureDialogueStore],
        split: str,
        speaker_vocab: SpeakerVocabulary,
    ) -> None:
        self.records: List[DialogueRecord] = []
        self.index: List[Tuple[int, int]] = []
        for store in stores:
            for dialogue_id in store.dialogue_ids(split):
                record_index = len(self.records)
                record = store.load_dialogue(dialogue_id, speaker_vocab)
                self.records.append(record)
                self.index.extend((record_index, i) for i in range(len(record)))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        record_index, turn = self.index[index]
        record = self.records[record_index]
        item = {name: getattr(record, name)[turn] for name in UTTERANCE_FIELDS}
        item.update(
            {
                "dataset_id": torch.tensor(record.dataset_id, dtype=torch.long),
                "speaker": record.speaker_ids[record.active_role[turn]],
                "dialogue_index": torch.tensor(record_index, dtype=torch.long),
                "turn_index": torch.tensor(turn, dtype=torch.long),
            }
        )
        return item

    def class_weights(self, num_classes: int = 7) -> Tensor:
        targets = torch.stack(
            [self.records[record].emotion[turn] for record, turn in self.index]
        )
        valid = targets >= 0
        counts = torch.bincount(targets[valid], minlength=num_classes).float()
        # 没有样本的类别权重置 0，避免小数据 fold 产生无穷大。
        weights = torch.zeros_like(counts)
        weights[counts > 0] = counts[counts > 0].rsqrt()
        positive = weights > 0
        weights[positive] /= weights[positive].mean()
        return weights


class DialogueWindowDataset(Dataset):
    def __init__(
        self,
        stores: Sequence[FeatureDialogueStore],
        split: str,
        speaker_vocab: SpeakerVocabulary,
        window_length: int = 33,
        stride: int = 4,
    ) -> None:
        self.window_length = window_length
        self.records: List[DialogueRecord] = []
        self.index: List[Tuple[int, int]] = []
        for store in stores:
            for dialogue_id in store.dialogue_ids(split):
                record_index = len(self.records)
                record = store.load_dialogue(dialogue_id, speaker_vocab)
                self.records.append(record)
                starts = list(range(0, max(len(record) - window_length + 1, 1), stride))
                final_start = max(len(record) - window_length, 0)
                if not starts or starts[-1] != final_start:
                    starts.append(final_start)
                self.index.extend((record_index, start) for start in starts)

    def __len__(self) -> int:
        return len(self.index)

    def class_weights(self, num_classes: int = 7) -> Tensor:
        targets = torch.cat([record.emotion for record in self.records])
        valid = targets >= 0
        counts = torch.bincount(targets[valid], minlength=num_classes).float()
        weights = torch.zeros_like(counts)
        weights[counts > 0] = counts[counts > 0].rsqrt()
        positive = weights > 0
        weights[positive] /= weights[positive].mean()
        return weights

    def _pad(self, value: Tensor, length: int, fill: float = 0.0) -> Tensor:
        shape = (self.window_length,) + tuple(value.shape[1:])
        output = torch.full(shape, fill, dtype=value.dtype)
        output[:length] = value[:length]
        return output

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        record_index, start = self.index[index]
        record = self.records[record_index]
        stop = min(start + self.window_length, len(record))
        length = stop - start
        item: Dict[str, Tensor] = {}
        for name in UTTERANCE_FIELDS:
            value = getattr(record, name)[start:stop]
            fill = -1 if name == "emotion" else 0
            item[name] = self._pad(value, length, fill=fill)

        valid_mask = torch.zeros(self.window_length, dtype=torch.bool)
        valid_mask[:length] = True
        item.update(
            {
                "valid_mask": valid_mask,
                "speaker_ids": record.speaker_ids.clone(),
                "dataset_id": torch.full(
                    (self.window_length,), record.dataset_id, dtype=torch.long
                ),
                "dialogue_index": torch.tensor(record_index, dtype=torch.long),
                "turn_indices": torch.arange(
                    start, start + self.window_length, dtype=torch.long
                ),
            }
        )
        return item
