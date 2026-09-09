"""Full chronological dialogues with padding confined to each minibatch."""
import torch
from torch.utils.data import Dataset

from .datasets import UTTERANCE_FIELDS, _pad_record_faces


class FullDialogueDataset(Dataset):
    def __init__(self, stores, split, speaker_vocab, *unused):
        self.records = [store.load_dialogue(name, speaker_vocab) for store in stores
                        for name in store.dialogue_ids(split)]
        _pad_record_faces(self.records)
        self.lengths = [len(record) for record in self.records]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        order = torch.argsort(record.end_time, stable=True)
        result = {name: getattr(record, name)[order] for name in UTTERANCE_FIELDS}
        ends = result["end_time"]
        result["dt_to_next"] = torch.cat([(ends[1:] - ends[:-1]).clamp_min(0), ends.new_zeros(1)])
        result.update(valid_mask=torch.ones(len(record), dtype=torch.bool),
                      speaker_ids=record.speaker_ids.clone(),
                      dataset_id=torch.full((len(record),), record.dataset_id, dtype=torch.long),
                      dialogue_index=torch.tensor(index), turn_indices=order)
        return result

    def class_weights(self, num_classes=7):
        labels = torch.cat([r.emotion for r in self.records])
        counts = torch.bincount(labels[labels >= 0], minlength=num_classes).float()
        weights = torch.where(counts > 0, counts.clamp_min(1).rsqrt(), 0.)
        weights /= weights[weights > 0].mean().clamp_min(1e-6)
        return weights


def collate_full_dialogues(items):
    maximum = max(len(item["valid_mask"]) for item in items)
    result = {}
    for name in items[0]:
        if name in ("speaker_ids", "dialogue_index"):
            result[name] = torch.stack([item[name] for item in items])
            continue
        first = items[0][name]
        padded = first.new_full((len(items), maximum) + first.shape[1:], -1 if name == "emotion" else 0)
        for row, item in enumerate(items):
            padded[row, :len(item[name])] = item[name]
        result[name] = padded
    return result
