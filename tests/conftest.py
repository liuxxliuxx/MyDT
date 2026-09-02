from pathlib import Path

import pytest
import torch


@pytest.fixture
def feature_root(tmp_path: Path) -> Path:
    root = tmp_path / "features"
    dialogue = root / "dialogues" / "dialogue_001"
    dialogue.mkdir(parents=True)
    torch.save(torch.randn(4, 768), dialogue / "audio_features.pt")
    torch.save(torch.randn(4, 768), dialogue / "text_features.pt")
    torch.save(
        {
            "normalized": True,
            "utterance_ids": [f"utt_{i}" for i in range(4)],
            "au_sequences": [torch.randn(3, 35) for _ in range(4)],
            "confidence_sequences": [torch.ones(3) for _ in range(4)],
            "frame_valid_masks": [torch.ones(3, dtype=torch.bool) for _ in range(4)],
        },
        dialogue / "face_au_features.pt",
    )
    utterances = []
    for turn in range(4):
        utterances.append(
            {
                "utterance_id": f"utt_{turn}",
                "speaker_id": f"speaker_{turn % 2}",
                "speaker": "A" if turn % 2 == 0 else "B",
                "emotion_id": turn % 7,
                "sentiment_score": float(turn - 2),
                "intensity_abs": float(turn % 3),
                "start_time": float(turn),
                "end_time": float(turn) + 0.5,
            }
        )
    import json

    (dialogue / "labels.json").write_text(
        json.dumps({"utterances": utterances}), encoding="utf-8"
    )
    for split in ("train", "val", "test"):
        split_root = root / "splits"
        split_root.mkdir(parents=True, exist_ok=True)
        (split_root / f"{split}_dialogues.txt").write_text(
            "dialogue_001\n", encoding="utf-8"
        )
    return root


@pytest.fixture
def sequence_batch():
    batch_size, length = 3, 6
    valid = torch.ones(batch_size, length, dtype=torch.bool)
    valid[2, 4:] = False
    return {
        "audio": torch.randn(batch_size, length, 768),
        "face": torch.randn(batch_size, length, 35),
        "text": torch.randn(batch_size, length, 768),
        "modality_mask": torch.ones(batch_size, length, 3, dtype=torch.bool),
        "reliability": torch.ones(batch_size, length, 3),
        "dataset_id": torch.zeros(batch_size, length, dtype=torch.long),
        "valid_mask": valid,
        "speaker_ids": torch.tensor([[0, 1], [2, 3], [-1, -1]]),
        "active_role": torch.tensor([[0, 1, 0, 1, 0, 1]]).expand(batch_size, -1).clone(),
        "dt_to_next": torch.full((batch_size, length), 0.5),
        "emotion": torch.randint(0, 7, (batch_size, length)),
        "intensity": torch.rand(batch_size, length),
        "vad": torch.rand(batch_size, length, 3) * 2 - 1,
        "vad_mask": torch.ones(batch_size, length, 3, dtype=torch.bool),
        "turn_position": torch.linspace(0, 1, length)[None].expand(batch_size, -1),
        "dialogue_index": torch.arange(batch_size),
    }
