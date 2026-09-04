import json
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.io import wavfile

from emotion_ssm.data import (
    DialogueWindowDataset,
    DualTalkChunkDataset,
    FeatureDialogueStore,
    SpeakerVocabulary,
    UnifiedUtteranceDataset,
)
from emotion_ssm.preprocess.iemocap import (
    EMOTION_MAP,
    find_evaluation_files,
    parse_evaluations,
    parse_transcript,
    slice_face_frames,
)
from emotion_ssm.utils.paths import ensure_output_directory


def test_iemocap_mapping_and_vad(tmp_path: Path):
    transcript = tmp_path / "transcript.txt"
    transcript.write_text("Ses05F_x_F000 [0.00-1.00]: hello\n", encoding="utf-8")
    evaluation = tmp_path / "eval.txt"
    evaluation.write_text(
        "[0.00 - 1.00] Ses05F_x_F000 exc [5.0, 3.0, 1.0]\n"
        "[1.00 - 2.00] Ses05F_x_M001 fru [2.0, 4.0, 3.0]\n",
        encoding="utf-8",
    )
    values = parse_evaluations(evaluation, parse_transcript(transcript))
    assert EMOTION_MAP["exc"] == EMOTION_MAP["hap"] == 3
    assert values[0]["vad"] == [1.0, 0.0, -1.0]
    assert values[1]["emotion_id"] == -1
    assert values[1]["vad_mask"] == [True, True, True]


def test_iemocap_evaluation_discovery_ignores_appledouble(tmp_path: Path):
    evaluation_root = tmp_path / "Session1" / "dialog" / "EmoEvaluation"
    evaluation_root.mkdir(parents=True)
    real = evaluation_root / "Ses01F_impro01.txt"
    real.write_text("annotation", encoding="utf-8")
    (evaluation_root / "._Ses01F_impro01.txt").write_text(
        "metadata", encoding="utf-8"
    )

    assert find_evaluation_files(tmp_path) == [real]


def test_au_time_slice_uses_best_face():
    frames = [
        {"timestamp": 0.5, "confidence": 0.7, "success": 1, "au": [1.0] * 35},
        {"timestamp": 0.5, "confidence": 0.9, "success": 1, "au": [2.0] * 35},
        {"timestamp": 1.5, "confidence": 0.95, "success": 1, "au": [3.0] * 35},
    ]
    au, confidence, valid = slice_face_frames(frames, 0.0, 1.0, 0.8)
    assert au.shape == (1, 35)
    assert torch.all(au == 2)
    assert confidence.item() == pytest.approx(0.9)
    assert valid.item()


def test_feature_store_and_l33_window(feature_root: Path):
    store = FeatureDialogueStore(feature_root, "emotiontalk", 0)
    vocab = SpeakerVocabulary(store.collect_speakers("train"))
    utterances = UnifiedUtteranceDataset([store], "train", vocab)
    windows = DialogueWindowDataset([store], "train", vocab, window_length=33, stride=4)
    assert len(utterances) == 4
    assert windows[0]["audio"].shape == (33, 768)
    assert windows[0]["valid_mask"].sum().item() == 4
    assert not windows[0]["vad_mask"][:, 2].any()


def test_feature_store_accepts_declared_partner_without_utterance(tmp_path: Path):
    root = tmp_path / "features"
    dialogue = root / "dialogues" / "single_active"
    dialogue.mkdir(parents=True)
    torch.save(torch.randn(2, 768), dialogue / "audio_features.pt")
    torch.save(torch.randn(2, 768), dialogue / "text_features.pt")
    torch.save(torch.randn(2, 35), dialogue / "face_au_features.pt")
    labels = {
        "speaker_map": {"01": "A", "02": "B"},
        "speaker_ids": ["01", "02"],
        "utterances": [
            {
                "utterance_id": f"utt_{turn}",
                "speaker_id": "02",
                "speaker": "B",
                "emotion_id": 4,
                "sentiment_score": 0.0,
                "intensity_abs": 0.0,
                "start_time": float(turn),
                "end_time": float(turn) + 0.5,
            }
            for turn in range(2)
        ],
    }
    (dialogue / "labels.json").write_text(json.dumps(labels), encoding="utf-8")
    splits = root / "splits"
    splits.mkdir()
    (splits / "train_dialogues.txt").write_text(
        "single_active\n", encoding="utf-8"
    )

    store = FeatureDialogueStore(root, "emotiontalk", 0)
    vocab = SpeakerVocabulary(store.collect_speakers("train"))
    record = store.load_dialogue("single_active", vocab)

    assert len(vocab) == 2
    assert record.active_role.tolist() == [1, 1]
    assert record.speaker_ids.tolist() == [
        vocab.encode("emotiontalk:01"),
        vocab.encode("emotiontalk:02"),
    ]


def test_dualtalk_wav_npz_fixture(tmp_path: Path):
    sample_rate = 16000
    frames = 5
    for role in ("speaker1", "speaker2"):
        stem = f"tiny_{role}"
        np.savez(
            tmp_path / f"{stem}.npz",
            exp=np.zeros((frames, 50), dtype=np.float32),
            pose=np.zeros((frames, 6), dtype=np.float32),
        )
        wavfile.write(
            tmp_path / f"{stem}.wav",
            sample_rate,
            np.zeros(sample_rate // 5, dtype=np.int16),
        )
    dataset = DualTalkChunkDataset(tmp_path, chunk_frames=5, fps=25)
    assert len(dataset) == 2
    assert dataset[0]["target_blendshape"].shape == (5, 56)


def test_output_guard_rejects_dataset_directory(tmp_path: Path):
    data_root = tmp_path / "datasets"
    data_root.mkdir()
    with pytest.raises(ValueError):
        ensure_output_directory(data_root / "generated", [data_root])
