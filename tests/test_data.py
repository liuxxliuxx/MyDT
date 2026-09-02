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
