from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import librosa
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


def partner_stem(stem: str) -> str:
    if stem.endswith("speaker1"):
        return stem[:-8] + "speaker2"
    if stem.endswith("speaker2"):
        return stem[:-8] + "speaker1"
    raise ValueError(f"DualTalk sample does not end in speaker1/speaker2: {stem}")


def normalize_waveform(waveform: np.ndarray) -> Tensor:
    value = torch.from_numpy(waveform).float()
    return (value - value.mean()) / value.std().clamp_min(1e-6)


def speech_is_active(waveform: np.ndarray, rms_threshold: float) -> bool:
    """Energy VAD on the unnormalised chunk, before silence loses its scale."""
    if waveform.size == 0:
        return False
    finite = np.nan_to_num(waveform.astype(np.float32, copy=False))
    rms = float(np.sqrt(np.mean(np.square(finite), dtype=np.float64)))
    return rms >= float(rms_threshold)


class DualTalkChunkDataset(Dataset):
    """Read paired DualTalk WAV/FLAME files and expose fixed 200-frame chunks."""

    def __init__(
        self,
        root: Path,
        chunk_frames: int = 200,
        fps: int = 25,
        include_both_directions: bool = True,
        speech_rms_threshold: float = 1e-4,
    ) -> None:
        self.root = Path(root)
        self.chunk_frames = chunk_frames
        self.fps = fps
        self.samples: List[Dict[str, object]] = []
        target_files = sorted(self.root.glob("*.npz"))
        if not include_both_directions:
            target_files = [path for path in target_files if path.stem.endswith("speaker1")]

        for target_npz in target_files:
            partner = partner_stem(target_npz.stem)
            target_wav = target_npz.with_suffix(".wav")
            partner_npz = self.root / f"{partner}.npz"
            partner_wav = self.root / f"{partner}.wav"
            if not target_wav.exists() or not partner_npz.exists() or not partner_wav.exists():
                continue
            with np.load(str(target_npz)) as target_flame:
                target_bs = np.concatenate(
                    [
                        target_flame["exp"],
                        target_flame["pose"][:, 3:],
                        target_flame["pose"][:, :3],
                    ],
                    axis=-1,
                )
            with np.load(str(partner_npz)) as partner_flame:
                partner_bs = np.concatenate(
                    [
                        partner_flame["exp"],
                        partner_flame["pose"][:, 3:],
                        partner_flame["pose"][:, :3],
                    ],
                    axis=-1,
                )
            target_audio, _ = librosa.load(str(target_wav), sr=16000)
            partner_audio, _ = librosa.load(str(partner_wav), sr=16000)
            num_frames = min(len(target_bs), len(partner_bs))
            num_chunks = num_frames // chunk_frames
            audio_samples = int(chunk_frames / fps * 16000)
            for chunk in range(num_chunks):
                frame_start = chunk * chunk_frames
                audio_start = chunk * audio_samples
                target_chunk = target_audio[audio_start : audio_start + audio_samples]
                partner_chunk = partner_audio[audio_start : audio_start + audio_samples]
                self.samples.append(
                    {
                        "name": target_npz.stem,
                        "chunk": chunk,
                        "target_audio": normalize_waveform(target_chunk),
                        "partner_audio": normalize_waveform(partner_chunk),
                        "target_speech_active": torch.tensor(
                            speech_is_active(target_chunk, speech_rms_threshold)
                        ),
                        "partner_speech_active": torch.tensor(
                            speech_is_active(partner_chunk, speech_rms_threshold)
                        ),
                        "target_blendshape": torch.from_numpy(
                            target_bs[frame_start : frame_start + chunk_frames]
                        ).float(),
                        "partner_blendshape": torch.from_numpy(
                            partner_bs[frame_start : frame_start + chunk_frames]
                        ).float(),
                        "dt": torch.tensor(chunk_frames / fps, dtype=torch.float32),
                    }
                )
        if not self.samples:
            raise ValueError(f"No complete DualTalk pairs found in {self.root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, object]:
        return self.samples[index]


class DualTalkDialogueDataset(DualTalkChunkDataset):
    """Return every consecutive chunk for one directed DualTalk dialogue.

    A sequence is keyed by the target stream name, therefore the two output
    directions of a conversation remain separate state trajectories.  Chunks
    are sorted and checked here instead of relying on a shuffled DataLoader to
    happen to emit adjacent chunks in temporal order.
    """

    def __init__(
        self,
        root: Path,
        chunk_frames: int = 200,
        fps: int = 25,
        include_both_directions: bool = True,
        speech_rms_threshold: float = 1e-4,
    ) -> None:
        super().__init__(
            root, chunk_frames, fps, include_both_directions, speech_rms_threshold
        )
        groups: Dict[str, List[Dict[str, object]]] = {}
        for sample in self.samples:
            groups.setdefault(str(sample["name"]), []).append(sample)
        self.dialogues: List[List[Dict[str, object]]] = []
        for name in sorted(groups):
            chunks = sorted(groups[name], key=lambda item: int(item["chunk"]))
            indices = [int(item["chunk"]) for item in chunks]
            if indices != list(range(len(chunks))):
                raise ValueError(f"DualTalk chunks for {name} are not consecutive")
            self.dialogues.append(chunks)

    def __len__(self) -> int:
        return len(self.dialogues)

    def __getitem__(self, index: int) -> Dict[str, object]:
        chunks = self.dialogues[index]
        tensor_fields = (
            "target_audio",
            "partner_audio",
            "target_blendshape",
            "partner_blendshape",
            "dt",
            "target_speech_active",
            "partner_speech_active",
        )
        item = {
            name: torch.stack([chunk[name] for chunk in chunks])
            for name in tensor_fields
        }
        item.update(
            {
                "name": str(chunks[0]["name"]),
                "chunk_indices": torch.tensor(
                    [int(chunk["chunk"]) for chunk in chunks], dtype=torch.long
                ),
            }
        )
        return item


def collate_dualtalk_dialogues(samples: Sequence[Dict[str, object]]) -> Dict[str, object]:
    """Pad only the chunk axis while preserving every per-chunk tensor value."""
    if not samples:
        raise ValueError("Cannot collate an empty DualTalk dialogue batch")
    length = max(int(sample["dt"].shape[0]) for sample in samples)
    tensor_fields = (
        "target_audio",
        "partner_audio",
        "target_blendshape",
        "partner_blendshape",
        "dt",
        "target_speech_active",
        "partner_speech_active",
        "chunk_indices",
    )
    output: Dict[str, object] = {}
    for name in tensor_fields:
        values = [sample[name] for sample in samples]
        shape = (len(samples), length) + tuple(values[0].shape[1:])
        fill = -1 if name == "chunk_indices" else 0
        padded = torch.full(shape, fill, dtype=values[0].dtype)
        for row, value in enumerate(values):
            padded[row, : value.shape[0]] = value
        output[name] = padded
    mask = torch.zeros(len(samples), length, dtype=torch.bool)
    for row, sample in enumerate(samples):
        mask[row, : sample["dt"].shape[0]] = True
    output["chunk_mask"] = mask
    output["name"] = [str(sample["name"]) for sample in samples]
    return output
