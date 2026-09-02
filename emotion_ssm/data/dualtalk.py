from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

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


class DualTalkChunkDataset(Dataset):
    """Read paired DualTalk WAV/FLAME files and expose fixed 200-frame chunks."""

    def __init__(
        self,
        root: Path,
        chunk_frames: int = 200,
        fps: int = 25,
        include_both_directions: bool = True,
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
                self.samples.append(
                    {
                        "name": target_npz.stem,
                        "chunk": chunk,
                        "target_audio": normalize_waveform(
                            target_audio[audio_start : audio_start + audio_samples]
                        ),
                        "partner_audio": normalize_waveform(
                            partner_audio[audio_start : audio_start + audio_samples]
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
