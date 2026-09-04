from __future__ import annotations

import csv
import json
import re
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import librosa
import torch
from torch import Tensor

from emotion_ssm.config import parse_config_args
from emotion_ssm.data.common import EMOTION_NAMES, load_pt
from emotion_ssm.utils.paths import ensure_output_directory


AU_COLUMNS = (
    "AU01_r", "AU02_r", "AU04_r", "AU05_r", "AU06_r", "AU07_r",
    "AU09_r", "AU10_r", "AU12_r", "AU14_r", "AU15_r", "AU17_r",
    "AU20_r", "AU23_r", "AU25_r", "AU26_r", "AU45_r", "AU01_c",
    "AU02_c", "AU04_c", "AU05_c", "AU06_c", "AU07_c", "AU09_c",
    "AU10_c", "AU12_c", "AU14_c", "AU15_c", "AU17_c", "AU20_c",
    "AU23_c", "AU25_c", "AU26_c", "AU28_c", "AU45_c",
)

EMOTION_MAP = {
    "ang": 0,
    "dis": 1,
    "fea": 2,
    "hap": 3,
    "exc": 3,
    "neu": 4,
    "sad": 5,
    "sur": 6,
}

EVALUATION_RE = re.compile(
    r"^\[(?P<start>[0-9.]+)\s*-\s*(?P<end>[0-9.]+)\]\s*"
    r"(?P<utterance>\S+)\s+(?P<emotion>[a-zA-Z]+)\s+"
    r"\[(?P<vad>[^\]]+)\]"
)
TRANSCRIPT_RE = re.compile(
    r"^(?P<utterance>\S+)\s+\[(?P<start>[0-9.]+)-(?P<end>[0-9.]+)\]:\s*(?P<text>.*)$"
)


def parse_transcript(path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = TRANSCRIPT_RE.match(line.strip())
        if match:
            values[match.group("utterance")] = match.group("text").strip()
    return values


def parse_evaluations(path: Path, transcript: Mapping[str, str]) -> List[Dict]:
    utterances = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = EVALUATION_RE.match(line.strip())
        if not match:
            continue
        raw_vad = [float(value.strip()) for value in match.group("vad").split(",")]
        if len(raw_vad) != 3:
            continue
        vad = [max(-1.0, min((value - 3.0) / 2.0, 1.0)) for value in raw_vad]
        utterance_id = match.group("utterance")
        role_match = re.search(r"_([FM])\d+$", utterance_id)
        if role_match is None:
            continue
        sex = role_match.group(1)
        session_match = re.match(r"Ses(\d{2})", utterance_id)
        if session_match is None:
            continue
        speaker_id = f"Ses{session_match.group(1)}{sex}"
        emotion_code = match.group("emotion").lower()
        utterances.append(
            {
                "utterance_id": utterance_id,
                "speaker_id": speaker_id,
                "speaker": "A" if sex == "F" else "B",
                "active_role": 0 if sex == "F" else 1,
                "text": transcript.get(utterance_id, ""),
                "emotion_code": emotion_code,
                "emotion_id": EMOTION_MAP.get(emotion_code, -1),
                "emotion": (
                    EMOTION_NAMES[EMOTION_MAP[emotion_code]]
                    if emotion_code in EMOTION_MAP
                    else emotion_code
                ),
                "vad": vad,
                "vad_mask": [True, True, True],
                "intensity": (vad[1] + 1.0) / 2.0,
                "start_time": float(match.group("start")),
                "end_time": float(match.group("end")),
            }
        )
    utterances.sort(key=lambda item: (item["start_time"], item["utterance_id"]))
    for turn, item in enumerate(utterances):
        item["turn_id"] = turn
        item["turn_position"] = turn / max(len(utterances) - 1, 1)
    return utterances


def find_evaluation_files(raw_root: Path) -> List[Path]:
    """Return real IEMOCAP annotations, excluding macOS AppleDouble files."""
    return sorted(
        path
        for path in raw_root.glob("Session*/dialog/EmoEvaluation/*.txt")
        if not path.name.startswith(".")
    )


def find_video(session_root: Path, dialogue_id: str) -> Optional[Path]:
    video_root = session_root / "dialog" / "avi" / "DivX"
    candidates = []
    for extension in ("avi", "mp4", "mov", "mkv"):
        candidates.extend(video_root.glob(f"{dialogue_id}*.{extension}"))
    return sorted(candidates)[0] if candidates else None


def run_openface(binary: str, video: Path, output_dir: Path) -> Optional[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    expected = output_dir / f"{video.stem}.csv"
    if expected.exists():
        return expected
    command = [binary, "-f", str(video), "-out_dir", str(output_dir), "-aus"]
    subprocess.run(command, check=True)
    return expected if expected.exists() else None


def require_external_tool(command: str, purpose: str) -> str:
    resolved = shutil.which(command)
    if resolved is None and Path(command).is_file():
        resolved = str(Path(command).resolve())
    if resolved is None:
        raise FileNotFoundError(
            f"{purpose} executable was not found: {command}. "
            "Set its absolute path in the preprocessing YAML."
        )
    return resolved


def read_openface_csv(path: Optional[Path]) -> List[Dict[str, object]]:
    if path is None or not path.exists():
        return []
    frames: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            row = {str(key).strip(): value for key, value in raw.items()}
            try:
                values = [float(row.get(column, "0") or 0.0) for column in AU_COLUMNS]
                frames.append(
                    {
                        "timestamp": float(row.get("timestamp", "0") or 0.0),
                        "confidence": float(row.get("confidence", "0") or 0.0),
                        "success": int(float(row.get("success", "0") or 0.0)),
                        "au": values,
                    }
                )
            except ValueError:
                continue
    return frames


def slice_face_frames(
    frames: Sequence[Mapping[str, object]],
    start: float,
    end: float,
    confidence_threshold: float,
) -> Tuple[Tensor, Tensor, Tensor]:
    # 多人 CSV 同一时间戳可能有多行；保留置信度最高的一张脸。
    best_by_timestamp: Dict[float, Mapping[str, object]] = {}
    for frame in frames:
        timestamp = float(frame["timestamp"])
        if timestamp < start or timestamp > end:
            continue
        current = best_by_timestamp.get(timestamp)
        if current is None or float(frame["confidence"]) > float(current["confidence"]):
            best_by_timestamp[timestamp] = frame
    selected = [best_by_timestamp[key] for key in sorted(best_by_timestamp)]
    if not selected:
        return (
            torch.zeros(1, len(AU_COLUMNS), dtype=torch.float32),
            torch.zeros(1, dtype=torch.float32),
            torch.zeros(1, dtype=torch.bool),
        )
    au = torch.tensor([frame["au"] for frame in selected], dtype=torch.float32)
    confidence = torch.tensor(
        [float(frame["confidence"]) for frame in selected], dtype=torch.float32
    )
    success = torch.tensor(
        [bool(frame["success"]) for frame in selected], dtype=torch.bool
    )
    valid = success & (confidence >= confidence_threshold)
    return au, confidence, valid


class FeatureExtractors:
    def __init__(self, cfg) -> None:
        from transformers import AutoModel, AutoTokenizer, Wav2Vec2Model, Wav2Vec2Processor

        self.device = torch.device(cfg.PREPROCESS.DEVICE)
        local_only = bool(cfg.PREPROCESS.LOCAL_FILES_ONLY)
        self.audio_processor = Wav2Vec2Processor.from_pretrained(
            cfg.PREPROCESS.AUDIO_MODEL, local_files_only=local_only
        )
        self.audio_model = Wav2Vec2Model.from_pretrained(
            cfg.PREPROCESS.AUDIO_MODEL, local_files_only=local_only
        ).to(self.device).eval()
        self.text_tokenizer = AutoTokenizer.from_pretrained(
            cfg.PREPROCESS.TEXT_MODEL, local_files_only=local_only
        )
        self.text_model = AutoModel.from_pretrained(
            cfg.PREPROCESS.TEXT_MODEL, local_files_only=local_only
        ).to(self.device).eval()
        self.batch_size = int(cfg.PREPROCESS.BATCH_SIZE)

    @torch.no_grad()
    def encode_audio(self, paths: Sequence[Path]) -> Tensor:
        outputs = []
        for start in range(0, len(paths), self.batch_size):
            waves = [librosa.load(str(path), sr=16000)[0] for path in paths[start:start + self.batch_size]]
            inputs = self.audio_processor(
                waves, sampling_rate=16000, return_tensors="pt", padding=True
            )
            input_values = inputs.input_values.to(self.device)
            attention_mask = getattr(inputs, "attention_mask", None)
            model_output = self.audio_model(
                input_values,
                attention_mask=None if attention_mask is None else attention_mask.to(self.device),
            ).last_hidden_state
            if attention_mask is not None and hasattr(
                self.audio_model, "_get_feature_vector_attention_mask"
            ):
                mask = self.audio_model._get_feature_vector_attention_mask(
                    model_output.shape[1], attention_mask.to(self.device)
                )
                pooled = (model_output * mask[:, :, None]).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
            else:
                pooled = model_output.mean(dim=1)
            outputs.append(pooled.cpu())
        result = torch.cat(outputs)
        if result.shape[-1] != 768:
            raise ValueError(f"Audio encoder must output 768 dimensions, got {result.shape[-1]}")
        return result

    @torch.no_grad()
    def encode_text(self, texts: Sequence[str]) -> Tensor:
        outputs = []
        for start in range(0, len(texts), self.batch_size):
            inputs = self.text_tokenizer(
                list(texts[start:start + self.batch_size]),
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            inputs = {name: value.to(self.device) for name, value in inputs.items()}
            hidden = self.text_model(**inputs).last_hidden_state
            mask = inputs["attention_mask"].float()
            pooled = (hidden * mask[:, :, None]).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
            outputs.append(pooled.cpu())
        result = torch.cat(outputs)
        if result.shape[-1] != 768:
            raise ValueError(f"Text encoder must output 768 dimensions, got {result.shape[-1]}")
        return result


def session_number(dialogue_id: str) -> int:
    match = re.match(r"Ses(\d{2})", dialogue_id)
    if match is None:
        raise ValueError(f"Cannot infer IEMOCAP session from {dialogue_id}")
    return int(match.group(1))


def write_splits(output_root: Path, dialogue_ids: Sequence[str]) -> None:
    by_session: Dict[int, List[str]] = defaultdict(list)
    for dialogue_id in dialogue_ids:
        by_session[session_number(dialogue_id)].append(dialogue_id)
    for test_session in range(1, 6):
        val_session = 5 if test_session == 1 else test_session - 1
        split_root = output_root / "splits" / f"fold_{test_session}"
        split_root.mkdir(parents=True, exist_ok=True)
        assignments = {
            "test": sorted(by_session[test_session]),
            "val": sorted(by_session[val_session]),
            "train": sorted(
                dialogue
                for session, dialogues in by_session.items()
                if session not in {test_session, val_session}
                for dialogue in dialogues
            ),
        }
        for split, values in assignments.items():
            (split_root / f"{split}_dialogues.txt").write_text(
                "\n".join(values) + "\n", encoding="utf-8"
            )


def compute_fold_face_stats(output_root: Path, fold: int) -> None:
    train_file = output_root / "splits" / f"fold_{fold}" / "train_dialogues.txt"
    dialogue_ids = [value for value in train_file.read_text(encoding="utf-8").splitlines() if value]
    total = torch.zeros(len(AU_COLUMNS), dtype=torch.float64)
    total_square = torch.zeros_like(total)
    count = 0
    for dialogue_id in dialogue_ids:
        face = load_pt(output_root / "dialogues" / dialogue_id / "face_au_features.pt")
        for au, valid in zip(face["au_sequences"], face["frame_valid_masks"]):
            values = au[valid].double()
            if len(values) == 0:
                continue
            total += values.sum(0)
            total_square += values.square().sum(0)
            count += len(values)
    if count == 0:
        mean = torch.zeros(len(AU_COLUMNS))
        std = torch.ones(len(AU_COLUMNS))
    else:
        mean = (total / count).float()
        variance = total_square / count - mean.double().square()
        std = variance.clamp_min(1e-8).sqrt().float()
    metadata = output_root / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"mean": mean, "std": std, "count": count},
        metadata / f"fold_{fold}_face_stats.pt",
    )


def preprocess(cfg) -> None:
    raw_root = Path(cfg.DATA.IEMOCAP_RAW_ROOT)
    if not raw_root.is_dir():
        raise FileNotFoundError(f"IEMOCAP raw root does not exist: {raw_root}")
    output_root = ensure_output_directory(
        cfg.DATA.IEMOCAP_FEATURE_ROOT,
        [
            cfg.DATA.ROOT,
            raw_root,
            cfg.DATA.EMOTIONTALK_ROOT,
            cfg.DATA.DUALTALK_ROOT,
        ],
    )
    openface_binary = str(cfg.PREPROCESS.OPENFACE_BIN)
    if not cfg.PREPROCESS.SKIP_OPENFACE:
        openface_binary = require_external_tool(openface_binary, "OpenFace")
        require_external_tool(str(cfg.PREPROCESS.FFMPEG_BIN), "FFmpeg")
    extractors = FeatureExtractors(cfg)
    errors = []
    completed = []
    for evaluation_file in find_evaluation_files(raw_root):
        dialogue_id = evaluation_file.stem
        session_root = evaluation_file.parents[2]
        output_dialogue = output_root / "dialogues" / dialogue_id
        required = [
            output_dialogue / "audio_features.pt",
            output_dialogue / "text_features.pt",
            output_dialogue / "face_au_features.pt",
            output_dialogue / "labels.json",
        ]
        if all(path.exists() for path in required):
            completed.append(dialogue_id)
            continue
        try:
            transcript_path = session_root / "dialog" / "transcriptions" / f"{dialogue_id}.txt"
            transcript = parse_transcript(transcript_path)
            utterances = parse_evaluations(evaluation_file, transcript)
            if not utterances:
                raise ValueError("No evaluable utterances")
            wav_paths = [
                session_root / "sentences" / "wav" / dialogue_id / f"{item['utterance_id']}.wav"
                for item in utterances
            ]
            missing = [str(path) for path in wav_paths if not path.exists()]
            if missing:
                raise FileNotFoundError(f"Missing utterance WAV files: {missing[:3]}")
            audio = extractors.encode_audio(wav_paths)
            text = extractors.encode_text([str(item["text"]) for item in utterances])

            openface_csv = None
            if not cfg.PREPROCESS.SKIP_OPENFACE:
                video = find_video(session_root, dialogue_id)
                if video is not None:
                    openface_csv = run_openface(
                        openface_binary,
                        video,
                        output_root / "openface_csv" / dialogue_id,
                    )
            frames = read_openface_csv(openface_csv)
            au_sequences = []
            confidence_sequences = []
            frame_valid_masks = []
            for item in utterances:
                au, confidence, valid = slice_face_frames(
                    frames,
                    float(item["start_time"]),
                    float(item["end_time"]),
                    float(cfg.PREPROCESS.CONFIDENCE_THRESHOLD),
                )
                au_sequences.append(au)
                confidence_sequences.append(confidence)
                frame_valid_masks.append(valid)

            output_dialogue.mkdir(parents=True, exist_ok=True)
            torch.save(audio, output_dialogue / "audio_features.pt")
            torch.save(text, output_dialogue / "text_features.pt")
            torch.save(
                {
                    "dialogue_id": dialogue_id,
                    "au_columns": list(AU_COLUMNS),
                    "utterance_ids": [item["utterance_id"] for item in utterances],
                    "au_sequences": au_sequences,
                    "confidence_sequences": confidence_sequences,
                    "frame_valid_masks": frame_valid_masks,
                    "utterance_valid_mask": torch.tensor(
                        [bool(mask.any()) for mask in frame_valid_masks]
                    ),
                    "normalized": False,
                },
                output_dialogue / "face_au_features.pt",
            )
            labels = {
                "dialogue_id": dialogue_id,
                "speaker_ids": sorted({item["speaker_id"] for item in utterances}),
                "utterances": utterances,
            }
            (output_dialogue / "labels.json").write_text(
                json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            completed.append(dialogue_id)
        except Exception as error:
            errors.append({"dialogue_id": dialogue_id, "error": repr(error)})

    if not completed:
        raise RuntimeError("IEMOCAP preprocessing produced no dialogues")
    write_splits(output_root, completed)
    for fold in range(1, 6):
        compute_fold_face_stats(output_root, fold)
    metadata = output_root / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    (metadata / "preprocess_summary.json").write_text(
        json.dumps(
            {"completed_dialogues": len(completed), "errors": errors},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    if errors:
        (metadata / "preprocess_errors.jsonl").write_text(
            "\n".join(json.dumps(item, ensure_ascii=False) for item in errors) + "\n",
            encoding="utf-8",
        )


def main() -> None:
    cfg, _ = parse_config_args("Preprocess official IEMOCAP into the shared feature layout")
    preprocess(cfg)


if __name__ == "__main__":
    main()
