"""Build source-group splits and prefix-only audio/text features for DualTalk."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from emotion_ssm.data.protocol import PROTOCOL_VERSION, fingerprint, make_split_manifest
from emotion_ssm.data.dualtalk import partner_stem, speech_is_active
from emotion_ssm.data.timed_dualtalk import load_flame
from emotion_ssm.models.features import FrozenFeatures
from emotion_ssm.preprocess.ctc_align import TranscriptAligner


def prepare(root, output, source, device="cpu", seed=6666, rebuild=False, limit=0, words_root=None, speech_threshold=1e-4):
    import librosa
    from emotion_ssm.utils.paths import ensure_output_directory
    root = Path(root)
    output = ensure_output_directory(output, [root])
    manifest = make_split_manifest([p.stem for p in (root / "train").glob("*.npz")], seed)
    manifest_path = output / "splits.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8")) != manifest and not rebuild:
        raise ValueError("Split manifest changed; choose a new artifact directory")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    extractor = FrozenFeatures(source).to(device)
    from emotion_ssm.data.protocol import require_compatible
    require_compatible(source, extractor.source)
    aligner = None if words_root else TranscriptAligner(device=device)
    settings = {"source": source, "version": PROTOCOL_VERSION, "fps": 25, "max_tokens": 256,
                "speech_rms_threshold": speech_threshold,
                "alignment_model": None if aligner is None else aligner.model_name,
                "text_protocol": "external_asr" if words_root else "aligned_transcript"}
    errors, completed = [], 0
    cache_ids = {}
    for split in ("train", "test", "ood"):
        (output / split).mkdir(exist_ok=True)
        for path in sorted((root / split).glob("*speaker1.npz")):
            stems = (path.stem, partner_stem(path.stem))
            source_files = [root / split / (name + extension) for name in stems for extension in (".wav", ".npz", ".txt")]
            raw_signature = [(p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in source_files if p.exists()]
            words_files = [Path(words_root) / split / (name + ".json") for name in stems] if words_root else []
            words_signature = [(p.name, p.read_text(encoding="utf-8")) for p in words_files if p.exists()]
            cache_id = fingerprint({"settings": settings, "files": raw_signature, "words": words_signature})
            for name in stems:
                cache_ids[f"{split}/{name}"] = cache_id
            destinations = [output / split / (name + ".pt") for name in stems]
            if all(p.exists() for p in destinations) and not rebuild:
                if any(torch.load(p, weights_only=False)["cache_id"] != cache_id for p in destinations):
                    raise ValueError(f"Stale features for {path.stem}; rebuild explicitly")
                continue
            waves = [librosa.load(str(root / split / (name + ".wav")), sr=16000)[0] for name in stems]
            words = []
            for name, wave, role in zip(stems, waves, ("speaker1", "speaker2")):
                try:
                    if words_root:
                        words.extend(json.loads((Path(words_root) / split / (name + ".json")).read_text(encoding="utf-8")))
                    else:
                        text = (root / split / (name + ".txt")).read_text(encoding="utf-8")
                        words.extend(aligner.align(wave, text, role))
                except (ValueError, FileNotFoundError) as error:
                    errors.append({"name": name, "text_masked": True, "error": str(error)})
            frames = min(*(len(load_flame(root / split / (name + ".npz"))) for name in stems),
                         int(min(map(len, waves)) / 16000 * 25))
            chunks = []
            for start in range(0, frames, 25):
                stop = min(start+25, frames)
                pair = []
                for role, wave in zip(("speaker1", "speaker2"), waves):
                    raw = wave[round(start / 25 * 16000):round(stop / 25 * 16000)]
                    values = extractor(torch.from_numpy(raw)[None].to(device), words, stop/25, start/25,
                                       role, speech_is_active(raw, speech_threshold))
                    values = {k: v.cpu() for k, v in values.items()}
                    values["available_at"] = stop / 25
                    values["audio_length"] = len(raw)
                    values["valid_frames"] = stop-start
                    pair.append(values)
                chunks.append(pair)
            for direction, destination in enumerate(destinations):
                torch.save({**settings, "feature_source": source, "cache_id": cache_id,
                            "dialogue_id": path.stem.removesuffix("_speaker1"),
                            "roles": ["speaker1", "speaker2"], "observed_until": frames/25,
                            "raw_signature": raw_signature, "words_digest": fingerprint(words_signature),
                            "aligned_words": words,
                            "chunks": [{"target": pair[direction], "partner": pair[1-direction]} for pair in chunks]}, destination)
            completed += 1
            if limit and completed >= limit:
                break
        if limit and completed >= limit:
            break
    (output / "preparation_report.json").write_text(json.dumps({"completed_pairs": completed, "errors": errors,
        "text_protocol": settings["text_protocol"], "split_digest": manifest["digest"],
        "cross_file_continuity": "unavailable: no verified dialogue mapping/timestamps; reset per paired clip",
        "independent_dualtalk_emotion_labels": "not provided"}, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "dataset_manifest.json").write_text(json.dumps({"version": 2, "cache_ids": cache_ids,
        "digest": fingerprint(cache_ids), "complete": not limit}, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--words-root", type=Path)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    from emotion_ssm.config import load_config
    from emotion_ssm.utils.generation_checkpoint import source_for_config
    cfg = load_config(args.config, args.opts)
    _, source = source_for_config(cfg)
    prepare(cfg.DATA.DUALTALK_ROOT, args.output, source, args.device, cfg.DUALTALK.SPLIT_SEED,
            args.rebuild, args.limit, args.words_root, cfg.DUALTALK.SPEECH_RMS_THRESHOLD)


if __name__ == "__main__":
    main()
