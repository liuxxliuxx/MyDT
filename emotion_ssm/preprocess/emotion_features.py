"""Rebuild causal A/T features from explicit WAV mappings, labels and verified AU."""
from __future__ import annotations

import argparse
import copy
import json
import shutil
from pathlib import Path

import torch

from emotion_ssm.data.common import load_pt, read_json
from emotion_ssm.data.feature_store import resolve_utterances, resolve_speaker_id
from emotion_ssm.data.protocol import FeatureSource, fingerprint, role_text
from emotion_ssm.models.features import FrozenFeatures
from emotion_ssm.utils.paths import ensure_output_directory


def causal_texts(utterances, dialogue_id):
    """Sentence transcripts become available only at their declared endpoint.

    This protocol is explicitly offline_transcript_endpoint, distinct from
    word-aligned DualTalk and externally timestamped ASR.
    """
    speakers = sorted({resolve_speaker_id(item, dialogue_id) for item in utterances})
    if len(speakers) > 2:
        raise ValueError(f"{dialogue_id}: more than two speakers")
    words = []
    for index, item in enumerate(utterances):
        start, end = float(item["start_time"]), float(item["end_time"])
        available = float(item.get("text_available_at", end))
        if not __import__("math").isfinite(start+end+available) or start < 0 or end < start or available < end:
            raise ValueError(f"{dialogue_id}: invalid timestamps at {index}")
        text = str(item.get("text", item.get("transcript", ""))).strip()
        if text:
            words.append({"id": index, "start": start, "end": end, "available_at": available,
                          "role": "AB"[speakers.index(resolve_speaker_id(item, dialogue_id))], "text": text})
    contexts, events, masks = [], [], []
    for index, item in enumerate(utterances):
        now = float(item["end_time"])
        visible = sorted((w for w in words if w["available_at"] <= now),
                         key=lambda w: (w["available_at"], w["id"]))
        own = [w for w in visible if w["id"] == index]
        contexts.append(role_text(visible))
        events.append(role_text(own))
        masks.append((bool(visible), bool(own)))
    return contexts, events, masks


def prepare(input_root, output_root, audio_manifest, source, dataset, device="cpu", rebuild=False):
    import librosa
    input_root = Path(input_root).resolve()
    output_root = ensure_output_directory(output_root, [input_root])
    mapping = read_json(Path(audio_manifest))
    extractor = FrozenFeatures(source).to(device)
    source = extractor.source
    input_dialogues = input_root / "dialogues"
    if not input_dialogues.is_dir():
        input_dialogues = input_root
    errors, completed = [], []
    for labels_path in sorted(input_dialogues.glob("*/labels.json")):
        name, old = labels_path.parent.name, labels_path.parent
        labels = read_json(labels_path)
        utterances = resolve_utterances(labels)
        output = output_root / "dialogues" / name
        paths = []
        try:
            for index, item in enumerate(utterances):
                uid = str(item.get("utterance_id", item.get("id", index)))
                value = mapping.get(f"{name}/{uid}", mapping.get(uid))
                if value is None:
                    raise ValueError(f"WAV manifest has no explicit path for {name}/{uid}")
                paths.append(Path(value).resolve(strict=True))
            context_text, event_text, masks = causal_texts(utterances, name)
            files = paths + [labels_path, old / "face_au_features.pt"]
            cache_id = fingerprint({"source": source, "labels": labels,
                "files": [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in files]})
            if (output / "provenance.json").exists() and not rebuild:
                if read_json(output / "provenance.json")["cache_id"] != cache_id:
                    raise ValueError("Cache changed; use a new output or --rebuild")
                completed.append(name)
                continue
            audio_values, context_values, event_values = [], [], []
            for item, path, context, event, (has_context, has_event) in zip(utterances, paths, context_text, event_text, masks):
                wave = torch.from_numpy(librosa.load(str(path), sr=16000)[0])[None]
                # Single utterance extraction uses exactly the deployment
                # normalization and valid-length pooling implementation.
                from emotion_ssm.data.dualtalk import speech_is_active
                active = speech_is_active(wave.numpy(), 1e-4)
                values = extractor(wave.to(device), [], 0., 0., "A", speech_active=active)
                audio_values.append(values["audio"].cpu())
                context_values.append(extractor.encode_text(context).cpu())
                event_values.append(extractor.encode_text(event).cpu())
                item.update(audio_present=bool(values["modality_mask"][0, 0]), context_present=has_context, event_present=has_event,
                            text_available_at=float(item.get("text_available_at", item["end_time"])))
            face = load_pt(old / "face_au_features.pt")
            verified = read_json(old / "provenance.json").get("face_identity_verified", False) if (old / "provenance.json").exists() else False
            if dataset == "iemocap" and not verified:
                # Unknown identities must never be assigned to the active role.
                face = {"utterance_ids": [str(i.get("utterance_id", i.get("id", n))) for n, i in enumerate(utterances)],
                        "au_sequences": [torch.zeros(1, 35) for _ in utterances],
                        "confidence_sequences": [torch.zeros(1) for _ in utterances],
                        "frame_valid_masks": [torch.zeros(1, dtype=torch.bool) for _ in utterances], "normalized": True}
            output.mkdir(parents=True, exist_ok=True)
            # Store a normalized common label envelope without changing input.
            updated = {"dialogue_id": name, "utterances": utterances}
            for key in ("speaker_ids", "role_speakers", "speakers"):
                if isinstance(labels, dict) and key in labels:
                    updated[key] = labels[key]
            for filename, values in (("audio", audio_values), ("text", context_values), ("event_text", event_values)):
                torch.save(torch.cat(values), output / f"{filename}_features.pt")
            torch.save(face, output / "face_au_features.pt")
            (output / "labels.json").write_text(json.dumps(updated, ensure_ascii=False, indent=2), encoding="utf-8")
            provenance = {"cache_id": cache_id, "feature_source": source, "labels_digest": fingerprint(updated),
                          "text_protocol": "offline_transcript_endpoint", "face_identity_verified": verified}
            (output / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
            completed.append(name)
            if len(completed) % 25 == 0:
                print(json.dumps({"dataset": dataset, "completed_dialogues": len(completed), "errors": len(errors)}), flush=True)
        except (ValueError, FileNotFoundError, KeyError) as error:
            errors.append({"dialogue": name, "error": str(error)})
    for directory in ("splits", "metadata"):
        if (input_root / directory).is_dir():
            shutil.copytree(input_root / directory, output_root / directory, dirs_exist_ok=True)
    (output_root / "metadata").mkdir(exist_ok=True)
    (output_root / "metadata" / "feature_source.json").write_text(json.dumps(source, indent=2), encoding="utf-8")
    (output_root / "preparation_report.json").write_text(json.dumps({"completed": completed, "errors": errors}, ensure_ascii=False, indent=2), encoding="utf-8")
    if errors or not completed:
        raise RuntimeError("Feature preparation has gaps; inspect preparation_report.json before training")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input-root", "output-root", "audio-manifest", "audio-model", "text-model", "dataset"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    if args.dataset not in ("emotiontalk", "iemocap"):
        parser.error("dataset must be emotiontalk or iemocap")
    prepare(args.input_root, args.output_root, args.audio_manifest,
            FeatureSource(args.audio_model, args.text_model).to_dict(), args.dataset, args.device, args.rebuild)


if __name__ == "__main__":
    main()
