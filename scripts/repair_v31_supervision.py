"""Migrate a V3.1 IEMOCAP token root with proxy intensity supervision masked.

Frozen features are reused; this command does not load extractors or raw media.
The resulting manifests require a new experiment and optimizer, not a resume.
Completed dialogues are checksummed so an interrupted migration can be rerun.
EmotionTalk and DualTalk roots must be reused directly by the new pipeline.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emotion_ssm.data.packets_v3 import TOKEN_PROTOCOL, SUPERVISION_REVISION, validate_packet
from emotion_ssm.data.protocol import DOMAINS, fingerprint


PROXY_SOURCE = "arousal_proxy_for_candidate_matching_only"
STATE_FILE = ".supervision-repair.json"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path, writer, *, replace=False):
    """Publish a complete file; existing artifacts must have identical bytes."""
    path = Path(path)
    if path.is_symlink() or (path.exists() and path.stat().st_nlink > 1):
        raise ValueError(f"Destination file must not be a link: {path}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=".repair-", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        digest = file_sha256(temporary)
        if path.exists() and not replace:
            if file_sha256(path) != digest:
                raise ValueError(f"Refusing to overwrite a different destination artifact: {path}")
        else:
            temporary.replace(path)
        return digest
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(path, value, *, replace=False):
    encoded = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    return _atomic_write(path, lambda stream: stream.write(encoded), replace=replace)


def _check_manifest(manifest):
    if manifest.get("protocol") != TOKEN_PROTOCOL or manifest.get("errors"):
        raise ValueError("Source must be a complete V3.1 token manifest")
    expected = fingerprint({key: value for key, value in manifest.items() if key != "digest"})
    if manifest.get("digest") != expected:
        raise ValueError("Source token manifest digest mismatch")
    dialogues = manifest.get("dialogues", {})
    if not dialogues:
        raise ValueError("Source token manifest has no dialogues")
    assigned = set()
    for names in manifest.get("splits", {}).values():
        for name in names:
            if name in assigned or name not in dialogues:
                raise ValueError("Source splits contain duplicate or unknown dialogues")
            assigned.add(name)
    if assigned != set(dialogues):
        raise ValueError("Every source dialogue must belong to exactly one split")
    for name, record in dialogues.items():
        dataset = record.get("dataset")
        if dataset not in DOMAINS or name.partition(":")[0] != dataset:
            raise ValueError(f"Source dialogue lacks a consistent explicit dataset identity: {name}")
        if dataset not in manifest.get("feature_sources", {}):
            raise ValueError(f"Source dialogue lacks its declared feature source: {name}")
    if any(record["dataset"] != "iemocap" for record in dialogues.values()):
        raise ValueError("Only an IEMOCAP token root can be repaired; reuse ET/DualTalk roots unchanged")


def _check_payload(payload, name, record, feature_source):
    if (payload.get("protocol") != TOKEN_PROTOCOL or payload.get("cache_id") != record["cache_id"]
            or payload.get("feature_source") != feature_source):
        raise ValueError(f"Source token cache provenance mismatch: {name}")
    expected_id = fingerprint({"protocol": TOKEN_PROTOCOL, "source": feature_source,
                               "audit": payload["audit"], "dialogue": name})
    if expected_id != payload["cache_id"]:
        raise ValueError(f"Source cache_id does not bind its declared audit: {name}")
    if len(payload["packets"]) != record["packets"]:
        raise ValueError(f"Source packet count mismatch: {name}")
    for packet in payload["packets"]:
        if (packet.get("dialogue_id") != name
                or packet.get("source_domain") != DOMAINS[record["dataset"]]
                or len(packet.get("targets", [])) != 2
                or any(int(role["domain_id"]) != packet["source_domain"] for role in packet["roles"])):
            raise ValueError(f"Source packet identity/role metadata mismatch: {name}")
        validate_packet(packet)


def _mask_iemocap_proxy(payload, name, source_manifest_sha256, source_payload_sha256):
    """Verify the legacy arousal proxy formula; never infer labels from a filename."""
    targets_before = fingerprint([packet["targets"] for packet in payload["packets"]])
    source_cache_id = payload["cache_id"]
    checked = changed = 0
    for packet in payload["packets"]:
        for labels in packet["targets"]:
            for label in labels:
                vad, mask = label.get("vad", []), label.get("vad_mask", [])
                if (len(vad) != 3 or len(mask) != 3 or not mask[1]
                        or not isinstance(label.get("intensity_mask"), bool)
                        or label.get("intensity_source") not in (None, PROXY_SOURCE)):
                    raise ValueError(f"Unverified IEMOCAP intensity semantics: {name}/{label.get('utterance_id', '')}")
                arousal, intensity = float(vad[1]), float(label["intensity"])
                if (not math.isfinite(arousal) or not math.isfinite(intensity) or not -1 <= arousal <= 1
                        or not math.isclose(intensity, (arousal + 1.) / 2., rel_tol=1e-6, abs_tol=1e-7)):
                    raise ValueError(f"IEMOCAP intensity is not the verified arousal proxy: {name}/{label.get('utterance_id', '')}")
                checked += 1
                changed += int(label["intensity_mask"])
                label["intensity_mask"] = False
                label["intensity_source"] = PROXY_SOURCE
    payload["supervision_revision"] = SUPERVISION_REVISION
    payload["audit"]["supervision_revision"] = SUPERVISION_REVISION
    # The original label_digest still describes the unchanged raw annotations.
    # A separate target digest binds the corrected endpoint masks explicitly.
    payload["audit"]["supervision_repair"] = {
        "revision": SUPERVISION_REVISION, "source_manifest_sha256": source_manifest_sha256,
        "source_cache_id": source_cache_id, "source_payload_sha256": source_payload_sha256,
        "source_targets_digest": targets_before,
        "targets_digest": fingerprint([packet["targets"] for packet in payload["packets"]]),
        "intensity_policy": PROXY_SOURCE, "labels_checked": checked, "masks_disabled": changed,
    }
    payload["cache_id"] = fingerprint({"protocol": TOKEN_PROTOCOL, "source": payload["feature_source"],
                                       "audit": payload["audit"], "dialogue": name})
    return checked, changed


def repair_supervision(source_root, output_root, *, emit=None):
    """Stream one dialogue at a time into a new or matching resumable directory."""
    source_root = Path(source_root).resolve(strict=True)
    output_root = Path(output_root).resolve()
    if (source_root == output_root or source_root in output_root.parents or output_root in source_root.parents):
        raise ValueError("Source and destination paths must not overlap")
    source_manifest_path = source_root / "manifest.json"
    source_bytes = source_manifest_path.read_bytes()
    source_sha = hashlib.sha256(source_bytes).hexdigest()
    source = json.loads(source_bytes)
    _check_manifest(source)
    source_paths = {}
    for name, record in source["dialogues"].items():
        # Respect explicitly declared external payload paths, including shards.
        path = (source_root / record["path"]).resolve(strict=True)
        if output_root == path or output_root in path.parents:
            raise ValueError(f"Destination overlaps a source shard artifact: {path}")
        if not path.is_file():
            raise ValueError(f"Source artifact is not a file: {path}")
        source_paths[name] = path
    binding = {"revision": SUPERVISION_REVISION, "protocol": TOKEN_PROTOCOL,
               "source_root": str(source_root), "source_manifest_sha256": source_sha,
               "output_root": str(output_root), "requires_new_experiment": True,
               "requires_new_optimizer": True, "resume_from_old_checkpoint": False}
    state_path = output_root / STATE_FILE
    if output_root.exists():
        if not output_root.is_dir():
            raise ValueError("Destination is not a directory")
        if state_path.is_symlink():
            raise ValueError("Destination repair state must not be a link")
        if state_path.exists():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if state.get("binding") != binding:
                raise ValueError("Destination belongs to a different source or supervision revision")
        elif any(output_root.iterdir()):
            raise ValueError("Destination must be new, empty, or a matching repair directory")
        else:
            state = {"binding": binding, "dialogues": {}}
    else:
        output_root.mkdir(parents=True)
        state = {"binding": binding, "dialogues": {}}
    _write_json(state_path, state, replace=state_path.exists())
    dialogues_root = output_root / "dialogues"
    if dialogues_root.is_symlink() or dialogues_root.resolve() != output_root / "dialogues":
        raise ValueError("Destination dialogues directory must not be a link")
    dialogues_root.mkdir(exist_ok=True)
    manifest = copy.deepcopy(source)
    manifest.pop("digest", None)
    manifest["supervision_revision"] = SUPERVISION_REVISION
    manifest["supervision_repair"] = dict(binding)
    report = {"dialogues": len(source_paths), "iemocap_labels_checked": 0, "intensity_masks_disabled": 0}
    log = emit if emit is not None else lambda event: print(json.dumps(event, ensure_ascii=False), flush=True)
    for completed, (name, record) in enumerate(source["dialogues"].items(), 1):
        path = source_paths[name]
        source_file_sha = file_sha256(path)
        relative = "dialogues/" + fingerprint(name) + ".pt"
        destination = output_root / relative
        if destination.is_symlink() or (destination.exists() and destination.stat().st_nlink > 1):
            raise ValueError(f"Destination artifact must not be a link: {destination}")
        receipt = state["dialogues"].get(name)
        if receipt is not None:
            if (receipt["source_payload_sha256"] != source_file_sha
                    or receipt["record"]["path"] != relative
                    or not destination.is_file()
                    or file_sha256(destination) != receipt["output_payload_sha256"]):
                raise ValueError(f"Previously migrated artifact changed; choose a fresh destination: {name}")
        else:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            _check_payload(payload, name, record, source["feature_sources"][record["dataset"]])
            checked, changed = _mask_iemocap_proxy(payload, name, source_sha, source_file_sha)
            output_sha = _atomic_write(destination, lambda stream: torch.save(payload, stream))
            if file_sha256(path) != source_file_sha:
                raise ValueError(f"Source artifact changed during migration: {name}")
            new_record = {**record, "path": relative, "cache_id": payload["cache_id"]}
            receipt = {"source_payload_sha256": source_file_sha, "output_payload_sha256": output_sha,
                       "record": new_record, "labels_checked": checked, "masks_disabled": changed}
            state["dialogues"][name] = receipt
            _write_json(state_path, state, replace=True)
            del payload
        manifest["dialogues"][name] = receipt["record"]
        report["iemocap_labels_checked"] += receipt["labels_checked"]
        report["intensity_masks_disabled"] += receipt["masks_disabled"]
        log({"stage": "supervision_repair", "dialogue": name, "completed": completed,
             "total": len(source_paths), "labels_checked": receipt["labels_checked"],
             "masks_disabled": receipt["masks_disabled"]})
    if source_manifest_path.read_bytes() != source_bytes:
        raise ValueError("Source manifest changed during migration")
    manifest["supervision_repair"].update(report)
    manifest["digest"] = fingerprint(manifest)
    # Published last: absence of manifest.json means the destination is incomplete.
    _write_json(output_root / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, help="Existing read-only V3.1 IEMOCAP token root")
    parser.add_argument("--output-root", required=True, help="New disjoint token root, or an interrupted matching repair")
    args = parser.parse_args()
    manifest = repair_supervision(args.source_root, args.output_root)
    print(json.dumps({"status": "complete", "output_root": str(Path(args.output_root).resolve()),
                      "manifest_digest": manifest["digest"], **manifest["supervision_repair"]}), flush=True)


if __name__ == "__main__":
    main()
