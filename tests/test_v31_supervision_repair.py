from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import pytest
import torch

from emotion_ssm.data.packets_v3 import (TOKEN_PROTOCOL, SUPERVISION_REVISION,
                                         TokenPacketDataset, empty_role_features)
from emotion_ssm.data.protocol import DOMAINS, fingerprint
from scripts import repair_v31_supervision as repair


def write_source(root, datasets=("iemocap", "emotiontalk", "dualtalk")):
    root.mkdir()
    (root / "dialogues").mkdir()
    manifest = {"protocol": TOKEN_PROTOCOL, "feature_sources": {},
                "splits": {name: [] for name in ("train", "val", "test", "ood")},
                "dialogues": {}, "quality": {"unchanged": True}, "data_gaps": [], "errors": []}
    for index, dataset in enumerate(datasets):
        name = f"{dataset}:opaque-{index}"
        source = {"audio_model": "fixture-audio", "text_model": "fixture-text", "audio_dim": 8, "text_dim": 8}
        manifest["feature_sources"][dataset] = source
        roles = [empty_role_features(8, 8, now=1., domain_id=DOMAINS[dataset]) for _ in (0, 1)]
        # Storage sharing survives serialization without rebuilding feature tensors.
        shared = torch.arange(24, dtype=torch.float32).reshape(3, 8)
        for role, values in enumerate(roles):
            values["text_tokens"] = shared[role:role + 1]
            values["text_mask"].fill_(True)
        labels = [[{"start": .1, "end": .5, "emotion": 3, "raw_emotion": "happy",
                    "intensity": .25, "intensity_mask": True, "vad": [.6, -.5, .1],
                    "vad_mask": [True, True, True], "utterance_id": "known-utterance"}],
                  [{"start": .4, "end": .75, "emotion": -1, "raw_emotion": "fru",
                    "intensity": .4, "intensity_mask": False, "intensity_source": None, "vad": [.2, -.2, .3],
                    "vad_mask": [True, True, True], "utterance_id": "unmapped-emotion"}]]
        if dataset == "dualtalk":
            labels = [[], []]
        packet = {"protocol": TOKEN_PROTOCOL, "dialogue_id": name, "source_domain": DOMAINS[dataset],
                  "start": 0., "end": 1., "dt": 1., "roles": roles, "targets": labels,
                  "text_protocol": "offline_transcript_endpoint"}
        audit = {"label_digest": fingerprint({"raw_labels": index}), "raw_signature": [], "speakers": ["A", "B"]}
        cache_id = fingerprint({"protocol": TOKEN_PROTOCOL, "source": source, "audit": audit, "dialogue": name})
        filename = "dialogues/" + fingerprint(name) + ".pt"
        torch.save({"protocol": TOKEN_PROTOCOL, "cache_id": cache_id, "feature_source": source,
                    "packets": [packet], "audit": audit}, root / filename)
        manifest["dialogues"][name] = {"path": filename, "cache_id": cache_id, "packets": 1, "dataset": dataset}
        manifest["splits"]["train"].append(name)
    manifest["digest"] = fingerprint(manifest)
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def load_payload(root, manifest, name):
    return torch.load(root / manifest["dialogues"][name]["path"], map_location="cpu", weights_only=False)


def assert_same(left, right):
    if torch.is_tensor(left):
        assert left.dtype == right.dtype and left.shape == right.shape
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for name in left:
            assert_same(left[name], right[name])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right) and len(left) == len(right)
        for old, new in zip(left, right):
            assert_same(old, new)
    else:
        assert left == right


def mutate_source(root, manifest, change):
    name = next(iter(manifest["dialogues"]))
    path = root / manifest["dialogues"][name]["path"]
    payload = load_payload(root, manifest, name)
    change(payload)
    torch.save(payload, path)


def test_repair_only_proxy_masks_and_preserve_all_features_without_reading_other_domains(tmp_path, monkeypatch):
    source_root, output = tmp_path / "source", tmp_path / "new"
    source = write_source(source_root, ("iemocap",))
    write_source(tmp_path / "other-domains", ("emotiontalk", "dualtalk"))
    before = {path: repair.file_sha256(path) for path in source_root.rglob("*") if path.is_file()}
    original_load = torch.load
    loaded = []

    def record_load(path, *args, **kwargs):
        loaded.append(Path(path).resolve())
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr(torch, "load", record_load)
    result = repair.repair_supervision(source_root, output, emit=lambda _: None)
    assert loaded == [(source_root / source["dialogues"]["iemocap:opaque-0"]["path"]).resolve()]
    assert result["supervision_revision"] == SUPERVISION_REVISION
    assert result["digest"] == fingerprint({key: value for key, value in result.items() if key != "digest"})
    assert result["digest"] != source["digest"]
    assert result["splits"] == source["splits"] and result["quality"] == source["quality"]
    provenance = result["supervision_repair"]
    assert provenance["source_manifest_sha256"] == before[source_root / "manifest.json"]
    assert provenance["iemocap_labels_checked"] == 2 and provenance["intensity_masks_disabled"] == 1
    assert provenance["requires_new_optimizer"] and not provenance["resume_from_old_checkpoint"]
    for name, record in source["dialogues"].items():
        old, new = load_payload(source_root, source, name), load_payload(output, result, name)
        expected = copy.deepcopy(old["packets"])
        for packet in expected:
            for labels in packet["targets"]:
                for label in labels:
                    label["intensity_mask"] = False
                    label["intensity_source"] = repair.PROXY_SOURCE
        assert_same(new["packets"], expected)
        assert new["feature_source"] == old["feature_source"]
        assert new["audit"]["label_digest"] == old["audit"]["label_digest"]
        assert new["cache_id"] != old["cache_id"]
        audit = new["audit"]["supervision_repair"]
        assert audit["source_cache_id"] == old["cache_id"]
        assert audit["targets_digest"] == fingerprint([packet["targets"] for packet in expected])
        assert new["cache_id"] == fingerprint({"protocol": TOKEN_PROTOCOL, "source": new["feature_source"],
                                               "audit": new["audit"], "dialogue": name})
        roles = new["packets"][0]["roles"]
        assert roles[0]["text_tokens"].untyped_storage().data_ptr() == roles[1]["text_tokens"].untyped_storage().data_ptr()
    assert before == {path: repair.file_sha256(path) for path in before}
    assert len(list((output / "dialogues").glob("*.pt"))) == 1
    dataset = TokenPacketDataset(output, "train")
    assert len(dataset) == 1
    for index in range(len(dataset)):
        dataset[index]
    with pytest.raises(ValueError, match="supervision_revision"):
        TokenPacketDataset(source_root, "train")


def test_reject_mixed_roots_before_reading_payloads_or_creating_output(tmp_path, monkeypatch):
    source, output = tmp_path / "source", tmp_path / "new"
    write_source(source)

    def forbidden(*args, **kwargs):
        raise AssertionError("Mixed-domain payloads must not be loaded")

    monkeypatch.setattr(torch, "load", forbidden)
    with pytest.raises(ValueError, match="Only an IEMOCAP"):
        repair.repair_supervision(source, output)
    assert not output.exists()


@pytest.mark.parametrize("target", ["same", "child", "ancestor"])
def test_reject_overlapping_roots_before_writing(tmp_path, target):
    source = tmp_path / "source"
    write_source(source)
    destination = {"same": source, "child": source / "new", "ancestor": tmp_path}[target]
    with pytest.raises(ValueError, match="overlap"):
        repair.repair_supervision(source, destination)
    assert not (destination / repair.STATE_FILE).exists()


def test_reject_destination_containing_referenced_shard(tmp_path):
    source_root, shard = tmp_path / "source", tmp_path / "shard"
    source = write_source(source_root, ("iemocap",))
    shard.mkdir()
    name = next(iter(source["dialogues"]))
    original = source_root / source["dialogues"][name]["path"]
    moved = shard / "payload.pt"
    original.rename(moved)
    source["dialogues"][name]["path"] = os.path.relpath(moved, source_root)
    source.pop("digest")
    source["digest"] = fingerprint(source)
    (source_root / "manifest.json").write_text(json.dumps(source), encoding="utf-8")
    with pytest.raises(ValueError, match="source shard"):
        repair.repair_supervision(source_root, shard)
    assert not (shard / repair.STATE_FILE).exists()


@pytest.mark.parametrize("change", [
    lambda p: p["packets"][0]["targets"][0][0].update(intensity=.9),
    lambda p: p["packets"][0]["targets"][0][0].update(vad_mask=[True, False, True]),
    lambda p: p["packets"][0]["targets"][0][0].update(intensity_source="independent_intensity"),
])
def test_reject_unverified_proxy_semantics_without_complete_manifest(tmp_path, change):
    source_root, output = tmp_path / "source", tmp_path / "new"
    source = write_source(source_root, ("iemocap",))
    mutate_source(source_root, source, change)
    before = repair.file_sha256(source_root / source["dialogues"][next(iter(source["dialogues"]))]["path"])
    with pytest.raises(ValueError, match="IEMOCAP intensity"):
        repair.repair_supervision(source_root, output, emit=lambda _: None)
    assert not (output / "manifest.json").exists()
    assert before == repair.file_sha256(source_root / source["dialogues"][next(iter(source["dialogues"]))]["path"])


def test_reject_unowned_destination_and_changed_source_manifest(tmp_path):
    source, output = tmp_path / "source", tmp_path / "new"
    manifest = write_source(source, ("iemocap",))
    output.mkdir()
    sentinel = output / "user-file.txt"
    sentinel.write_text("keep me")
    with pytest.raises(ValueError, match="new, empty"):
        repair.repair_supervision(source, output)
    assert sentinel.read_text() == "keep me"
    destination = tmp_path / "repair"
    repair.repair_supervision(source, destination, emit=lambda _: None)
    manifest["quality"]["new_source_metadata"] = True
    manifest.pop("digest")
    manifest["digest"] = fingerprint(manifest)
    (source / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="different source"):
        repair.repair_supervision(source, destination)


def test_interrupted_migration_resumes_and_completed_rerun_keeps_payload_and_manifest_bytes(tmp_path, monkeypatch):
    source, output = tmp_path / "source", tmp_path / "new"
    manifest = write_source(source, ("iemocap", "iemocap"))
    original_save = torch.save
    calls = 0

    def interrupted_save(payload, stream, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated interruption")
        return original_save(payload, stream, *args, **kwargs)

    monkeypatch.setattr(torch, "save", interrupted_save)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        repair.repair_supervision(source, output, emit=lambda _: None)
    assert not (output / "manifest.json").exists()
    completed = list((output / "dialogues").glob("*.pt"))
    assert len(completed) == 1
    completed_stat = completed[0].stat().st_mtime_ns
    monkeypatch.setattr(torch, "save", original_save)
    result = repair.repair_supervision(source, output, emit=lambda _: None)
    assert result["supervision_repair"]["intensity_masks_disabled"] == 2
    assert completed[0].stat().st_mtime_ns == completed_stat
    files = [output / "manifest.json", *list((output / "dialogues").glob("*.pt"))]
    before = {path: (repair.file_sha256(path), path.stat().st_mtime_ns) for path in files}
    again = repair.repair_supervision(source, output, emit=lambda _: None)
    assert again == result
    assert before == {path: (repair.file_sha256(path), path.stat().st_mtime_ns) for path in files}
    assert not list(output.rglob("*.tmp"))


def test_completed_rerun_rejects_tampered_destination_and_source_payload(tmp_path):
    source, output = tmp_path / "source", tmp_path / "new"
    manifest = write_source(source, ("iemocap",))
    result = repair.repair_supervision(source, output, emit=lambda _: None)
    name = next(iter(result["dialogues"]))
    destination = output / result["dialogues"][name]["path"]
    original = destination.read_bytes()
    destination.write_bytes(original + b"tamper")
    with pytest.raises(ValueError, match="Previously migrated artifact changed"):
        repair.repair_supervision(source, output)
    destination.write_bytes(original)
    mutate_source(source, manifest, lambda p: p["packets"][0]["roles"][0]["text_tokens"].add_(1))
    with pytest.raises(ValueError, match="Previously migrated artifact changed"):
        repair.repair_supervision(source, output)


def test_interruption_between_payload_and_receipt_reuses_identical_orphan(tmp_path, monkeypatch):
    source, output = tmp_path / "source", tmp_path / "new"
    write_source(source, ("iemocap",))
    original_write = repair._write_json

    def interrupted_write(path, value, **kwargs):
        if Path(path).name == repair.STATE_FILE and value["dialogues"]:
            raise RuntimeError("interrupted before receipt")
        return original_write(path, value, **kwargs)

    monkeypatch.setattr(repair, "_write_json", interrupted_write)
    with pytest.raises(RuntimeError, match="before receipt"):
        repair.repair_supervision(source, output, emit=lambda _: None)
    orphan = next((output / "dialogues").glob("*.pt"))
    before = (repair.file_sha256(orphan), orphan.stat().st_mtime_ns)
    monkeypatch.setattr(repair, "_write_json", original_write)
    repair.repair_supervision(source, output, emit=lambda _: None)
    assert (repair.file_sha256(orphan), orphan.stat().st_mtime_ns) == before
