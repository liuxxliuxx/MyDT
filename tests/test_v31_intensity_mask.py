import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from emotion_ssm.data.packets_v3 import (SUPERVISION_REVISION, TOKEN_PROTOCOL, TokenPacketDataset,
                                        collate_endpoint_labels, endpoint_label,
                                        validate_supervision_manifest)
from emotion_ssm.preprocess.iemocap import parse_evaluations
from emotion_ssm.train.observation_v3 import globally_weighted_supervision


@pytest.mark.parametrize("dataset,field,raw,normalized", [
    ("emotiontalk", "intensity_abs", 1.5, .75),
    ("annotated_fixture", "intensity", .75, .75),
])
@pytest.mark.parametrize("metadata,expected", [
    ({}, True),
    ({"intensity_mask": True}, True),
    ({"intensity_valid": True}, True),
    ({"intensity_mask": False}, False),
    ({"intensity_valid": False}, False),
    ({"intensity_mask": True, "intensity_valid": False}, False),
    ({"intensity_mask": False, "intensity_valid": True}, False),
    ({"intensity_mask": False, "intensity_valid": False}, False),
    ({"intensity_source": "arousal_proxy_for_candidate_matching_only"}, False),
    ({"intensity_source": "arousal_proxy_for_candidate_matching_only",
      "intensity_mask": True, "intensity_valid": True}, False),
    ({"intensity_source": "human_annotation"}, True),
])
def test_intensity_supervision_respects_source_and_both_validity_flags(
        dataset, field, raw, normalized, metadata, expected):
    item = {"start_time": 0., "end_time": 1., field: raw, **metadata}
    result = endpoint_label(item, dataset)
    assert result["intensity"] == normalized
    assert result["intensity_mask"] is expected
    assert result["intensity_source"] == metadata.get("intensity_source")


@pytest.mark.parametrize("dataset", ["emotiontalk", "iemocap"])
@pytest.mark.parametrize("metadata", [
    {}, {"intensity_mask": True, "intensity_valid": True},
])
def test_intensity_flags_do_not_create_a_missing_label(dataset, metadata):
    result = endpoint_label({"start_time": 0., "end_time": 1., **metadata}, dataset)
    assert result["intensity"] == 0.
    assert result["intensity_mask"] is False


@pytest.mark.parametrize("dataset,field", [
    ("emotiontalk", "intensity_abs"), ("annotated_fixture", "intensity"),
])
def test_explicit_zero_intensity_is_a_valid_legacy_label(dataset, field):
    result = endpoint_label({"start_time": 0., "end_time": 1., field: 0.}, dataset)
    assert result["intensity"] == 0.
    assert result["intensity_mask"] is True


@pytest.mark.parametrize("metadata,expected", [
    ({}, False),
    ({"intensity_mask": True}, False),
    ({"intensity_valid": True}, False),
    ({"intensity_mask": True, "intensity_valid": True}, False),
    ({"intensity_source": "independent_annotation"}, False),
    ({"intensity_source": "independent_annotation", "intensity_mask": True}, True),
    ({"intensity_source": "independent_annotation", "intensity_mask": True,
      "intensity_valid": False}, False),
    ({"intensity_source": "independent_annotation", "intensity_mask": False,
      "intensity_valid": True}, False),
    ({"intensity_source": "arousal_proxy_for_candidate_matching_only",
      "intensity_mask": True, "intensity_valid": True}, False),
])
def test_iemocap_requires_explicit_independent_intensity_provenance(metadata, expected):
    # Matches legacy labels.json: intensity is the arousal proxy, with no masks.
    item = {"start_time": 0., "end_time": 1., "vad": [-.5, .5, 0.],
            "intensity": .75, **metadata}
    result = endpoint_label(item, "iemocap")
    assert result["intensity"] == .75
    assert result["intensity_mask"] is expected
    assert result["intensity_source"] == metadata.get(
        "intensity_source", "arousal_proxy_for_candidate_matching_only")
    assert result["vad"] == [-.5, .5, 0.]
    assert result["vad_mask"] == [True, True, True]


def test_iemocap_parser_proxy_keeps_vad_but_never_supervises_intensity(tmp_path):
    annotation = tmp_path / "evaluation.txt"
    annotation.write_text(
        "[0.0000 - 0.5000]\tSes01F_impro01_F000\tfru\t[2.0000, 4.0000, 3.0000]\n",
        encoding="utf-8",
    )
    record, = parse_evaluations(annotation, {})
    assert record["intensity"] == .75
    assert record["intensity_mask"] is False
    assert record["intensity_source"] == "arousal_proxy_for_candidate_matching_only"

    result = endpoint_label(record, "iemocap")
    assert result["emotion"] == -1
    assert result["vad"] == [-.5, .5, 0.]
    assert result["vad_mask"] == [True, True, True]
    assert result["intensity"] == .75
    assert result["intensity_mask"] is False
    assert result["intensity_source"] == record["intensity_source"]
    assert collate_endpoint_labels([result])["intensity_mask"].tolist() == [False]


def test_iemocap_proxy_mask_removes_intensity_gradient_but_preserves_vad_supervision():
    class ControlledObserver(nn.Module):
        def __init__(self):
            super().__init__()
            self.intensity = nn.Parameter(torch.tensor(.25))
            self.vad = nn.Parameter(torch.zeros(3))

        def encode(self, features, subset=None):
            return {"valid": torch.ones(len(features["domain_id"]), dtype=torch.bool),
                    "observation": SimpleNamespace(aff=features["x"])}

        def decode_affect(self, affect):
            return {"emotion_logits": affect.new_zeros(len(affect), 7),
                    "intensity": self.intensity.expand(len(affect)),
                    "vad": self.vad.expand(len(affect), 3)}

    observer = ControlledObserver()
    features = {"x": torch.ones(1, 1), "domain_id": torch.tensor([1])}
    # This older raw IEMOCAP shape carries an arousal proxy without provenance.
    raw = {"start_time": 0., "end_time": 1., "emotion": "fru",
           "vad": [-.5, .5, 0.], "intensity": .75}
    corrected = endpoint_label(raw, "iemocap")
    assert corrected["intensity_mask"] is False
    legacy = {**corrected, "intensity_mask": True}

    old_loss = globally_weighted_supervision(observer, features, collate_endpoint_labels([legacy]))
    old_intensity_grad, old_vad_grad = torch.autograd.grad(
        old_loss["total"], (observer.intensity, observer.vad))
    new_loss = globally_weighted_supervision(observer, features, collate_endpoint_labels([corrected]))
    new_intensity_grad, new_vad_grad = torch.autograd.grad(
        new_loss["total"], (observer.intensity, observer.vad))

    assert old_loss["intensity_weight_count"].item() == 1
    assert old_loss["intensity"].item() > 0
    assert old_intensity_grad.abs().item() > 0
    assert new_loss["intensity_weight_count"].item() == 0
    assert new_loss["intensity"].item() == 0
    assert new_intensity_grad.item() == 0
    assert new_loss["vad_weight_count"].item() == 3
    assert new_loss["vad"].item() > 0
    assert new_vad_grad.abs().sum().item() > 0
    torch.testing.assert_close(new_loss["vad"], old_loss["vad"])
    torch.testing.assert_close(new_vad_grad, old_vad_grad)


@pytest.mark.parametrize("declaration", [
    {"feature_sources": {"iemocap": {}}},
    {"feature_sources": {"emotiontalk": {}}, "dialogues": {"example": {"dataset": "iemocap"}}},
])
@pytest.mark.parametrize("revision", [None, "endpoint-intensity-mask-old"])
def test_manifest_rejects_stale_iemocap_supervision(declaration, revision):
    manifest = dict(declaration)
    if revision is not None:
        manifest["supervision_revision"] = revision
    with pytest.raises(ValueError, match="IEMOCAP.*supervision_revision"):
        validate_supervision_manifest(manifest)


@pytest.mark.parametrize("declaration", [
    {"feature_sources": {"iemocap": {}}},
    {"dialogues": {"example": {"dataset": "iemocap"}}},
])
def test_manifest_accepts_current_iemocap_supervision(declaration):
    validate_supervision_manifest({**declaration, "supervision_revision": SUPERVISION_REVISION})


@pytest.mark.parametrize("source", ["emotiontalk", "dualtalk"])
def test_unaffected_legacy_manifest_still_loads(source, tmp_path):
    manifest = {"protocol": TOKEN_PROTOCOL, "feature_sources": {source: {}},
                "splits": {"train": []}, "dialogues": {"example": {"dataset": source}}}
    validate_supervision_manifest(manifest)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert len(TokenPacketDataset(tmp_path)) == 0


def test_dataset_rejects_old_iemocap_before_loading_ids(tmp_path):
    manifest = {"protocol": TOKEN_PROTOCOL, "feature_sources": {"iemocap": {}}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="IEMOCAP.*supervision_revision"):
        TokenPacketDataset(tmp_path)


def test_dataset_loads_iemocap_with_current_supervision_revision(tmp_path):
    manifest = {"protocol": TOKEN_PROTOCOL, "supervision_revision": SUPERVISION_REVISION,
                "feature_sources": {"iemocap": {}}, "splits": {"train": []}, "dialogues": {}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert len(TokenPacketDataset(tmp_path)) == 0
