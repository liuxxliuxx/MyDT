from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from emotion_ssm.data.packets_v3 import collate_role_features, collate_endpoint_labels
from emotion_ssm.models.token_observer import SUBSETS, TokenObserver
from emotion_ssm.train.observation_v3 import (DialogueBalancedBatches, MixedDialogueDataset, build_training_index,
    clone_dropout_diagnostics, confusion_metrics, globally_weighted_supervision, semantic_gate,
    initialize_dualtalk_adapters, evaluate, calibration_objective)
from test_token_observer_v3 import features, model, fake_source, labelled_provenance, write_cache


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class FakeDialogues:
    def __init__(self, domain, count=8):
        self.ids = [f"source{domain}:dialogue{i}" for i in range(count)]
        self.domain, self.loads = domain, 0

    def __getitem__(self, index):
        self.loads += 1
        packets = []
        for second in range(4):
            roles, targets = [], []
            for role in (0, 1):
                feature = features(index * 17 + second * 2 + role)
                feature["domain_id"] = torch.tensor(self.domain)
                roles.append(feature)
                labels = []
                if self.domain != 2 and second in (1, 3):
                    frustrated = self.domain == 1 and index == 0 and second == 1 and role == 0
                    labels = [{"end": 1., "emotion": -1 if frustrated else (index+role)%2,
                               "raw_emotion": "fru" if frustrated else str((index+role)%2),
                               "vad": [.5, 0., 0.], "vad_mask": [True, False, False],
                               "intensity": 0., "intensity_mask": False}]
                targets.append(labels)
            packets.append({"roles": roles, "targets": targets})
        return {"packets": packets}


class FakePackets:
    def __init__(self, domain, count=8):
        self.dialogues = FakeDialogues(domain, count)
        self.index = [(dialogue, packet, role) for dialogue in range(count) for packet in range(4) for role in (0, 1)]

    def __len__(self):
        return len(self.index)


def test_train_index_preserves_sparse_endpoints_and_frustration_vad():
    data = [FakePackets(0), FakePackets(1), FakePackets(2)]
    index = build_training_index(data)
    assert index["split"] == "train"
    assert len(index["rows"]) == 192
    assert index["domains"]["0"]["endpoint_count"] == 32
    assert index["domains"]["1"]["raw_class_counts"]["fru"] == 1
    assert index["domains"]["1"]["unknown_class_count"] == 1
    assert sum(index["domains"]["1"]["class_counts"]) == 31
    assert index["domains"]["1"]["vad_count"][0] == 32
    assert index["domains"]["2"]["class_counts"] == [0]*7
    assert sum(bool(row["supervised"]) for row in index["rows"]) == 64
    assert index["domains"]["0"]["class_weights"][:2] == [1., 1.]
    assert index["domains"]["0"]["vad_mean"] == [.5, None, None]


def test_mixed_sampler_has_real_supervision_roles_and_dialogues_on_both_ranks():
    datasets = [FakePackets(i) for i in range(3)]
    index = build_training_index(datasets)
    lookup = {row["index"]: row for row in index["rows"]}
    ranks = [list(DialogueBalancedBatches(datasets, 8, rank=rank, world_size=2, training_index=index)) for rank in (0, 1)]
    for batches in zip(*ranks):
        assert not set(batches[0]) & set(batches[1])
        for rows in batches:
            assert len({lookup[i]["dialogue"] for i in rows}) >= 3
            assert {lookup[i]["role"] for i in rows} == {0, 1}
            assert sum(bool(lookup[i]["supervised"]) for i in rows) >= 4
    # A single cached dialogue serves its many packets without repeated loads.
    wrapper = MixedDialogueDataset(datasets, cache_dialogues=12)
    before = datasets[0].dialogues.loads
    for index in range(8):
        wrapper[index]
    assert datasets[0].dialogues.loads == before+1


class TinySupervision(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Parameter(torch.arange(21, dtype=torch.float32).reshape(3, 7)/30)

    def encode(self, values, subset=None):
        return {"valid": torch.ones(len(values["x"]), dtype=torch.bool),
                "observation": SimpleNamespace(aff=values["x"])}

    def decode_affect(self, affect):
        logits = affect @ self.projection
        return {"emotion_logits": logits, "intensity": logits[:, 0], "vad": logits[:, :3]}


def supervision_batch():
    x = torch.tensor([[1., .2, -.3], [.1, -.5, .7], [-.4, .3, .1]])
    features = {"x": x, "domain_id": torch.tensor([0, 0, 1])}
    labels = {"emotion": torch.tensor([0, 1, -1]), "endpoint_mask": torch.ones(3, dtype=torch.bool),
              "intensity": torch.tensor([.2, -.2, .4]), "intensity_mask": torch.tensor([True, False, True]),
              "vad": torch.zeros(3, 3), "vad_mask": torch.tensor([[True, False, False], [True, True, True], [False, False, True]])}
    priors = {"domains": {"0": {"class_weights": [2., .5, 0., 0., 0., 0., 0.]}}}
    return features, labels, priors


def test_supervision_uses_valid_scalar_counts_and_train_class_weights():
    observer = TinySupervision()
    values, labels, prior = supervision_batch()
    loss = globally_weighted_supervision(observer, values, labels, training_statistics=prior)
    logits = observer.decode_affect(values["x"])["emotion_logits"]
    expected = (F.cross_entropy(logits[:2], labels["emotion"][:2], reduction="none")*torch.tensor([2., .5])).sum()/2.5
    torch.testing.assert_close(loss["emotion"], expected)
    assert loss["emotion_weight_count"] == 2.5
    assert loss["intensity_weight_count"] == 2
    assert loss["vad_weight_count"] == 5


def test_weighted_class_denominator_below_one_is_not_clamped_to_one():
    observer = TinySupervision()
    values, labels, prior = supervision_batch()
    values = {key: value[1:2] for key, value in values.items()}
    labels = {key: value[1:2] for key, value in labels.items()}
    loss = globally_weighted_supervision(observer, values, labels, training_statistics=prior)
    expected = F.cross_entropy(observer.decode_affect(values["x"])["emotion_logits"], labels["emotion"])
    assert loss["emotion_weight_count"] == .5
    torch.testing.assert_close(loss["emotion"], expected)


def _ddp_worker(rank, rendezvous, output):
    import torch.distributed as distributed
    torch.set_num_threads(1)
    distributed.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    observer = TinySupervision()
    values, labels, prior = supervision_batch()
    # Rank one has no endpoints at all. It must still enter all three reductions.
    if rank == 1:
        values = labels = None
    objective = globally_weighted_supervision(observer, values, labels, training_statistics=prior)
    objective["total"].backward()
    distributed.all_reduce(observer.projection.grad)
    observer.projection.grad /= 2
    if rank == 0:
        torch.save(observer.projection.grad, output)
    distributed.destroy_process_group()


@pytest.mark.skipif(not torch.distributed.is_available(), reason="No distributed backend")
def test_ddp_empty_rank_matches_single_global_valid_objective(tmp_path):
    import torch.multiprocessing as multiprocessing
    rendezvous = (tmp_path/"rendezvous").resolve().as_uri()
    output = str(tmp_path/"gradient.pt")
    multiprocessing.spawn(_ddp_worker, args=(rendezvous, output), nprocs=2, join=True)
    observer = TinySupervision()
    values, labels, priors = supervision_batch()
    globally_weighted_supervision(observer, values, labels, training_statistics=priors)["total"].backward()
    torch.testing.assert_close(torch.load(output, weights_only=True), observer.projection.grad, atol=1e-7, rtol=1e-6)


def test_clone_diagnostics_do_not_confuse_dropout_with_data_variance_or_mutate_rng():
    config = model().construction(); config["dropout"] = .5
    observer = TokenObserver(config).eval()
    batch = collate_role_features([features(1)])
    rng = torch.get_rng_state().clone()
    diagnostic = clone_dropout_diagnostics(observer, batch)
    assert diagnostic["identical_input_clean_std"] < 1e-7
    assert diagnostic["identical_input_dropout_std"] > 1e-4
    assert not observer.training
    assert torch.equal(rng, torch.get_rng_state())


def good_metrics():
    matrix = torch.zeros(7, 7, dtype=torch.int64)
    matrix[0, 0] = 4; matrix[1, 1] = 4
    baseline = torch.zeros_like(matrix); baseline[0, 0] = 4; baseline[1, 0] = 4
    result = {**confusion_metrics(matrix), "emotion_samples": 8, "samples": 8, "missing": False,
              "constant_class": confusion_metrics(baseline), "embedding_variance_trace": .4,
              "vad_elements": 8, "vad_mse": .1, "vad_constant_train_mean_mse": .5,
              "modality_counts": {"A": 8, "V": 0, "T": 8}}
    return {"training_statistics": {"split": "train", "domains": {"1": {"class_counts": [5, 3, 0, 0, 0, 0, 0], "vad_count": [8, 0, 0]}}},
            "evaluation_model": "student", "evaluation_split": "val", "selection_loss": 0.,
            **{f"domain1/{subset}": copy.deepcopy(result) for subset in ("A", "AT", "AVT")}}


def test_semantic_gate_accepts_label_gain_and_records_missing_visual():
    report = semantic_gate(good_metrics())
    assert report["passed"] and report["allowed"]
    assert any("V missing" in item for item in report["limitations"])


@pytest.mark.parametrize("failure", ["constant", "variance", "vad"])
def test_semantic_gate_rejects_trivial_models_and_requires_explicit_smoke_bypass(failure):
    metrics = good_metrics()
    result = metrics["domain1/A"]
    if failure == "constant":
        result.update(macro_f1=result["constant_class"]["macro_f1"], uar=result["constant_class"]["uar"], supported_predicted_classes=1)
    elif failure == "variance":
        result["embedding_variance_trace"] = 0.
    else:
        result["vad_mse"] = result["vad_constant_train_mean_mse"]
    assert not semantic_gate(metrics)["allowed"]
    bypass = semantic_gate(metrics, allow_untrained=True)
    assert bypass["allowed"] and bypass["bypassed"] and not bypass["passed"]


def test_gate_reads_best_weights_and_metrics_not_last_validation(tmp_path):
    from emotion_ssm.config_v3 import default_config
    from scripts.validate_a0_v31 import validate_export
    config = default_config(); metrics = good_metrics()
    payload = {"format_version": config["format_version"], "protocol": config["protocol"],
               "kind": "observation_v3", "config": config, "construction": {"exported_model": "student"},
               "models": {"observer": {"weight": torch.ones(1)}, "teacher": {"weight": torch.ones(1)}},
               "metrics": {**metrics, "student": metrics, "selected_model": "student"}, "global_step": 20}
    torch.save(payload, tmp_path/"best.pt")
    (tmp_path/"validation.json").write_text(json.dumps({"selection_loss": 100, "semantic_gate": {"passed": False}}))
    report = validate_export(tmp_path)
    assert report["passed"] and report["checkpoint_step"] == 20
    payload["models"]["teacher"]["weight"].zero_()
    torch.save(payload, tmp_path/"best.pt")
    with pytest.raises(ValueError, match="weights differ"):
        validate_export(tmp_path, allow_untrained=True)


def test_calibration_prosody_follows_labelled_audio_source():
    observer = model(); source = fake_source("english")
    initialize_dualtalk_adapters(observer, labelled_provenance({"iemocap": source}), source)
    for key, value in observer.adapters["prosody"][1].state_dict().items():
        torch.testing.assert_close(value, observer.adapters["prosody"][2].state_dict()[key], atol=0, rtol=0)


def test_validation_batch_size_does_not_change_weighted_label_metrics(tmp_path):
    root = tmp_path/"cache"; write_cache(root)
    observer = model()
    first = evaluate(observer, [root], "cpu", batch_size=1)
    second = evaluate(observer, [root], "cpu", batch_size=3)
    for subset in ("A", "AT", "AVT"):
        a, b = first["domain0/"+subset], second["domain0/"+subset]
        assert a["confusion"] == b["confusion"]
        for key in ("vad_mse", "vad_constant_train_mean_mse", "embedding_variance_trace"):
            assert a[key] == pytest.approx(b[key], abs=1e-6)


def _calibration_fixture(case=0):
    values = [features(index + 10) for index in range(4)]
    values[1]["action_present"] = torch.tensor(False)
    if case >= 1:
        values[0]["flame_mask"].zero_()
    if case >= 2:
        for value in values:
            value["flame_mask"].zero_()
    batch = collate_role_features(values)
    # Unequal real token counts distinguish global element means from averaging
    # the two local means. The first rank gets one row, the second three rows.
    batch["calibration_chosen"] = torch.arange(5)[None] < torch.arange(1, 5)[:, None]
    return batch


def _fixed_calibration_corruption(values, *args, **kwargs):
    return {"flame": values["calibration_chosen"]}


def _calibration_models():
    observer = model().eval()
    observer.sample_corruption = _fixed_calibration_corruption
    teacher = copy.deepcopy(observer).requires_grad_(False).eval()
    with torch.no_grad():
        teacher.affect_head[-1].bias.add_(torch.linspace(-.3, .3, 8))
        teacher.action_head[-1].bias.add_(.2)
    observer.requires_grad_(False)
    observer.adapters["flame"].requires_grad_(True)
    observer.reconstruct["flame"].requires_grad_(True)
    return observer, teacher


def _calibration_gradient(observer):
    return torch.cat([(parameter.grad if parameter.grad is not None else torch.zeros_like(parameter)).flatten()
                      for parameter in observer.parameters() if parameter.requires_grad])


def test_calibration_counts_only_actual_rows_and_masked_flame_elements():
    observer, teacher = _calibration_models()
    def forbidden(*args, **kwargs):
        raise AssertionError("Calibration must not build unused self-supervision/representation graphs")
    observer.masked_loss = forbidden
    result = calibration_objective(observer, {"features": _calibration_fixture()}, teacher)
    assert result["coordinate_weight_count"] == 4
    assert result["action_weight_count"] == 3*8
    assert result["reconstruction_weight_count"] == (1+2+3+4)*56
    result["total"].backward()
    assert _calibration_gradient(observer).abs().sum() > 0
    assert all(parameter.grad is None for parameter in teacher.parameters())


def _calibration_ddp_worker(rank, rendezvous, output):
    from datetime import timedelta
    import torch.distributed as distributed
    from torch.nn.parallel import DistributedDataParallel
    from emotion_ssm.train.observation_v3 import _TrainingObjective
    torch.set_num_threads(1)
    distributed.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2,
                                   timeout=timedelta(seconds=30))
    observer, teacher = _calibration_models()
    wrapped = DistributedDataParallel(_TrainingObjective(observer, teacher, {"stage": "calibration"}),
                                      find_unused_parameters=True, broadcast_buffers=False)
    results = []
    for case in range(3):
        observer.zero_grad(set_to_none=True)
        all_features = _calibration_fixture(case)
        section = slice(0, 1) if rank == 0 else slice(1, 4)
        batch = {name: value[section] for name, value in all_features.items()}
        loss = wrapped({"features": batch}, "V")
        values = torch.stack([loss[name].detach() for name in ("coordinate", "action_teacher", "masked_reconstruction", "total")])
        distributed.all_reduce(values); values /= 2
        loss["total"].backward()
        results.append({"loss": values, "gradient": _calibration_gradient(observer),
                        "counts": torch.stack([loss[name] for name in ("coordinate_weight_count", "action_weight_count", "reconstruction_weight_count")])})
    torch.save(results, output + str(rank) + ".pt")
    distributed.destroy_process_group()


@pytest.mark.skipif(not torch.distributed.is_available() or not torch.distributed.is_gloo_available(), reason="Gloo unavailable")
def test_calibration_ddp_unequal_and_empty_flame_ranks_match_global_batch(tmp_path):
    import torch.multiprocessing as multiprocessing
    rendezvous = (tmp_path / "calibration_rendezvous").resolve().as_uri()
    output = str(tmp_path / "calibration_rank")
    multiprocessing.spawn(_calibration_ddp_worker, args=(rendezvous, output), nprocs=2, join=True)
    ranks = [torch.load(output + str(rank) + ".pt", weights_only=True) for rank in (0, 1)]
    observer, teacher = _calibration_models()
    for case in range(3):
        observer.zero_grad(set_to_none=True)
        reference = calibration_objective(observer, {"features": _calibration_fixture(case)}, teacher)
        reference["total"].backward()
        values = torch.stack([reference[name].detach() for name in ("coordinate", "action_teacher", "masked_reconstruction", "total")])
        counts = torch.stack([reference[name] for name in ("coordinate_weight_count", "action_weight_count", "reconstruction_weight_count")])
        for result in ranks:
            torch.testing.assert_close(result[case]["loss"], values, rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(result[case]["gradient"], _calibration_gradient(observer), rtol=1e-4, atol=2e-6)
            torch.testing.assert_close(result[case]["counts"], counts)
    assert ranks[0][1]["counts"].tolist() == [3., 16., 9*56.]
    assert ranks[0][2]["counts"].count_nonzero() == 0
    assert ranks[0][2]["loss"].count_nonzero() == 0
    assert ranks[0][2]["gradient"].count_nonzero() == 0
