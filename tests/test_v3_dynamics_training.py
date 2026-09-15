import copy
import json
from pathlib import Path

import pytest
import torch

from emotion_ssm.config_v3 import PROTOCOL, default_config
from emotion_ssm.data.packets_v3 import empty_role_features
from emotion_ssm.models.token_observer import TokenObserver
from emotion_ssm.models.state_core import UnifiedEmotionStateCore
from emotion_ssm.train import dynamics_v3
from emotion_ssm.utils.checkpoint_v3 import read_checkpoint, save_checkpoint


def make_training(tmp_path):
    torch.manual_seed(321)
    root = tmp_path / "tokens"
    root.mkdir()
    manifest = {"protocol": PROTOCOL, "feature_sources": {"test": "synthetic-local-token-v3"},
                "splits": {"train": ["train0", "train1"], "val": ["val0", "val1"], "test": []},
                "dialogues": {}}
    for index, name in enumerate(("train0", "train1", "val0", "val1")):
        packets = []
        for tick in range(5):
            roles = []
            for role in (0, 1):
                features = empty_role_features(audio_dim=4, text_dim=4, now=tick+1, dt=1)
                features["audio_tokens"] = torch.full((2, 4), .1 * (tick+1+index+role))
                features["audio_mask"] = torch.ones(2, dtype=torch.bool)
                features["audio_times"] = torch.tensor([tick+.25, tick+.75])
                features["audio_starts"] = torch.tensor([tick, tick+.5])
                features["audio_fresh_mask"] = features["audio_mask"].clone()
                features["prosody_tokens"] = torch.zeros(2, 8)
                features["prosody_mask"] = features["audio_mask"].clone()
                features["prosody_times"] = features["audio_times"].clone()
                features["fresh_observation"][0] = True
                features["modality_mask"][0] = True
                features["action_present"] = torch.tensor(True)
                features["action_duration"] = torch.tensor(1.)
                roles.append(features)
            labels = [[], []]
            if tick == 2:
                labels[0].append({"start": 0., "end": 3., "emotion": index % 7,
                                  "vad": [.2, 0., 0.], "vad_mask": [True, False, False],
                                  "intensity": .3, "intensity_mask": True})
            packets.append({"protocol": PROTOCOL, "dialogue_id": name, "start": float(tick),
                            "end": float(tick+1), "dt": 1., "roles": roles, "targets": labels})
        filename = name + ".pt"
        manifest["dialogues"][name] = {"path": filename, "cache_id": name, "packets": len(packets)}
        torch.save({"protocol": PROTOCOL, "cache_id": name, "packets": packets}, root / filename)
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    config = default_config()
    config["observer"].update(audio_dim=4, text_dim=4, model_dim=8, affect_dim=4,
                              num_layers=1, num_heads=2, dropout=.1)
    config["state"].update(relation_dim=3, hidden_dim=8, max_integration_step=.5)
    config["train"].update(device="cpu", amp=False, max_steps=2, dynamics_steps=2, global_chunks_per_step=2,
                           tbptt_seconds=2, validate_every=1, checkpoint_every=1, log_every=1,
                           validation_max_dialogues=0, observer_batch_size=4,
                           forecast_seconds=[1, 2], modality_dropout=True, observation_dropout=.1)
    config["data"]["token_roots"] = [str(root)]
    config["paths"]["output"] = str(tmp_path / "dynamics")
    source = tmp_path / "observation.pt"
    config["paths"]["observation_checkpoint"] = str(source)
    observer = TokenObserver(config["observer"])
    save_checkpoint(source, {"observer": observer, "teacher": copy.deepcopy(observer)}, config,
                    {"observer": observer.construction()}, "observation_v3")
    return config


@pytest.fixture(autouse=True)
def small_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def test_future_matching_uses_actual_timestamps_without_index_interpolation():
    times = [0.3, 1.3, 2., 2.3, 4.3]
    assert dynamics_v3.future_matches(times, 0, [1, 2, 4]) == [(1., 1), (2., 3), (4., 4)]
    assert dynamics_v3.future_matches(times, 1, [1, 2, 4]) == [(1., 3)]


def test_sentence_endpoint_inside_packet_cannot_read_later_audio(tmp_path):
    config = make_training(tmp_path)
    collection = dynamics_v3.DialogueCollection(config["data"]["token_roots"], "train")
    packet = copy.deepcopy(collection[0]["packets"][2])
    packet["targets"][0][0]["end"] = 2.5
    observer = TokenObserver(config["observer"]).eval()
    core = UnifiedEmotionStateCore(observation_dim=4, **config["state"])
    prior = core.initialize(1)
    def endpoint_loss(value):
        pair, _ = dynamics_v3.encode_pair(observer, value, "cpu")
        current = core.advance(prior, pair, 1.)
        return dynamics_v3.endpoint_objective(observer, core, prior, current, value, "cpu"), current
    first, state_before = endpoint_loss(packet)
    packet["roles"][0]["audio_tokens"][1] = torch.tensor([20., -13., 4., 2.])
    second, state_after = endpoint_loss(packet)
    torch.testing.assert_close(first, second, rtol=0, atol=1e-7)
    assert not torch.allclose(state_before.fast, state_after.fast)


def test_sentence_endpoint_clips_future_prosody_with_audio_and_keeps_float64_clock(tmp_path):
    config = make_training(tmp_path)
    collection = dynamics_v3.DialogueCollection(config["data"]["token_roots"], "train")
    packet = copy.deepcopy(collection[0]["packets"][2])
    endpoint=2.5000004
    packet["targets"][0][0]["end"]=endpoint
    observer=TokenObserver(config["observer"]).eval()
    core=UnifiedEmotionStateCore(observation_dim=4,**config["state"])
    prior=core.initialize(1)
    def loss(value):
        pair,_=dynamics_v3.encode_pair(observer,value,"cpu")
        current=core.advance(prior,pair,1.)
        return dynamics_v3.endpoint_objective(observer,core,prior,current,value,"cpu"),current
    before,state_before=loss(packet)
    packet["roles"][0]["prosody_tokens"][1]=torch.tensor([80.,-40.,32.,-72.,18.,9.,-21.,11.])
    after,state_after=loss(packet)
    torch.testing.assert_close(before,after,rtol=0,atol=1e-7)
    assert not torch.allclose(state_before.fast,state_after.fast)
    partial=dynamics_v3.packet_prefix(packet,endpoint)
    first=partial["roles"][0]
    assert first["audio_mask"].tolist()==[True,False]
    assert first["prosody_mask"].tolist()==[True,False]
    assert first["now"].dtype==torch.float64
    assert float(first["now"])==endpoint
    assert packet["roles"][0]["prosody_mask"].all()


def test_real_small_dynamics_training_checkpoint_and_endpoint_counts(tmp_path):
    config = make_training(tmp_path)
    initial = read_checkpoint(config["paths"]["observation_checkpoint"])
    result = dynamics_v3.run(config)
    assert result["step"] == 2
    path = Path(config["paths"]["output"])
    assert (path / "best.pt").is_file()
    assert json.loads((path / "training_status.json").read_text())["status"] == "complete"
    checkpoint = read_checkpoint(path / "last.pt")
    assert set(checkpoint["models"]) == {"observer", "teacher", "state"}
    # Heads stay in the fixed coordinate while other upstream paths learn.
    torch.testing.assert_close(checkpoint["models"]["observer"]["affect_head.1.weight"],
                               initial["models"]["observer"]["affect_head.1.weight"], rtol=0, atol=0)
    assert not torch.equal(checkpoint["models"]["observer"]["action_head.1.weight"],
                           initial["models"]["observer"]["action_head.1.weight"])
    validation = result["validation"]
    # There are exactly two sentence labels in two validation dialogues.
    assert validation["current_endpoint/emotion_ce_count"] == 2
    assert validation["current_endpoint/vad_mse_count"] == 2
    for method in ("learned_open_loop", "last_state", "pure_decay", "training_mean"):
        assert validation[method + "/affect_mse_count"] == 112


def test_resume_restores_packet_memory_cursor_rng_and_optimizer_exactly(tmp_path, monkeypatch):
    config = make_training(tmp_path)
    snapshot = tmp_path / "step1.pt"
    original = dynamics_v3.save_checkpoint
    def saving(path, *args, **kwargs):
        payload = original(path, *args, **kwargs)
        if payload["global_step"] == 1 and Path(path).name == "last.pt":
            torch.save(payload, snapshot)
        return payload
    monkeypatch.setattr(dynamics_v3, "save_checkpoint", saving)
    dynamics_v3.run(config)
    uninterrupted = read_checkpoint(Path(config["paths"]["output"]) / "last.pt")
    config["paths"]["resume"] = str(snapshot)
    dynamics_v3.run(config)
    resumed = read_checkpoint(Path(config["paths"]["output"]) / "last.pt")
    for model in uninterrupted["models"]:
        for name in uninterrupted["models"][model]:
            torch.testing.assert_close(uninterrupted["models"][model][name], resumed["models"][model][name],
                                       atol=0, rtol=0)
    assert uninterrupted["run_state"]["ranks"][0]["cursor"]["packet_index"] == resumed["run_state"]["ranks"][0]["cursor"]["packet_index"]


def test_validation_partition_and_batch_size_produce_same_weighted_metrics(tmp_path):
    config = make_training(tmp_path)
    observer = TokenObserver(config["observer"]).eval()
    teacher = copy.deepcopy(observer).eval()
    core = UnifiedEmotionStateCore(observation_dim=4, **config["state"])
    collection = dynamics_v3.DialogueCollection(config["data"]["token_roots"], "val")
    device = torch.device("cpu")
    def run(rank, world, batch):
        cache = dynamics_v3.TeacherTargets(teacher, device, batch_size=batch)
        return dynamics_v3.validate(observer, teacher, core, collection, cache, config, device,
                                    torch.zeros(4), rank=rank, world=world)
    complete = run(0, 1, 4)
    left, right = run(0, 2, 1), run(1, 2, 7)
    for key in ("learned_open_loop/affect_mse", "current_endpoint/emotion_ce", "pure_decay/affect_mse"):
        count = left[key+"_count"] + right[key+"_count"]
        combined = (left[key]*left[key+"_count"] + right[key]*right[key+"_count"]) / count
        assert count == complete[key+"_count"]
        assert combined == pytest.approx(complete[key], rel=2e-6, abs=2e-6)


@pytest.mark.parametrize("until", [None, 4])
def test_resume_configuration_changes_only_explicit_budget_and_output(tmp_path, until):
    config = make_training(tmp_path)
    payload = {"config": config, "global_step": 2}
    expected = copy.deepcopy(config)
    output = tmp_path / "extension"
    if until is not None:
        expected["train"].update(max_steps=until, dynamics_steps=until)
    expected["paths"]["output"] = str(output.resolve())
    actual = dynamics_v3._resume_configuration(payload, until, output)
    assert actual == expected
    assert config["train"]["max_steps"] == 2
    assert config["paths"]["output"] != str(output)


@pytest.mark.parametrize("until", [0, 1, 2, -1, 3.5, True])
def test_resume_extension_rejects_non_extension_or_non_integer(tmp_path, until):
    config = make_training(tmp_path)
    with pytest.raises(ValueError):
        dynamics_v3._resume_configuration({"config": config, "global_step": 2}, until)


def test_resume_extension_requires_checkpoint(tmp_path):
    config = make_training(tmp_path)
    with pytest.raises(ValueError, match="Resume overrides require"):
        dynamics_v3.run(config, resume_until_step=4)


def test_completed_run_extension_matches_uninterrupted_training(tmp_path):
    import hashlib
    import numpy as np

    def equal(left, right):
        if torch.is_tensor(left):
            torch.testing.assert_close(left, right, atol=0, rtol=0)
        elif isinstance(left, np.ndarray):
            np.testing.assert_array_equal(left, right)
        elif isinstance(left, dict):
            assert left.keys() == right.keys()
            for key in left:
                equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            assert len(left) == len(right)
            for a, b in zip(left, right):
                equal(a, b)
        else:
            assert left == right

    config = make_training(tmp_path)
    full = copy.deepcopy(config)
    full["train"].update(max_steps=4, dynamics_steps=4)
    full["paths"]["output"] = str(tmp_path / "full")
    dynamics_v3.run(full)
    uninterrupted = read_checkpoint(Path(full["paths"]["output"]) / "last.pt")
    dynamics_v3.run(config)
    source = Path(config["paths"]["output"]) / "last.pt"
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    resumed_input = copy.deepcopy(config)
    resumed_input["paths"]["resume"] = str(source)
    # Arbitrary config changes must not leak into a resumed experiment.
    resumed_input["train"].update(max_steps=99, dynamics_steps=99, lr=.3, observation_dropout=.7)
    destination = tmp_path / "extended"
    dynamics_v3.run(resumed_input, resume_until_step=4, resume_output=destination)
    extended = read_checkpoint(destination / "last.pt")
    assert extended["global_step"] == 4
    assert extended["config"]["train"] == uninterrupted["config"]["train"]
    assert extended["experiment"]["max_steps"] == 4
    for key in ("models", "optimizer", "run_state", "rng_state", "metrics"):
        equal(uninterrupted[key], extended[key])
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before


@pytest.mark.parametrize("rate", [0., -1e-4, float("nan"), float("inf"), True])
def test_resume_lr_rejects_invalid_values(tmp_path, rate):
    config = make_training(tmp_path)
    payload = {"config": config, "global_step": 2, "optimizer": {"param_groups": [{"lr": 1e-4}]}}
    with pytest.raises(ValueError, match="finite positive"):
        dynamics_v3._resume_configuration(payload, 4, tmp_path / "branch", rate)


@pytest.mark.parametrize("same_output", [False, True])
def test_resume_lr_requires_distinct_output(tmp_path, same_output):
    config = make_training(tmp_path)
    payload = {"config": config, "global_step": 2, "optimizer": {"param_groups": [{"lr": 1e-4}]}}
    destination = config["paths"]["output"] if same_output else None
    with pytest.raises(ValueError, match="separate resume_output"):
        dynamics_v3._resume_configuration(payload, 4, destination, 3e-5)


def test_resume_lr_without_checkpoint_is_rejected(tmp_path):
    config = make_training(tmp_path)
    with pytest.raises(ValueError, match="Resume overrides require"):
        dynamics_v3.run(config, resume_lr=3e-5, resume_output=tmp_path / "branch")


def test_lr_branch_preserves_moments_applies_after_load_and_survives_resume(tmp_path):
    import hashlib
    import numpy as np

    def equal(left, right):
        if torch.is_tensor(left):
            torch.testing.assert_close(left, right, atol=0, rtol=0)
        elif isinstance(left, np.ndarray):
            np.testing.assert_array_equal(left, right)
        elif isinstance(left, dict):
            assert left.keys() == right.keys()
            for key in left:
                equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            assert len(left) == len(right)
            for a, b in zip(left, right):
                equal(a, b)
        else:
            assert left == right

    config = make_training(tmp_path)
    dynamics_v3.run(config)
    source = Path(config["paths"]["output"]) / "last.pt"
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    original = read_checkpoint(source)
    # Independent reference: adjust only the serialized group rates/config;
    # leave Adam moments, data cursor, recurrent memory and RNG untouched.
    manual = copy.deepcopy(original)
    manual["config"]["train"].update(lr=3e-5, state_lr=3e-5)
    for group in manual["optimizer"]["param_groups"]:
        group["lr"] = 3e-5
    manual_path = tmp_path / "manual_lr.pt"
    torch.save(manual, manual_path)
    request = copy.deepcopy(config)
    request["paths"]["resume"] = str(manual_path)
    reference_dir = tmp_path / "reference_lr"
    dynamics_v3.run(request, resume_until_step=3, resume_output=reference_dir)
    request["paths"]["resume"] = str(source)
    branch_dir = tmp_path / "branch_lr"
    dynamics_v3.run(request, resume_until_step=3, resume_output=branch_dir, resume_lr=3e-5)
    reference = read_checkpoint(reference_dir / "last.pt")
    branch = read_checkpoint(branch_dir / "last.pt")
    for key in ("models", "optimizer", "run_state", "rng_state", "metrics"):
        equal(reference[key], branch[key])
    assert [g["lr"] for g in branch["optimizer"]["param_groups"]] == [3e-5, 3e-5]
    assert branch["config"]["train"]["state_lr"] == branch["config"]["train"]["lr"] == 3e-5
    assert branch["config"]["learning_rate_branch"]["source_step"] == 2
    receipt = json.loads((branch_dir / "resume_receipt.json").read_text())
    assert receipt["source_learning_rates"] == [1e-4, 1e-4]
    assert receipt["actual_learning_rates"] == [3e-5, 3e-5]
    assert receipt["optimizer_step_range"] == [2, 2]
    assert receipt["cursor"]["packet_index"] == original["run_state"]["ranks"][0]["cursor"]["packet_index"]
    # A later normal resume must retain the forked LR without passing it again.
    request["paths"]["resume"] = str(branch_dir / "last.pt")
    dynamics_v3.run(request, resume_until_step=4)
    final = read_checkpoint(branch_dir / "last.pt")
    assert final["global_step"] == 4
    assert [g["lr"] for g in final["optimizer"]["param_groups"]] == [3e-5, 3e-5]
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash
