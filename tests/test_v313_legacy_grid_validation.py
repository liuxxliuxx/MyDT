"""Keep archived v3.1.2 readouts and teacher-grid metrics stable after v3.1.3."""
import copy

import pytest
import torch
from torch.nn import functional as F

from emotion_ssm.config_v3 import FUTURE_LABEL_PROTOCOL, LEGACY_FUTURE_LABEL_PROTOCOL
from emotion_ssm.models.state_core import UnifiedEmotionStateCore
from emotion_ssm.models.token_observer import TokenObserver
from emotion_ssm.train import dynamics_v3 as dynamics
from test_v3_dynamics_training import make_training


V312 = "v3.1.2-state-memory-forecast"


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def validation_case(tmp_path):
    config = make_training(tmp_path)
    # This fixture exercises archived linear-flow evaluation, not the new
    # default architecture. Keep the historical construction explicit.
    config["state"] = {"relation_dim": 3, "hidden_dim": 8, "max_integration_step": 1.}
    observer = TokenObserver(config["observer"]).eval()
    # A large, known baseline makes raw and unit readouts distinguishable.
    core = UnifiedEmotionStateCore(observation_dim=4, fast_correction_rate=.05, **config["state"])
    with torch.no_grad():
        core.baseline.copy_(torch.tensor([3., 1., .5, -.25]))
        observer.emotion_head.weight.zero_()
        observer.emotion_head.bias.zero_()
        observer.emotion_head.weight[0, 0] = 1.
        observer.emotion_head.weight[2, 0] = -.5
        observer.vad_head.weight.zero_()
        observer.vad_head.bias.zero_()
        observer.vad_head.weight[0, 0] = .5
    teacher = copy.deepcopy(observer).requires_grad_(False).eval()
    collection = dynamics.DialogueCollection(config["data"]["token_roots"], "val")
    mean = torch.tensor([.1, -.2, .3, -.4])
    return config, observer, teacher, core, collection, mean


def validate_case(case, config):
    _, observer, teacher, core, collection, mean = case
    cache = dynamics.TeacherTargets(teacher, torch.device("cpu"), batch_size=3)
    return dynamics.validate(observer, teacher, core, collection, cache, config,
                             torch.device("cpu"), mean)


@torch.no_grad()
def reference_grid(case, normalize_labels):
    """Compute counts/SSE and endpoint head losses without the validation helpers.

    Real observer/core modules supply the trajectory. Metric arithmetic and
    unit-coordinate decoding are independent of forecast_objective/label_loss.
    """
    config, observer, teacher, core, collection, mean = case
    observer.eval(), core.eval()
    totals, counts, unit_grid_sse = {}, {}, 0.
    cache = dynamics.TeacherTargets(teacher, torch.device("cpu"), batch_size=3)

    def add(name, total, count):
        totals[name] = totals.get(name, 0.) + float(total)
        counts[name] = counts.get(name, 0) + count

    for index in range(len(collection)):
        dialogue = collection[index]
        targets = cache.get(collection.identity(index), dialogue)
        memory = core.initialize(1)
        states = []
        for tick, packet in enumerate(dialogue["packets"]):
            pair, _ = dynamics.encode_pair(observer, packet, "cpu", event_id=tick)
            memory = core.advance(memory, pair, float(packet["dt"]))
            states.append(memory)
            affect = core.affect(memory)
            decoded = observer.decode_affect(F.normalize(affect, dim=-1) if normalize_labels else affect)
            for role, labels in enumerate(packet["targets"]):
                for label in labels:
                    # The reference fixture's current labels are exact grid ends.
                    assert label["end"] == packet["end"]
                    logits = decoded["emotion_logits"][:, role]
                    add("current_endpoint/emotion_ce", F.cross_entropy(logits, torch.tensor([label["emotion"]])), 1)
                    mask = torch.tensor(label["vad_mask"], dtype=torch.bool)
                    difference = decoded["vad"][:, role, mask] - torch.tensor(label["vad"])[mask][None]
                    add("current_endpoint/vad_mse", difference.square().sum(), int(mask.sum()))
        for origin, state in enumerate(states):
            for query in range(origin+1, len(states)):
                elapsed = targets["times"][query] - targets["times"][origin]
                if elapsed not in config["train"]["forecast_seconds"]:
                    continue
                present = targets["valid"][query:query+1]
                target = targets["affect"][query:query+1][present]
                prediction = core.affect(core.forecast(state, [elapsed])[0])
                estimates = {"learned_open_loop": prediction,
                             "last_state": core.affect(state),
                             "pure_decay": core.affect(core.decay_only(state, elapsed)),
                             "training_mean": mean[None,None].expand_as(prediction)}
                for name, value in estimates.items():
                    sse = (value[present].float() - target.float()).square().sum()
                    add(name+"/affect_mse", sse, target.numel())
                    add(name+f"/h{elapsed:g}/affect_mse", sse, target.numel())
                    decoded = observer.decode_affect(F.normalize(value, dim=-1) if normalize_labels else value)
                    for role, labels in enumerate(dialogue["packets"][query]["targets"]):
                        for label in labels:
                            logits = decoded["emotion_logits"][:, role]
                            add(name+"/endpoint/emotion_ce", F.cross_entropy(logits, torch.tensor([label["emotion"]])), 1)
                unit_grid_sse += float((F.normalize(prediction, dim=-1)[present] - target).square().sum())
    result = {name: value/counts[name] for name, value in totals.items()}
    result.update({name+"_count": count for name, count in counts.items()})
    return result, unit_grid_sse/counts["learned_open_loop/affect_mse"]


@pytest.mark.parametrize("revision,normalized", [(None, False), (V312, True)])
def test_archived_validation_decodes_labels_using_its_original_coordinate_protocol(validation_case, revision, normalized):
    config = copy.deepcopy(validation_case[0])
    config["train"]["future_label_protocol"] = LEGACY_FUTURE_LABEL_PROTOCOL
    if revision is None:
        config["train"].pop("dynamics_revision")
    else:
        config["train"]["dynamics_revision"] = revision
    expected, _ = reference_grid(validation_case, normalized)
    wrong, _ = reference_grid(validation_case, not normalized)
    actual = validate_case(validation_case, config)
    for key in ("current_endpoint/emotion_ce", "current_endpoint/vad_mse"):
        assert abs(expected[key]-wrong[key]) > .01
        assert actual[key] == pytest.approx(expected[key], rel=2e-6, abs=2e-6)
        assert actual[key+"_count"] == expected[key+"_count"] == 2
    assert actual["label_readout"] == ("unit_affect_zero_for_undefined_direction" if normalized else "legacy_raw_state")


def test_v312_teacher_grid_is_raw_mse_with_original_horizon_counts(validation_case):
    config = copy.deepcopy(validation_case[0])
    config["train"]["dynamics_revision"] = V312
    config["train"].pop("future_label_protocol")  # Actual archived configs lack this new field.
    expected, normalized_mse = reference_grid(validation_case, True)
    actual = validate_case(validation_case, config)
    grid = {key: value for key, value in expected.items() if
            "/affect_mse" in key or "/endpoint/emotion_ce" in key}
    for key, value in grid.items():
        assert actual[key] == pytest.approx(value, rel=2e-6, abs=2e-6)
    assert actual["learned_open_loop/affect_mse_count"] == 112
    assert abs(actual["learned_open_loop/affect_mse"] - normalized_mse) > .1
    assert actual["validation_loss"] == actual["learned_open_loop/affect_mse"]


def test_new_protocol_keeps_teacher_grid_metrics_identical_to_v312(validation_case):
    old_config = copy.deepcopy(validation_case[0])
    old_config["train"]["dynamics_revision"] = V312
    old_config["train"].pop("future_label_protocol")
    old = validate_case(validation_case, old_config)
    new = validate_case(validation_case, validation_case[0])
    keys = [key for key in old if "/affect_mse" in key and key.split("/")[0] in
            ("learned_open_loop", "last_state", "pure_decay", "training_mean")]
    assert len(keys) == 24  # Four methods, overall/h1/h2, values plus valid-element counts.
    for key in keys:
        assert new[key] == old[key]
    for method in ("learned_open_loop", "last_state", "pure_decay", "training_mean"):
        assert new[method+"/endpoint/emotion_ce"] == old[method+"/endpoint/emotion_ce"]
        assert new[method+"/endpoint/emotion_ce_count"] == old[method+"/endpoint/emotion_ce_count"] == 4
    assert "future_endpoint/emotion_ce" not in old
    assert new["future_endpoint/emotion_ce_count"] == 4
    assert new["future_endpoint_unique/emotion_ce_count"] == 2
    assert old["future_endpoint_metrics_protocol"] is None
    assert new["future_endpoint_metrics_protocol"] == FUTURE_LABEL_PROTOCOL
    assert new["grid_label_metrics_protocol"] == old["grid_label_metrics_protocol"] == LEGACY_FUTURE_LABEL_PROTOCOL


def test_interior_gold_uses_actual_seconds_and_separate_metrics_without_relabeling_grid(validation_case):
    config, observer, teacher, core, original, mean = validation_case

    class InteriorEndpoints:
        def __len__(self):
            return len(original)

        def balanced_indices(self, maximum=0):
            return original.balanced_indices(maximum)

        def identity(self, index):
            return original.identity(index) + ":interior"

        def __getitem__(self, index):
            value = copy.deepcopy(original[index])
            for packet in value["packets"]:
                for labels in packet["targets"]:
                    for label in labels:
                        label["end"] = 2.5
            return value

    collection = InteriorEndpoints()
    case = (config, observer, teacher, core, collection, mean)
    archived = copy.deepcopy(config)
    archived["train"].update(dynamics_revision=V312, future_label_protocol=LEGACY_FUTURE_LABEL_PROTOCOL)
    old, new = validate_case(case, archived), validate_case(case, config)
    # e=2.5 and h=1 select the complete origin t=1, so actual forecast time is
    # 1.5s. h=2 has no eligible completed origin; neither is rounded to t=3.
    losses = []
    observer.eval(), core.eval()
    with torch.no_grad():
        for index in range(len(collection)):
            dialogue = collection[index]
            pair, _ = dynamics.encode_pair(observer, dialogue["packets"][0], "cpu", event_id=0)
            state = core.advance(core.initialize(1), pair, 1.)
            future = core.affect(core.forecast(state, [1.5])[0])
            logits = observer.decode_affect(F.normalize(future, dim=-1))["emotion_logits"][:, 0]
            label = dialogue["packets"][2]["targets"][0][0]["emotion"]
            losses.append(float(F.cross_entropy(logits, torch.tensor([label]))))
    assert new["future_endpoint/emotion_ce"] == pytest.approx(sum(losses)/2, rel=2e-6, abs=2e-6)
    assert new["future_endpoint/emotion_ce_count"] == new["future_endpoint_unique/emotion_ce_count"] == 2
    assert new["future_endpoint/actual_seconds"] == 1.5
    assert new["future_endpoint/actual_seconds_count"] == 2
    assert new["future_endpoint_coverage/emotion_endpoints"] == 1.
    assert "future_endpoint/h2/emotion_ce" not in new
    assert "future_endpoint/emotion_ce" not in old
    assert old["current_endpoint/emotion_ce_count"] == new["current_endpoint/emotion_ce_count"] == 2
    for method in ("learned_open_loop", "last_state", "pure_decay", "training_mean"):
        assert method+"/endpoint/emotion_ce" not in old
        assert method+"/endpoint/emotion_ce" not in new
        assert old[method+"/affect_mse"] == new[method+"/affect_mse"]
        assert old[method+"/affect_mse_count"] == new[method+"/affect_mse_count"] == 112
