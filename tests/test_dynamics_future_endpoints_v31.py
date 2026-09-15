"""Actual-time future gold supervision, causal inputs, and validation coverage."""
import copy
import json

import pytest
import torch
from torch.nn import functional as F

from emotion_ssm.config_v3 import FUTURE_LABEL_PROTOCOL, PROTOCOL, default_config
from emotion_ssm.data.packets_v3 import empty_role_features
from emotion_ssm.models.state_core import UnifiedEmotionStateCore
from emotion_ssm.models.token_observer import TokenObserver
from emotion_ssm.train import dynamics_v3 as dynamics
from emotion_ssm.utils.checkpoint_v3 import save_checkpoint


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def gold(end, emotion=0, identity="utterance", **overrides):
    label = {"start": max(0., end - .4), "end": end, "emotion": emotion,
             "utterance_id": identity, "intensity_mask": False,
             "vad": [0., 0., 0.], "vad_mask": [False, False, False]}
    label.update(overrides)
    return label


def packets_at(times):
    result = []
    start = 0.
    for tick, end in enumerate(times):
        roles = []
        for role in (0, 1):
            features = empty_role_features(audio_dim=4, text_dim=4,
                                           now=end, dt=end - start)
            features["audio_tokens"] = torch.full((1, 4), .1 * (tick + role + 1))
            features["audio_mask"][:] = True
            features["audio_fresh_mask"][:] = True
            features["audio_times"][:] = end
            features["audio_starts"][:] = start
            features["prosody_mask"] = features["audio_mask"].clone()
            features["prosody_times"] = features["audio_times"].clone()
            features["fresh_observation"][0] = True
            features["modality_mask"][0] = True
            roles.append(features)
        result.append({"protocol": PROTOCOL, "start": start, "end": end, "dt": end - start,
                       "roles": roles, "targets": [[], []]})
        start = end
    return result


def small_models():
    torch.manual_seed(19)
    observer = TokenObserver(dict(audio_dim=4, text_dim=4, model_dim=8,
        affect_dim=4, num_layers=1, num_heads=2, summary_dim=2, dropout=0.)).eval()
    observer.requires_grad_(False)
    with torch.no_grad():
        observer.emotion_head.weight.zero_()
        observer.emotion_head.bias.zero_()
        observer.emotion_head.weight[0] = torch.tensor([2., -1., .5, 0.])
        observer.emotion_head.weight[1] = torch.tensor([-1., 2., 0., -.5])
    core = UnifiedEmotionStateCore(observation_dim=4, relation_dim=2,
                                   hidden_dim=8, max_integration_step=1.)
    return observer, core


def origin_memory(core):
    state = core.initialize(1)
    state.fast = torch.tensor([[[.45, -.2, .1, .25],
                                [-.15, .4, .2, -.1]]], requires_grad=True)
    state.slow = torch.tensor([[[.1, .2, -.1, .05],
                                [.15, -.1, .1, .2]]])
    return state


class MemoryCollection:
    def __init__(self, dialogues):
        self.dialogues = dialogues

    def __getitem__(self, index):
        return self.dialogues[index]

    def identity(self, index):
        return "synthetic:" + str(index)

    def balanced_indices(self, maximum=0):
        return list(range(len(self.dialogues)))[:maximum or None]


def planned_loss(observer, core, state, packets, horizons=(1.,), origin=0,
                 statistics=None):
    plan = dynamics.future_endpoint_queries([packet["end"] for packet in packets],
                                            packets, horizons)
    return dynamics.future_endpoint_objective(core, observer, state,
        plan.get(origin, []), statistics=statistics, normalize_labels=True)


def test_interior_future_label_changes_loss_and_gradient_at_the_origin():
    observer, core = small_models()
    packets = packets_at([1., 2., 3.])
    packets[2]["targets"][0] = [gold(2.5)]
    losses, gradients = [], []
    for emotion in (0, 1):
        packets[2]["targets"][0][0]["emotion"] = emotion
        state = origin_memory(core)
        loss = planned_loss(observer, core, state, packets)
        gradient = torch.autograd.grad(loss, state.fast)[0]
        assert torch.isfinite(loss) and torch.isfinite(gradient).all()
        assert loss > 0 and gradient.norm() > 0
        losses.append(loss.detach())
        gradients.append(gradient)
    assert not torch.allclose(losses[0], losses[1])
    assert not torch.allclose(gradients[0], gradients[1])
    assert all(parameter.grad is None for parameter in observer.parameters())


def test_irregular_endpoint_uses_latest_completed_origin_and_actual_seconds(monkeypatch):
    observer, core = small_models()
    packets = packets_at([.3, 1.3, 2., 2.3, 4.3])
    packets[4]["targets"][1] = [gold(3.75, emotion=1)]
    calls = []
    forecast = core.forecast

    def recording(state, seconds):
        calls.append(tuple(float(value) for value in seconds))
        return forecast(state, seconds)

    monkeypatch.setattr(core, "forecast", recording)
    state = origin_memory(core)
    losses = [planned_loss(observer, core, state, packets, (1., 2., 4.), origin=i)
              for i in range(len(packets))]
    assert [i for i, loss in enumerate(losses) if float(loss.detach()) > 0] == [1, 3]
    assert sorted(value for call in calls for value in call) == pytest.approx([1.45, 2.45])
    # The nominal 1/2 seconds must not replace actual 1.45/2.45 second forecasts.
    expected = dynamics.label_loss(observer,
        core.affect(forecast(state, [1.45])[0]), [[], [packets[4]["targets"][1][0]]],
        normalize=True)
    torch.testing.assert_close(losses[3], expected)


@pytest.mark.parametrize("field", ["audio_tokens", "text_tokens", "au_tokens", "flame_tokens"])
def test_future_modalities_cannot_change_gold_loss_or_origin_gradient(field, monkeypatch):
    observer, core = small_models()
    packets = packets_at([1., 2., 3.])
    packets[2]["targets"][1] = [gold(2.5, emotion=1)]

    def future_encoding_is_forbidden(*args, **kwargs):
        pytest.fail("Future gold objective attempted to encode future inputs")

    monkeypatch.setattr(observer, "encode", future_encoding_is_forbidden)
    monkeypatch.setattr(observer, "encode_clean", future_encoding_is_forbidden)
    monkeypatch.setattr(dynamics, "encode_pair", future_encoding_is_forbidden)
    state = origin_memory(core)
    first = planned_loss(observer, core, state, packets)
    first_gradient = torch.autograd.grad(first, state.fast)[0]
    changed = copy.deepcopy(packets)
    for packet in changed[1:]:
        for role in packet["roles"]:
            role[field] = torch.full_like(role[field], 12345.)
            role[field.removesuffix("_tokens") + "_mask"][:] = True
    second = planned_loss(observer, core, state, changed)
    second_gradient = torch.autograd.grad(second, state.fast)[0]
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    torch.testing.assert_close(first_gradient, second_gradient, rtol=0, atol=0)


def test_gold_is_supervised_when_future_teacher_has_no_valid_modality():
    observer, core = small_models()
    state = origin_memory(core)
    packets = packets_at([1., 2., 3.])
    packets[2]["targets"][0] = [gold(2.5)]
    for packet in packets:
        for features in packet["roles"]:
            for key, value in features.items():
                if torch.is_tensor(value) and value.dtype == torch.bool:
                    value.zero_()
    cache = dynamics.TeacherTargets(observer, "cpu", batch_size=2)
    targets = cache.get("missing-modalities", {"packets": packets}, horizons=[1.])
    assert not targets["valid"].any()
    assert targets["future_endpoint_coverage"]["covered"]["emotion_endpoints"] == 1
    coordinate = dynamics.forecast_objective(core, observer, state, targets,
        {"packets": packets}, 0, [1.], labels_weight=0., vector_loss=True)
    assert coordinate == 0
    labels = planned_loss(observer, core, state, packets)
    assert labels > 0
    assert torch.autograd.grad(coordinate + labels, state.fast)[0].norm() > 0


def test_shared_actual_query_is_forecast_once_with_separate_label_horizon_counts(monkeypatch):
    observer, core = small_models()
    state = origin_memory(core)
    packets = packets_at([1., 2., 5.])
    packets[2]["targets"] = [[gold(4.5, identity="first")],
                              [gold(4.5, emotion=1, identity="second")]]
    calls = []
    forecast = core.forecast

    def recording(memory, seconds):
        calls.append(list(seconds))
        return forecast(memory, seconds)

    monkeypatch.setattr(core, "forecast", recording)
    statistics = dynamics.SufficientStatistics()
    loss = planned_loss(observer, core, state, packets, (1., 2.), origin=1,
                        statistics=statistics)
    assert calls == [[2.5]]
    # Both nominal horizons select t=2 for e=4.5. Four labelled queries are
    # two independent endpoints, and the objective averages their four losses.
    affect = core.affect(forecast(state, [2.5])[0])
    expected = dynamics.label_loss(observer, affect, packets[2]["targets"], normalize=True) / 2
    torch.testing.assert_close(loss, expected)
    metrics = statistics.metrics()
    assert metrics["future_endpoint/emotion_ce_count"] == 4
    assert metrics["future_endpoint_unique/emotion_ce_count"] == 2
    assert metrics["future_endpoint/h1/emotion_ce_count"] == 2
    assert metrics["future_endpoint/h2/emotion_ce_count"] == 2
    assert metrics["future_endpoint/actual_seconds"] == 2.5


@pytest.mark.parametrize("times,horizons", [([1., 1.], [1.]), ([1., 2.], [1., 1.]),
    ([1., 2.], [2., 1.]), ([1., 2.], [0.]), ([1., 2.], [float("nan")])])
def test_invalid_or_duplicate_clocks_and_horizons_are_rejected(times, horizons):
    with pytest.raises(ValueError):
        dynamics.future_endpoint_queries(times, packets_at(times), horizons)


@pytest.mark.parametrize("endpoint", [.75, 2.5])
@pytest.mark.parametrize("second_emotion", [0, 1])
def test_duplicate_role_endpoint_is_rejected_even_when_uncovered(endpoint, second_emotion):
    packets = packets_at([1., 2., 3.])
    packet = packets[0 if endpoint < 1. else 2]
    packet["targets"][0] = [gold(endpoint, identity="first"),
                              gold(endpoint, emotion=second_emotion, identity="duplicate")]
    # Equal targets and conflicting targets must both fail explicitly. A second
    # utterance ID cannot inflate independent counts for the same role/time.
    with pytest.raises(ValueError, match="Duplicate valid gold endpoint"):
        dynamics.future_endpoint_queries([1., 2., 3.], packets, [1., 2.])


@pytest.mark.parametrize("early_label", [False, True])
def test_empty_or_too_early_endpoints_have_zero_loss_and_no_supervised_counts(early_label):
    observer, core = small_models()
    packets = packets_at([1., 2.])
    if early_label:
        packets[0]["targets"][0] = [gold(.75)]
    # An endpoint with no supervised fields is not an eligible gold target.
    packets[1]["targets"][1] = [gold(2., emotion=-1)]
    cache = dynamics.TeacherTargets(observer, "cpu", batch_size=2)
    targets = cache.get("empty", {"packets": packets}, horizons=[1., 2.])
    assert targets["future_endpoint_queries"] == {}
    coverage = targets["future_endpoint_coverage"]
    assert coverage["available"]["endpoints"] == int(early_label)
    assert coverage["covered"]["endpoints"] == 0
    statistics = dynamics.SufficientStatistics()
    state = origin_memory(core)
    loss = planned_loss(observer, core, state, packets, statistics=statistics)
    assert loss == 0
    assert torch.autograd.grad(loss, state.fast)[0].count_nonzero() == 0
    assert statistics.metrics().get("future_endpoint/emotion_ce_count", 0) == 0
    assert statistics.metrics()["future_endpoint/queries_per_origin"] == 0


def validation_dialogue(emotion):
    packets = packets_at([1., 2., 3.])
    packets[0]["targets"][0] = [gold(.5, identity="too-early")]
    packets[2]["targets"][0] = [gold(2.5, emotion=emotion, identity="interior",
        vad=[.2, -.4, .6], vad_mask=[True, False, True])]
    packets[2]["targets"][1] = [gold(3., emotion=1, identity="grid-aligned",
        vad=[-.1, .3, .2], vad_mask=[True, True, True],
        intensity=.3, intensity_mask=True)]
    return {"packets": packets}


def test_validation_matches_hand_selected_origins_and_weighted_rank_aggregation():
    observer, core = small_models()
    teacher = copy.deepcopy(observer)
    dialogues = [validation_dialogue(0), validation_dialogue(1)]
    collection = MemoryCollection(dialogues)
    config = default_config()
    config["train"].update(validation_max_dialogues=0, forecast_seconds=[1., 2.])

    def validate(rank=0, world=1, protocol=FUTURE_LABEL_PROTOCOL):
        local_config = copy.deepcopy(config)
        local_config["train"]["future_label_protocol"] = protocol
        cache = dynamics.TeacherTargets(teacher, "cpu", batch_size=3)
        return dynamics.validate(observer, teacher, core, collection, cache,
            local_config, "cpu", torch.zeros(4), rank=rank, world=world)

    full = validate()
    expected = dynamics.SufficientStatistics()
    for dialogue in dialogues:
        packets = dialogue["packets"]
        state = core.initialize(1)
        for tick, packet in enumerate(packets):
            pair, _ = dynamics.encode_pair(observer, packet, "cpu", event_id=tick)
            state = core.advance(state, pair, packet["dt"])
            # e=2.5,h=1 -> t=1,delta=1.5; e=3,h=2 -> t=1,delta=2;
            # e=3,h=1 -> t=2,delta=1. No origin exists for e=.5.
            queries = [(1.5, 0), (2., 1)] if tick == 0 else [(1., 1)] if tick == 1 else []
            for seconds, role in queries:
                label = packets[2]["targets"][role][0]
                labels = [[], []]
                labels[role] = [label]
                affect = core.affect(core.forecast(state, [seconds])[0])
                dynamics.label_loss(observer, affect, labels, expected, "manual", normalize=True)
    manual = expected.metrics()
    for metric in ("emotion_ce", "emotion_accuracy", "vad_mse", "intensity_mse"):
        assert full["future_endpoint/" + metric] == pytest.approx(manual["manual/" + metric], abs=2e-6)
        assert full["future_endpoint/" + metric + "_count"] == manual["manual/" + metric + "_count"]
    assert full["future_endpoint/emotion_ce_count"] == 6
    assert full["future_endpoint_unique/emotion_ce_count"] == 4
    assert full["future_endpoint/vad_mse_count"] == 16
    assert full["future_endpoint_unique/vad_mse_count"] == 10
    assert full["future_endpoint_coverage/endpoints"] == pytest.approx(2 / 3)
    assert full["future_endpoint_coverage/endpoints_count"] == 6
    assert full["future_endpoint_metrics_protocol"] == FUTURE_LABEL_PROTOCOL
    assert full["validation_loss"] == full["learned_open_loop/affect_mse"]

    left, right = validate(0, 2), validate(1, 2)
    for key in ("future_endpoint/emotion_ce", "future_endpoint/vad_mse",
                "future_endpoint_unique/emotion_ce", "future_endpoint_unique/vad_mse"):
        count = left[key + "_count"] + right[key + "_count"]
        weighted = (left[key] * left[key + "_count"] + right[key] * right[key + "_count"]) / count
        assert count == full[key + "_count"]
        assert weighted == pytest.approx(full[key], abs=2e-6)

    legacy = validate(protocol=dynamics.LEGACY_FUTURE_LABEL_PROTOCOL)
    assert legacy["future_endpoint_metrics_protocol"] is None
    assert "future_endpoint/emotion_ce_count" not in legacy
    # Adding true-endpoint metrics preserves the legacy grid's two endpoint
    # queries per dialogue and its raw teacher-affect selection criterion.
    assert legacy["learned_open_loop/endpoint/emotion_ce_count"] == 4
    for key in ("learned_open_loop/affect_mse", "learned_open_loop/endpoint/emotion_ce"):
        assert legacy[key] == pytest.approx(full[key], abs=2e-6)
        assert legacy[key + "_count"] == full[key + "_count"]


def test_one_training_step_does_not_count_grid_endpoint_label_twice(tmp_path, monkeypatch):
    observer, _ = small_models()
    packets = packets_at([1., 2.])
    packets[1]["targets"][0] = [gold(2.)]
    root = tmp_path / "tokens"
    root.mkdir()
    manifest = {"protocol": PROTOCOL,
                "feature_sources": {"test": "synthetic-future-gold"},
                "splits": {"train": ["train"], "val": ["val"], "test": []},
                "dialogues": {}}
    for name in ("train", "val"):
        manifest["dialogues"][name] = {"path": name + ".pt", "cache_id": name,
                                        "packets": len(packets)}
        torch.save({"protocol": PROTOCOL, "cache_id": name, "packets": packets},
                   root / (name + ".pt"))
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    config = default_config()
    config["observer"].update(observer.construction())
    config["state"].update(relation_dim=2, hidden_dim=8, max_integration_step=.5)
    config["train"].update(device="cpu", amp=False, dynamics_steps=1, max_steps=1,
        global_chunks_per_step=1, validate_every=1, checkpoint_every=1, log_every=1,
        validation_max_dialogues=0, observer_batch_size=4, forecast_seconds=[1.],
        modality_dropout=False, observation_dropout=0., future_weight=1.,
        coordinate_weight=0., label_weight=.7)
    config["data"]["token_roots"] = [str(root)]
    config["paths"]["output"] = str(tmp_path / "dynamics")
    source = tmp_path / "observer.pt"
    config["paths"]["observation_checkpoint"] = str(source)
    save_checkpoint(source, {"observer": observer, "teacher": copy.deepcopy(observer)},
        config, {"observer": observer.construction()}, "observation_v3")
    initial = {}
    configure = dynamics.configure_parameters

    def capture(model, core, configuration):
        groups = configure(model, core, configuration)
        initial.update(observer=copy.deepcopy(model), core=copy.deepcopy(core))
        return groups

    monkeypatch.setattr(dynamics, "configure_parameters", capture)
    result = dynamics.run(config)
    assert result["step"] == 1
    row = json.loads((tmp_path / "dynamics" / "metrics.jsonl").read_text().splitlines()[-1])
    # Compute the expected scalar directly from the pre-update models. The only
    # trained origin has no current label, and its future label is grid-aligned.
    model, core = initial["observer"].eval(), initial["core"]
    prior = core.initialize(1)
    pair, _ = dynamics.encode_pair(model, packets[0], "cpu", event_id=0)
    current = core.advance(prior, pair, 1.)
    predicted = core.affect(core.forecast(current, [1.])[0])
    teacher_pair, _ = dynamics.encode_pair(observer, packets[1], "cpu", event_id=1)
    target = torch.stack([item.aff for item in teacher_pair], 1)
    vector_error = F.mse_loss(predicted.float(), target.float()) * predicted.shape[-1]
    output = observer.decode_affect(dynamics.unit_state_readout(predicted))
    endpoint_ce = F.cross_entropy(output["emotion_logits"][:, 0], torch.tensor([0]))
    assert endpoint_ce > 0
    assert row["loss"] == pytest.approx(float((vector_error + .7 * endpoint_ce).detach()),
                                      rel=2e-5, abs=2e-6)
    assert row["cumulative_new_chunks"] == 1
