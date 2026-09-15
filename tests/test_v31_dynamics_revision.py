"""Gradient semantics and checkpoint boundaries for the revised dynamics stage."""
import copy

import pytest
import torch
from torch.nn import functional as F

from emotion_ssm.config_v3 import DYNAMICS_REVISION, FORMAT_VERSION, PROTOCOL, default_config
from emotion_ssm.data.packets_v3 import collate_role_features, empty_role_features
from emotion_ssm.models.state_core import UnifiedEmotionStateCore
from emotion_ssm.models.token_observer import TokenObserver
from emotion_ssm.train import dynamics_v3 as dynamics
from emotion_ssm.utils.checkpoint_v3 import (
    load_observer, read_checkpoint, require_training_revision, restore_training,
)


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def observer(dimension=4):
    torch.manual_seed(6666)
    return TokenObserver(dict(audio_dim=4, text_dim=4, model_dim=8,
        affect_dim=dimension, num_layers=1, num_heads=2, summary_dim=2)).eval()


def emotion_label(emotion=0, end=2.):
    return {"emotion": emotion, "end": end, "intensity_mask": False,
            "vad_mask": [False, False, False]}


def test_unit_state_readout_is_invariant_to_positive_scaling_above_threshold():
    value = torch.tensor([[[.3, -.4, .1, .2], [-.2, .7, .3, -.5]]])
    expected = dynamics.unit_state_readout(value)
    for scale in (.001, .5, 3., 1000.):
        torch.testing.assert_close(dynamics.unit_state_readout(value*scale), expected)
    torch.testing.assert_close(expected.norm(dim=-1), torch.ones(1, 2))


@pytest.mark.parametrize("magnitude", [0., 1e-12, 9e-6, 1e-5])
def test_near_zero_readout_and_gradient_are_finite_and_zero(magnitude):
    value = torch.tensor([[magnitude, 0., 0., 0.]], requires_grad=True)
    output = dynamics.unit_state_readout(value)
    gradient = torch.autograd.grad((output*torch.tensor([[1., 2., 3., 4.]])).sum(), value)[0]
    assert torch.isfinite(output).all() and torch.isfinite(gradient).all()
    assert output.count_nonzero() == 0 and gradient.count_nonzero() == 0


def test_normalized_production_label_loss_removes_radial_ce_incentive():
    model = observer().requires_grad_(False)
    with torch.no_grad():
        model.emotion_head.weight.zero_()
        model.emotion_head.bias.zero_()
        model.emotion_head.weight[0, 0] = 1.
    labels = [[emotion_label()], []]
    value = torch.tensor([[[2., 1., 0., 0.], [0., 0., 0., 0.]]], requires_grad=True)
    raw = dynamics.label_loss(model, value, labels, normalize=False)
    normalized = dynamics.label_loss(model, value, labels, normalize=True)
    raw_gradient = torch.autograd.grad(raw, value)[0]
    unit_gradient = torch.autograd.grad(normalized, value)[0]
    assert (raw_gradient*value.detach()).sum() < -.01
    assert abs(float((unit_gradient*value.detach()).sum())) < 1e-7
    assert unit_gradient.norm() > 0  # Directional learning survives normalization.
    torch.testing.assert_close(dynamics.label_loss(model, value*7., labels, normalize=True), normalized)
    assert dynamics.label_loss(model, value*7., labels, normalize=False) < raw
    assert all(parameter.grad is None for parameter in model.parameters())


def forecast_fixture(dimension):
    model = observer(dimension)
    core = UnifiedEmotionStateCore(observation_dim=dimension, relation_dim=2, hidden_dim=8)
    state = core.initialize(1)
    state.fast = torch.linspace(-.4, .7, 2*dimension).reshape(1, 2, dimension).requires_grad_()
    state.slow = torch.full_like(state.fast, .08)
    targets = {"times": [1., 2., 5.],
        "affect": F.normalize(torch.arange(1., 6*dimension+1).reshape(3, 2, dimension), dim=-1),
        "valid": torch.tensor([[False, False], [True, False], [True, True]])}
    dialogue = {"packets": [{"targets": [[], []]}, {"targets": [[emotion_label()], []]},
                             {"targets": [[], [emotion_label(1, 5.)]]}]}
    return model, core, state, targets, dialogue


@pytest.mark.parametrize("dimension", [4, 128])
def test_vector_forecast_scales_optimization_and_gradient_but_not_mse_logs(dimension):
    model, core, state, targets, dialogue = forecast_fixture(dimension)
    records, prediction_records, losses, gradients = [], [], [], []
    for vector_loss in (False, True):
        statistics, prediction = dynamics.SufficientStatistics(), dynamics.SufficientStatistics()
        loss = dynamics.forecast_objective(core, model, state, targets, dialogue, 0, [1., 4.],
            statistics=statistics, prediction_statistics=prediction,
            labels_weight=0., vector_loss=vector_loss)
        losses.append(loss)
        gradients.append(torch.autograd.grad(loss, state.fast)[0])
        records.append(statistics.metrics())
        prediction_records.append(prediction.metrics())
    torch.testing.assert_close(losses[1], losses[0]*dimension)
    torch.testing.assert_close(gradients[1], gradients[0]*dimension)
    assert losses[0] > 0 and gradients[0].abs().sum() > 0
    assert records[0] == records[1]
    assert prediction_records[0] == prediction_records[1]
    assert records[0]["learned_open_loop/affect_mse_count"] == 3*dimension
    assert prediction_records[0]["future_affect_mse_count"] == 3*dimension
    assert records[0]["learned_open_loop/affect_mse"] == pytest.approx(
        prediction_records[0]["future_affect_mse"])


def test_vector_scaling_does_not_multiply_the_endpoint_label_term():
    model, core, state, targets, dialogue = forecast_fixture(4)
    def loss(vector, labels):
        return dynamics.forecast_objective(core, model, state, targets, dialogue, 0, [1., 4.],
            labels_weight=labels, vector_loss=vector, normalize_labels=True)
    coordinate = loss(False, 0.)
    old_total, vector_total = loss(False, .7), loss(True, .7)
    assert old_total > coordinate
    torch.testing.assert_close(vector_total-old_total, coordinate*3)


def test_configure_parameters_keeps_shared_observer_fixed_during_optimizer_step():
    model = observer()
    core = UnifiedEmotionStateCore(observation_dim=4, relation_dim=2, hidden_dim=8)
    groups = dynamics.configure_parameters(model, core, default_config())
    permitted = {name for name, _ in model.named_parameters()
                 if name.startswith(("event_head.", "action_head."))}
    assert {name for name, value in model.named_parameters() if value.requires_grad} == permitted
    included = [id(value) for group in groups for value in group["params"]]
    assert len(included) == len(set(included))
    assert set(included) == {id(p) for p in core.parameters()} | {
        id(p) for p in model.parameters() if p.requires_grad}
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    features = empty_role_features(audio_dim=4, text_dim=4, now=1., dt=1.)
    for mode in ("audio", "prosody", "text"):
        features[mode+"_mask"][:] = True
        features[mode+"_tokens"] = torch.randn_like(features[mode+"_tokens"])
    features["text_fresh_mask"][:] = True
    features.update(fresh_observation=torch.tensor([True, False, True]),
        event_present=torch.tensor(True), action_present=torch.tensor(True), action_duration=torch.tensor(1.))
    optimizer = torch.optim.AdamW(groups, weight_decay=.1)
    encoded = model.encode_clean(collate_role_features([features]), "AT")["observation"]
    ((encoded.event-.3).square().mean()+(encoded.action+.4).square().mean()).backward()
    assert all(value.grad is None for name, value in model.named_parameters() if name not in permitted)
    optimizer.step()
    changed = {name for name, value in model.named_parameters() if not torch.equal(before[name], value)}
    assert changed <= permitted
    assert any(name.startswith("event_head.") for name in changed)
    assert any(name.startswith("action_head.") for name in changed)


def checkpoint_fixture(revision):
    model = observer()
    config = default_config()
    if revision is None:
        config["train"].pop("dynamics_revision")
    else:
        config["train"]["dynamics_revision"] = revision
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0123)
    payload = {"format_version": FORMAT_VERSION, "protocol": PROTOCOL, "kind": "dynamics_v3",
        "config": config, "construction": {"observer": model.construction()},
        "models": {"observer": copy.deepcopy(model.state_dict())}, "optimizer": optimizer.state_dict()}
    return model, payload


@pytest.mark.parametrize("revision", [None, "legacy", "future-unknown-revision"])
def test_old_dynamics_optimizer_rejected_before_weights_are_modified(revision):
    model, payload = checkpoint_fixture(revision)
    with torch.no_grad():
        next(model.parameters()).add_(1.)
    before = next(model.parameters()).detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.5)
    with pytest.raises(ValueError, match="optimizer cannot resume"):
        restore_training(payload, {"observer": model}, optimizer=optimizer, restore_rng=False)
    torch.testing.assert_close(next(model.parameters()), before, rtol=0, atol=0)
    assert optimizer.param_groups[0]["lr"] == .5


def test_old_dynamics_inference_weights_remain_readable(tmp_path):
    source, payload = checkpoint_fixture(None)
    path = tmp_path/"old_dynamics.pt"
    torch.save(payload, path)
    assert "dynamics_revision" not in read_checkpoint(path)["config"]["train"]
    loaded, _ = load_observer(path, device="cpu")
    for name, value in source.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[name], value, rtol=0, atol=0)
    target = observer()
    restore_training(path, {"observer": target}, optimizer=None, restore_rng=False)
    for name, value in source.state_dict().items():
        torch.testing.assert_close(target.state_dict()[name], value, rtol=0, atol=0)


def test_matching_revision_optimizer_restores_and_default_is_current():
    assert default_config()["train"]["dynamics_revision"] == DYNAMICS_REVISION
    source, payload = checkpoint_fixture(DYNAMICS_REVISION)
    target = observer()
    optimizer = torch.optim.AdamW(target.parameters(), lr=.5)
    require_training_revision(payload)
    restore_training(payload, {"observer": target}, optimizer=optimizer, restore_rng=False)
    assert optimizer.param_groups[0]["lr"] == .0123
    for name, value in source.state_dict().items():
        torch.testing.assert_close(target.state_dict()[name], value, rtol=0, atol=0)


def test_observation_optimizer_is_not_blocked_by_dynamics_revision_guard():
    model, payload = checkpoint_fixture(None)
    payload["kind"] = "observation_v3"
    optimizer = torch.optim.AdamW(model.parameters(), lr=.5)
    restore_training(payload, {"observer": model}, optimizer=optimizer, restore_rng=False)
    assert optimizer.param_groups[0]["lr"] == .0123


def test_run_rejects_old_checkpoint_before_device_or_data_initialization(monkeypatch):
    _, payload = checkpoint_fixture(None)
    config = default_config()
    config["paths"]["resume"] = "old-checkpoint.pt"
    monkeypatch.setattr(dynamics, "read_checkpoint", lambda path: payload)
    monkeypatch.setattr(dynamics, "_distributed_device", lambda config: pytest.fail("Device initialization preceded revision check"))
    with pytest.raises(ValueError, match="optimizer cannot resume"):
        dynamics.run(config)
