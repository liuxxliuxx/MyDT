"""Causal, role-equivariant continuous feedback and nonlinear flow contracts."""
import copy
import io

import pytest
import torch

from emotion_ssm.models.state_core import EmotionMemory, KnownFutureInput, UnifiedEmotionStateCore
from test_v3_state_core import observation


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def core(**kwargs):
    torch.manual_seed(762)
    return UnifiedEmotionStateCore(observation_dim=4, relation_dim=3, hidden_dim=12,
        flow_kind="adaptive_dyadic_v1", flow_rank=3, **kwargs)


def close(a, b, atol=2e-6):
    for name in ("fast", "slow", "relation", "baseline", "elapsed"):
        torch.testing.assert_close(getattr(a, name), getattr(b, name), atol=atol, rtol=2e-5)
    torch.testing.assert_close(a.last_event_ids, b.last_event_ids)


def test_cross_coordinate_jacobian_and_state_dependent_coefficients():
    model = core()
    state = model.initialize(1)
    state.fast = torch.tensor([[[.2, .3, -.1, .4], [-.3, .1, .5, -.1]]], requires_grad=True)
    state.slow = torch.randn_like(state.fast)*.2
    derivative = model.autonomous_derivative(state, enable_partner=False)[0]
    gradient = torch.autograd.grad(derivative[0, 0, 0], state.fast, retain_graph=True)[0]
    assert gradient[0, 0, 1:].abs().max() > 1e-5  # Affect i influences affect j.
    rates = model.rates()
    omega = model.max_autonomous_rotation*model.rotation.tanh()
    first = model.adaptive_flow.coefficients(state, rates, omega)[0]
    changed = state.clone()
    changed.slow = changed.slow+.7
    second = model.adaptive_flow.coefficients(changed, rates, omega)[0]
    assert not torch.allclose(first[0], second[0])
    assert not torch.allclose(first[1], second[1])
    with pytest.raises(ValueError, match="state-dependent"):
        model.autonomous_matrix()


def test_persistent_bidirectional_feedback_without_new_actions_or_events():
    model = core()
    initial = model.initialize(1)
    initial.slow[:, 1, 0] = 1.
    off = model.forecast(initial, [3., 7.], enable_partner=False)
    on = model.forecast(initial, [3., 7.], enable_partner=True)
    assert off[1].fast[:, 0].count_nonzero() == 0
    assert off[1].slow[:, 0].count_nonzero() == 0
    assert on[0].fast[:, 0].norm() > .01
    assert on[1].slow[:, 0].norm() > .001
    assert on[1].relation.norm() > 0
    assert torch.equal(on[1].last_event_ids, initial.last_event_ids)
    # A->B->A feedback is ongoing, not just a pulse at the prediction origin.
    amputated = on[0].clone()
    amputated.fast[:, 0].zero_(); amputated.slow[:, 0].zero_()
    resumed = model.forecast(on[0], [4.])[0]
    blocked = model.forecast(amputated, [4.])[0]
    assert not torch.allclose(resumed.z[:, 1], blocked.z[:, 1])
    close(resumed, on[1])


def test_partner_disable_removes_relation_and_other_role_paths():
    model = core()
    state = model.initialize(1)
    state.fast = torch.randn_like(state.fast)
    state.slow = torch.randn_like(state.slow)
    other = state.clone()
    other.fast[:, 1] += 4
    other.slow[:, 1] -= 2
    other.relation += 3
    first = model.forecast(state, [2.], enable_partner=False)[0]
    second = model.forecast(other, [2.], enable_partner=False)[0]
    torch.testing.assert_close(first.z[:, 0], second.z[:, 0], rtol=0, atol=0)
    assert not torch.allclose(model.forecast(state, [2.])[0].z[:, 0],
                              model.forecast(other, [2.])[0].z[:, 0])


def test_role_swap_and_no_input_stream_share_the_forecast_field():
    model = core()
    state = model.initialize(2)
    state.fast = torch.randn_like(state.fast)
    state.slow = torch.randn_like(state.slow)
    state.relation = torch.randn_like(state.relation)
    prediction = model.forecast(state, [.3, 1., 2.3, 5.])[3]
    close(prediction, model.forecast(state, [5.])[0], atol=0)
    close(prediction.role_swap(), model.forecast(state.role_swap(), [5.])[0])
    running = state.clone()
    for _ in range(5):
        running = model.advance(running, [observation(), observation()], 1.)
    close(prediction, running)
    close(model.forecast(state, [0.])[0], state, atol=0)


def test_irregular_time_integration_converges_and_preserves_batch_clocks():
    coarse = core(max_integration_step=.25)
    fine = core(max_integration_step=.03125)
    fine.load_state_dict(coarse.state_dict())
    state = coarse.initialize(2)
    state.fast = torch.randn_like(state.fast)
    state.slow = torch.randn_like(state.slow)
    a = coarse._propagate(state, torch.tensor([.37, 3.81]))
    b = fine._propagate(state, torch.tensor([.37, 3.81]))
    close(a, b, atol=4e-4)
    torch.testing.assert_close(a.elapsed, torch.tensor([.37, 3.81]), rtol=0, atol=0)
    split = coarse.forecast(coarse.forecast(state, [1.13])[0], [2.68])[0]
    direct = coarse.forecast(state, [3.81])[0]
    close(split, direct, atol=4e-4)


def test_feedback_is_differentiable_and_neutral_origin_remains_neutral():
    model = core()
    zero = model.initialize(1)
    neutral = model.forecast(zero, [32.])[0]
    assert neutral.fast.count_nonzero() == neutral.slow.count_nonzero() == neutral.relation.count_nonzero() == 0
    state = model.initialize(1)
    state.fast = torch.randn_like(state.fast).requires_grad_()
    state.slow = torch.randn_like(state.slow).requires_grad_()
    prediction = model.forecast(state, [4.])[0]
    prediction.z[:, 0].square().sum().backward()
    assert state.slow.grad[:, 1].abs().sum() > 0
    for name in ("left", "right", "message.weight", "conditioner.0.weight", "relation_target.0.weight"):
        gradient = dict(model.adaptive_flow.named_parameters())[name].grad
        assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0, name


def test_event_once_conditional_no_future_affect_and_checkpoint_roundtrip():
    model = core()
    state = model.initialize(2)
    event = observation(present=True, value=.8, event=True, event_id=10)
    empty = observation()
    once = model.advance(state, [event, empty], 0., correct=False)
    close(once, model.advance(once, [event, empty], 0., correct=False), atol=0)
    future = observation(present=True, fresh=True, action=True, duration=1., value=.4)
    planned = [KnownFutureInput(2., [future, empty])]
    a = model.forecast(once, [1., 2., 4.], planned)
    future.aff.fill_(float("nan"))
    b = model.forecast(once, [1., 2., 4.], planned)
    for first, second in zip(a, b):
        close(first, second, atol=0)
    close(a[0], model.forecast(once, [1.])[0])
    buffer = io.BytesIO()
    torch.save((model.get_config(), model.state_dict(), once.serialize()), buffer)
    buffer.seek(0)
    config, weights, memory = torch.load(buffer, weights_only=True)
    restored = UnifiedEmotionStateCore.from_config(config)
    restored.load_state_dict(weights, strict=True)
    close(model.forecast(once, [4.])[0], restored.forecast(EmotionMemory.deserialize(memory), [4.])[0], atol=0)


def test_long_feedback_rollout_stays_finite_without_state_clipping():
    model = core()
    state = model.initialize(2)
    state.fast = torch.randn_like(state.fast)*3
    state.slow = torch.randn_like(state.slow)*3
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        results = model.forecast(state, [1., 30., 120., 300.])
    for value in results:
        for name in ("fast", "slow", "relation"):
            tensor = getattr(value, name)
            assert tensor.dtype == torch.float32 and torch.isfinite(tensor).all()
            assert tensor.abs().max() < 20


def test_legacy_constructor_has_unchanged_weight_and_construction_schema():
    model = UnifiedEmotionStateCore(observation_dim=4, relation_dim=3, hidden_dim=8)
    assert "flow_kind" not in model.get_config()
    assert not any(name.startswith("adaptive_flow.") for name in model.state_dict())
    restored = UnifiedEmotionStateCore.from_config(copy.deepcopy(model.get_config()))
    restored.load_state_dict(model.state_dict(), strict=True)
    assert restored.adaptive_flow is None


def test_validation_records_actual_feedback_and_coordinate_mixing():
    from emotion_ssm.train.dynamics_v3 import state_statistics, SufficientStatistics
    model = core()
    state = model.initialize(1)
    state.fast = torch.randn_like(state.fast)
    state.slow = torch.randn_like(state.slow)
    statistics = SufficientStatistics()
    state_statistics(statistics, "probe", model, state)
    result = statistics.metrics()
    assert result["probe/adaptive/partner_feedback_norm"] > 0
    assert result["probe/adaptive/cross_coordinate_drive_norm"] > 0
