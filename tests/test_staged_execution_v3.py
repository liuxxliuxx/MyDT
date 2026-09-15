import copy

import pytest
import torch

from emotion_ssm.train import dynamics_staged_v3 as staged
from emotion_ssm.train.dynamics_v3 import synchronize_gradients
from emotion_ssm.train.staged_dynamics_support import (
    build_origin_bank, parameter_partition, prediction_loss, vectorized_endpoint_loss,
)
from emotion_ssm.utils.dynamics_execution import synchronize_gradients_batched
from test_staged_dynamics_v3 import setup_models


@pytest.mark.parametrize('phase', ['fixed', 'joint', 'calibration'])
def test_optimized_forecast_loss_and_gradients(tmp_path, phase):
    _, observer, _, reference, cache, collection = setup_models(tmp_path)
    bank = build_origin_bank(observer, reference, cache, collection, [0, 1], [1, 2], stride=1)
    optimized = copy.deepcopy(reference)
    optimized.configure_execution('optimized')
    other = copy.deepcopy(observer)
    parameters = [parameter_partition(o, c, phase) for o, c in [(observer, reference), (other, optimized)]]
    indices = torch.tensor([0, 1, 1, 2])  # Preserve repeated-origin label weighting.
    results = []
    for core, obs, params in [(reference, observer, parameters[0]), (optimized, other, parameters[1])]:
        loss, output = prediction_loss(core, bank, indices, 'cpu', obs, 1.)
        if loss.requires_grad:
            loss.backward()
        results.append((loss, output, [None if p.grad is None else p.grad.clone() for p in params]))
    torch.testing.assert_close(results[0][0], results[1][0], rtol=2e-6, atol=1e-7)
    for a, b in zip(results[0][1], results[1][1]):
        torch.testing.assert_close(a.z, b.z, rtol=1e-6, atol=1e-7)
    for a, b in zip(results[0][2], results[1][2]):
        if a is None or b is None:
            assert a is b
        else:
            torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-7)
    assert list(reference.state_dict()) == list(optimized.state_dict())


def test_vectorized_label_loss_keeps_per_query_denominators_and_missing_labels():
    torch.manual_seed(4)
    labels = [dict(emotion=2, vad=[.2, .8, 0.], vad_mask=[True, True, False]),
              dict(emotion=-1, vad=[.1, 99., 99.], vad_mask=[True, False, False], intensity=.7, intensity_mask=True),
              dict(emotion=3), dict(emotion=-1), dict(intensity=.3, intensity_mask=True)]
    decoded = dict(emotion_logits=torch.randn(5, 7, requires_grad=True),
                   vad=torch.randn(5, 3, requires_grad=True), intensity=torch.randn(5, requires_grad=True))
    ce = torch.nn.functional.cross_entropy
    expected = torch.stack([
        ce(decoded['emotion_logits'][0:1], torch.tensor([2])) + (decoded['vad'][0, :2]-torch.tensor([.2, .8])).square().mean(),
        (decoded['vad'][1, 0]-.1).square() + (decoded['intensity'][1]-.7).square(),
        ce(decoded['emotion_logits'][2:3], torch.tensor([3])),
        (decoded['intensity'][4]-.3).square(),
    ]).mean()
    actual = vectorized_endpoint_loss(decoded, labels)
    torch.testing.assert_close(actual, expected)
    before = torch.autograd.grad(expected, tuple(decoded.values()), retain_graph=True)
    after = torch.autograd.grad(actual, tuple(decoded.values()))
    for a, b in zip(before, after):
        torch.testing.assert_close(a, b)
    assert vectorized_endpoint_loss(decoded, [{}]*5) is None


def test_fractional_forecasts_and_query_validation(tmp_path):
    _, _, _, core, _, _ = setup_models(tmp_path)
    state = core.initialize(2, 'cpu')
    state.fast = torch.randn_like(state.fast)
    state.slow = torch.randn_like(state.slow)
    queries = [0., .12, .5, .73, 1., 2.13]
    reference = core.forecast(state, queries)
    core.configure_execution('optimized')
    for a, b in zip(reference, core.forecast(state, queries)):
        torch.testing.assert_close(a.z, b.z, rtol=0, atol=0)
        torch.testing.assert_close(a.elapsed, b.elapsed, rtol=0, atol=0)
    for invalid in [[-1.], [1., .5], [float('nan')]]:
        with pytest.raises(ValueError):
            core.forecast(state, invalid)


def test_grouped_gradients_preserve_mean_and_missing_gradients():
    reference = [torch.nn.Parameter(torch.randn(2, 3)), torch.nn.Parameter(torch.randn(7)), torch.nn.Parameter(torch.ones(1))]
    reference[0].grad = torch.randn_like(reference[0]); reference[1].grad = torch.randn_like(reference[1])
    other = [torch.nn.Parameter(p.detach().clone()) for p in reference]
    for a, b in zip(reference, other):
        b.grad = None if a.grad is None else a.grad.clone()
    assert synchronize_gradients(reference, 3, 'cpu') == synchronize_gradients_batched(other, 3, 'cpu') == 3
    for a, b in zip(reference, other):
        if a.grad is None:
            assert b.grad is None
        else:
            torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
    other[0].grad.fill_(float('nan'))
    with pytest.raises(FloatingPointError):
        synchronize_gradients_batched(other, 1, 'cpu')


def test_execution_override_survives_resume(tmp_path, monkeypatch):
    from test_staged_dynamics_v3 import test_small_complete_stages_and_self_contained_resume
    original = staged.run
    monkeypatch.setattr(staged, 'run', lambda config: original(config, execution_override='optimized'))
    test_small_complete_stages_and_self_contained_resume(tmp_path, monkeypatch)
