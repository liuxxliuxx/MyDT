"""Training baseline gradients survive TBPTT and checkpoint restoration."""
import io

import pytest
import torch

from emotion_ssm.models.state_core import (
    EmotionMemory, StateObservation, UnifiedEmotionStateCore,
)


def _core():
    torch.manual_seed(6666)
    return UnifiedEmotionStateCore(observation_dim=4, relation_dim=3, hidden_dim=8)


def _pair(tick, *, present=True):
    result = []
    for role in (0, 1):
        value = (0.3 + 0.05 * tick) * (1 if role == 0 else -1)
        result.append(StateObservation(
            aff=torch.full((1, 4), value),
            event=torch.full((1, 4), value),
            action=torch.full((1, 4), value),
            reliability=torch.full((1, 3), 0.7),
            modality_mask=torch.full((1, 3), present, dtype=torch.bool),
            event_present=torch.full((1,), present, dtype=torch.bool),
            action_duration=torch.full((1,), 0.75 if present else 0.0),
            fresh_observation=torch.full((1, 3), present, dtype=torch.bool),
            action_present=torch.full((1,), present, dtype=torch.bool),
            event_id=torch.full((1,), tick, dtype=torch.long),
        ))
    return result


def _assert_same_history(first, second):
    for name in ("fast", "slow", "relation", "elapsed", "last_event_ids"):
        assert getattr(first, name) is getattr(second, name)


def test_later_tbptt_segment_reconnects_baseline_without_resetting_history():
    core = _core()
    current = core.advance(core.initialize(1), _pair(0), 0.75)
    core.affect(core.forecast(current, [2.0])[0]).square().mean().backward()
    memory = current.detach()
    core.zero_grad(set_to_none=True)
    assert not memory.baseline.requires_grad
    assert memory.fast.abs().sum() > 0
    assert memory.slow.abs().sum() > 0
    assert memory.relation.abs().sum() > 0

    rebound = core.rebind_trainable_baseline(memory)
    _assert_same_history(rebound, memory)
    assert not rebound.fast.requires_grad
    assert not rebound.slow.requires_grad
    torch.testing.assert_close(core.affect(rebound), core.affect(memory), rtol=0, atol=0)

    next_state = core.advance(rebound, _pair(1), 0.75)
    prediction = core.affect(core.forecast(next_state, [4.0])[0])
    (prediction - 0.8).square().mean().backward()
    assert core.baseline.grad is not None
    assert torch.isfinite(core.baseline.grad).all()
    assert core.baseline.grad.abs().sum() > 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_rebind_uses_current_parameter_value_and_preserves_memory_dtype(dtype):
    core = _core()
    memory = core.initialize(2, dtype=dtype).detach().clone()
    with torch.no_grad():
        core.baseline.copy_(torch.tensor([0.1, -0.2, 0.3, -0.4]))
    rebound = core.rebind_trainable_baseline(memory)
    _assert_same_history(rebound, memory)
    assert rebound.baseline.dtype == dtype
    assert memory.baseline.count_nonzero() == 0
    torch.testing.assert_close(
        rebound.baseline, core.baseline.to(dtype)[None, None].expand_as(memory.fast),
        rtol=0, atol=0,
    )
    rebound.baseline.sum().backward()
    torch.testing.assert_close(core.baseline.grad, torch.full((4,), 4.0), rtol=0, atol=0)


def _train_segment(core, optimizer, memory, tick):
    optimizer.zero_grad(set_to_none=True)
    memory = core.rebind_trainable_baseline(memory)
    current = core.advance(memory, _pair(tick), 0.75)
    predicted = torch.stack([core.affect(value) for value in core.forecast(current, [1.0, 4.0])])
    (predicted - 0.4).square().mean().backward()
    gradient = core.baseline.grad.detach().clone()
    output = predicted.detach().clone()
    memory = current.detach()
    optimizer.step()
    return memory, output, gradient


def test_resume_and_continuous_segments_match_forward_baseline_gradients_and_updates():
    core = _core()
    optimizer = torch.optim.AdamW(core.parameters(), lr=0.01)
    memory, _, _ = _train_segment(core, optimizer, core.initialize(1), 0)
    stream = io.BytesIO()
    torch.save({"model": core.state_dict(), "optimizer": optimizer.state_dict(),
                "memory": memory.serialize()}, stream)
    stream.seek(0)
    checkpoint = torch.load(stream, weights_only=True)
    restored_core = UnifiedEmotionStateCore.from_config(core.get_config())
    restored_core.load_state_dict(checkpoint["model"])
    restored_optimizer = torch.optim.AdamW(restored_core.parameters(), lr=0.01)
    restored_optimizer.load_state_dict(checkpoint["optimizer"])
    restored_memory = EmotionMemory.deserialize(checkpoint["memory"])
    assert restored_memory.baseline.data_ptr() != restored_core.baseline.data_ptr()

    for tick in (1, 2, 3):
        memory, output, gradient = _train_segment(core, optimizer, memory, tick)
        restored_memory, restored_output, restored_gradient = _train_segment(
            restored_core, restored_optimizer, restored_memory, tick)
        assert gradient.abs().sum() > 0
        torch.testing.assert_close(output, restored_output, rtol=0, atol=0)
        torch.testing.assert_close(gradient, restored_gradient, rtol=0, atol=0)
        for name in ("fast", "slow", "relation", "baseline", "elapsed", "last_event_ids"):
            torch.testing.assert_close(getattr(memory, name), getattr(restored_memory, name),
                                       rtol=0, atol=0)
        for name, value in core.state_dict().items():
            torch.testing.assert_close(value, restored_core.state_dict()[name], rtol=0, atol=0)


def test_inference_preserves_explicit_memory_baseline():
    core = _core().eval()
    memory = core.initialize(1).detach()
    memory.baseline = torch.full_like(memory.baseline, 0.6)
    with torch.no_grad():
        core.baseline.fill_(-0.2)
        advanced = core.advance(memory, _pair(0, present=False), 0.75)
        predicted = core.forecast(advanced, [2.0])[0]
    assert advanced.baseline is memory.baseline
    assert predicted.baseline is memory.baseline
    torch.testing.assert_close(core.affect(predicted), torch.full_like(memory.fast, 0.6))
