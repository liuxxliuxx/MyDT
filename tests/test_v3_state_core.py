"""State-level contracts, using known dynamics and controlled interventions."""
import io

import pytest
import torch

from emotion_ssm.models.state_core import (
    EmotionMemory, KnownFutureInput, StateObservation, UnifiedEmotionStateCore,
)


def model(**kwargs):
    torch.manual_seed(6666)
    return UnifiedEmotionStateCore(observation_dim=4, relation_dim=3,
                                   hidden_dim=8, **kwargs)


def observation(batch=2, *, value=0.0, present=False, fresh=False,
                action=False, duration=0.0, event=False, event_id=None):
    return StateObservation(
        aff=torch.full((batch, 4), float(value)),
        event=torch.full((batch, 4), float(value)),
        action=torch.full((batch, 4), float(value)),
        reliability=torch.ones(batch, 3),
        modality_mask=torch.full((batch, 3), present, dtype=torch.bool),
        event_present=torch.full((batch,), event, dtype=torch.bool),
        action_duration=torch.full((batch,), float(duration)),
        fresh_observation=torch.full((batch, 3), fresh, dtype=torch.bool),
        action_present=torch.full((batch,), action, dtype=torch.bool),
        event_id=None if event_id is None else torch.full((batch,), event_id, dtype=torch.long),
    )


def assert_memory_close(a, b, atol=2e-6):
    for name in ("fast", "slow", "relation", "baseline", "elapsed"):
        torch.testing.assert_close(getattr(a, name), getattr(b, name), rtol=2e-5, atol=atol)
    torch.testing.assert_close(a.last_event_ids, b.last_event_ids)


def test_pure_decay_has_known_unequal_time_solution_and_composition():
    core = model()
    state = core.initialize(2)
    state.fast = torch.ones_like(state.fast)
    state.slow = torch.ones_like(state.slow)
    state.relation = torch.ones_like(state.relation)
    direct = core.decay_only(state, torch.tensor([3.0, 7.5]))
    split = core.decay_only(core.decay_only(state, [1.0, 2.5]), [2.0, 5.0])
    assert_memory_close(direct, split)
    torch.testing.assert_close(direct.fast[:, 0, 0], torch.exp(-torch.tensor([3.0, 7.5]) / 2))
    torch.testing.assert_close(direct.slow[:, 0, 0], torch.exp(-torch.tensor([3.0, 7.5]) / 120))


def test_exact_autonomous_query_clock_and_energy_stability():
    core = model()
    state = core.initialize(2)
    state.fast = torch.randn_like(state.fast)
    state.slow = torch.randn_like(state.slow)
    query = core.forecast(state, [0.0, 0.3, 2.0, 7.0, 32.0])
    assert_memory_close(query[0], state)
    # A query at .3 cannot change the forecast at 7 seconds.
    assert_memory_close(query[3], core.forecast(state, [7.0])[0])
    continued = core.forecast(core.forecast(state, [2.0])[0], [5.0])[0]
    assert_memory_close(query[3], continued)
    energies = [s.fast.square().sum() + s.slow.square().sum() for s in query]
    assert all(a >= b - 1e-5 for a, b in zip(energies, energies[1:]))
    assert not torch.allclose(query[3].fast, core.decay_only(state, 7.0).fast)


def test_fast_response_does_not_replace_slow_memory_every_second():
    core = model(initial_autonomous_rotation=0.0)
    state = core.initialize(2)
    state.slow = torch.ones_like(state.slow)
    pair = [observation(present=True, fresh=True), observation(present=True, fresh=True)]
    first = core.advance(state, pair, 1.0, enable_partner=False)
    assert first.slow.min() > 0.98
    assert core.affect(first).abs().max() < 0.4
    for _ in range(29):
        first = core.advance(first, pair, 1.0, enable_partner=False)
    # Fast memory accommodates contradictory current evidence; thirty fresh
    # neutral packets must not reset the slower initial signal to a constant.
    assert first.slow.min() > 0.65
    assert first.fast.max() < -0.3


def test_no_event_or_action_with_stored_text_and_missing_values():
    core = model()
    state = core.initialize(2)
    state.slow = torch.ones_like(state.slow) * 0.4
    context = observation(present=True, fresh=False, value=100.0)
    context.modality_mask[:, :2] = False
    result = core.advance(state, [context, context], 4.0)
    assert_memory_close(result, core.forecast(state, [4.0])[0])
    missing = observation()
    missing.aff.fill_(float("nan"))
    missing.event.fill_(float("nan"))
    missing.action.fill_(float("nan"))
    assert_memory_close(core.advance(state, [missing, missing], 4.0), result)


def test_neutral_origin_has_no_bias_driven_coupling():
    core = model()
    state = core.initialize(2)
    neutral = observation(present=True, action=True, duration=1.0)
    result = core.advance(state, [neutral, neutral], 1.0, correct=False)
    assert torch.count_nonzero(result.fast) == 0
    assert torch.count_nonzero(result.slow) == 0
    assert torch.count_nonzero(result.relation) == 0


def test_nonverbal_action_can_couple_and_duration_controls_amount():
    core = model(initial_autonomous_rotation=0.0)
    state = core.initialize(2)
    partner = observation(present=True, action=True, duration=1.0, value=1.0)
    partner.modality_mask[:, [0, 2]] = False  # visible, silent, no words
    empty = observation()
    result = core.advance(state, [empty, partner], 1.0, correct=False)
    assert result.fast[:, 0].abs().sum() > 0
    short = observation(present=True, action=True, duration=0.25, value=1.0)
    short.modality_mask[:, [0, 2]] = False
    short_result = core.advance(state, [empty, short], 1.0, correct=False)
    assert short_result.fast[:, 0].norm() < result.fast[:, 0].norm()
    assert_memory_close(core.advance(state, [empty, partner], 1.0, correct=False, enable_partner=False),
                        core.forecast(state, [1.0])[0])


def test_event_ids_are_once_only_and_no_text_mask_blocks_event():
    core = model()
    state = core.initialize(2)
    pulse = observation(present=True, event=True, value=1.0, event_id=42)
    empty = observation()
    once = core.advance(state, [pulse, empty], 0.0, correct=False)
    twice = core.advance(once, [pulse, empty], 0.0, correct=False)
    assert once.fast.abs().sum() > 0
    assert_memory_close(once, twice)
    pulse.modality_mask[:, 2] = False
    assert_memory_close(core.advance(state, [pulse, empty], 0.0, correct=False), state)


def test_multistep_role_renaming_and_directed_relations_are_equivariant():
    core = model()
    state = core.initialize(2)
    state.fast = torch.randn_like(state.fast)
    state.slow = torch.randn_like(state.slow)
    state.relation = torch.randn_like(state.relation)
    swapped = state.role_swap()
    for tick in range(4):
        left = observation(present=True, fresh=True, action=True, duration=0.5,
                           event=True, value=0.3 + tick, event_id=tick)
        right = observation(present=True, fresh=True, action=True, duration=0.25,
                            value=-0.8 + tick / 4)
        state = core.advance(state, [left, right], 0.7)
        swapped = core.advance(swapped, [right, left], 0.7)
        assert_memory_close(state.role_swap(), swapped)
    # Equivariance does not force actual opposite-direction relation values equal.
    assert not torch.allclose(state.relation[:, 0], state.relation[:, 1])


def test_conditional_forecast_never_reads_future_affect_and_requires_declaration():
    core = model()
    state = core.initialize(2)
    left = observation(present=True, fresh=True, action=True, duration=1.0, value=0.6)
    right = observation()
    known = KnownFutureInput(2.0, [left, right])
    before = core.forecast(state, [1.0, 2.0, 4.0], [known])
    left.aff.fill_(float("nan"))  # should not even require valid future affect
    after = core.forecast(state, [1.0, 2.0, 4.0], [known])
    for first, second in zip(before, after):
        assert_memory_close(first, second)
    assert_memory_close(before[0], core.forecast(state, [1.0])[0])
    assert before[1].fast.abs().sum() > 0
    with pytest.raises(ValueError, match="explicit"):
        core.forecast(state, [4], [(2, [left, right])])
    with pytest.raises(ValueError, match="origin"):
        core.forecast(state, [4], [KnownFutureInput(2, [left, right], available_at=1)])


def test_state_history_and_future_loss_backpropagate_to_observation_and_core():
    core = model()
    state = core.initialize(2)
    first = observation(present=True, fresh=True, action=True, duration=1.0, value=0.6)
    first.aff.requires_grad_()
    first.action.requires_grad_()
    for _ in range(3):
        state = core.advance(state, [first, observation()], 1.0)
    prediction = core.forecast(state, [4.0])[0]
    loss = (core.affect(prediction) - 0.4).square().mean()
    loss.backward()
    assert first.aff.grad is not None and first.aff.grad.abs().sum() > 0
    assert first.action.grad is not None and first.action.grad.abs().sum() > 0
    assert core.rotation.grad is not None and core.rotation.grad.abs().sum() > 0
    assert core.slow_correction_logits.grad.abs().sum() > 0
    assert core.influence[0].weight.grad.abs().sum() > 0


def test_memory_and_standalone_model_checkpoint_roundtrip_and_context_width():
    core = model()
    state = core.initialize(2)
    pair = [observation(present=True, fresh=True, value=1), observation()]
    state = core.advance(state, pair, 1)
    assert state.fast.grad_fn is not None
    assert state.detach().fast.grad_fn is None
    copied = state.clone()
    assert copied.fast.data_ptr() != state.fast.data_ptr()
    stream = io.BytesIO()
    torch.save({"config": core.get_config(), "model": core.state_dict(), "memory": state.serialize()}, stream)
    stream.seek(0)
    checkpoint = torch.load(stream, weights_only=True)
    restored = UnifiedEmotionStateCore.from_config(checkpoint["config"])
    restored.load_state_dict(checkpoint["model"])
    memory = EmotionMemory.deserialize(checkpoint["memory"])
    assert_memory_close(core.forecast(state, [3])[0], restored.forecast(memory, [3])[0])
    for variant in ("none", "affect", "self", "dyadic"):
        condition = core.context(state, pair, variant)
        assert condition.shape == (2, core.context_dim)
        if variant == "none":
            assert torch.count_nonzero(condition) == 0


@pytest.mark.parametrize("dt", [-1.0, float("nan"), [1, 2, 3]])
def test_invalid_clocks_fail_explicitly(dt):
    core = model()
    with pytest.raises(ValueError):
        core.advance(core.initialize(2), [observation(), observation()], dt)


@pytest.mark.parametrize("input_dtype", [torch.float16, torch.bfloat16])
def test_low_precision_observation_never_quantizes_persistent_slow_memory(input_dtype):
    core = model(initial_autonomous_rotation=0.0, slow_tau=1800., slow_correction_rate=1e-6)
    state = core.initialize(2, dtype=input_dtype)
    state.slow = torch.ones_like(state.slow)
    pair = [observation(present=True, fresh=True, value=.01), observation()]
    for item in pair:
        for name in ("aff", "event", "action", "reliability"):
            setattr(item, name, getattr(item, name).to(input_dtype))
    reference = state.clone()
    reference_pair = [StateObservation(**{name: (value.float() if torch.is_tensor(value) and value.is_floating_point()
                                               else value) for name, value in vars(item).items()}) for item in pair]
    # Also accept a session loaded or moved to low precision by an outer caller.
    state = state.to(dtype=input_dtype)
    for _ in range(100):
        state = core.advance(state, pair, 1., enable_partner=False)
        reference = core.advance(reference, reference_pair, 1., enable_partner=False)
        assert all(getattr(state, name).dtype == torch.float32 for name in
                   ("fast", "slow", "relation", "baseline", "elapsed"))
    assert state.slow.max() < .95
    assert_memory_close(state, reference, atol=2e-6)
    # The second person has no fresh evidence: its decrease is analytically known.
    expected = torch.exp(torch.tensor(-100. / 1800.))
    torch.testing.assert_close(state.slow[:, 1], expected.expand_as(state.slow[:, 1]), atol=4e-6, rtol=2e-5)


def test_float64_state_is_preserved_and_autocast_outputs_keep_fp32_memory():
    core = model()
    double = core.initialize(2, dtype=torch.float64)
    assert core.advance(double, [observation(), observation()], 1.).slow.dtype == torch.float64
    pair = [observation(present=True, fresh=True, action=True, duration=1., value=.2), observation()]
    state = core.initialize(2, dtype=torch.bfloat16)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        for _ in range(3):
            state = core.advance(state, pair, 1.)
    assert state.fast.dtype == state.slow.dtype == state.relation.dtype == torch.float32
    loss = state.fast.square().sum() + state.slow.square().sum()
    loss.backward()
    assert core.slow_correction_logits.grad is not None
    assert torch.isfinite(core.slow_correction_logits.grad).all()
