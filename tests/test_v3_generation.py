import copy
import io
from dataclasses import replace

import pytest
import torch
from torch import nn

from emotion_ssm.data.packets_v3 import empty_role_features, collate_role_features
from emotion_ssm.models.state_core import UnifiedEmotionStateCore
from emotion_ssm.models.streaming_v3 import StreamingAvatarV3, StreamSessionsV3
from emotion_ssm.models.token_observer import TokenObserver
from emotion_ssm.train.generation_v3 import configure_generation_stage, reconstruction_numerators, train_segment, SegmentCursor, JointAuxiliary, AMPGradientOverflow, rebind_training_baseline


class TinyGenerator(nn.Module):
    def __init__(self, context_dim):
        super().__init__()
        self.context_dim = context_dim
        self.baseline = nn.Linear(56, 56)
        self.film = nn.Linear(context_dim, 56)

    def forward(self, audio, partner_audio, visual, context, enable_film=True):
        value = self.baseline(visual)
        if enable_film:
            value = value + self.film(context)
        return value


def small_model(variant="dyadic"):
    observer = TokenObserver(dict(audio_dim=8, text_dim=8, model_dim=16, affect_dim=8,
                                  num_layers=1, num_heads=2, dropout=0.))
    core = UnifiedEmotionStateCore(observation_dim=8, relation_dim=4, hidden_dim=16)
    return StreamingAvatarV3(TinyGenerator(core.context_dim), observer, core, variant=variant)


def packet(t=1., session="dialogue", event=False):
    # Seed is local to fixture generation and does not disturb model RNG.
    generator = torch.Generator().manual_seed(round(t * 31))
    pair = []
    for index in range(2):
        f = empty_role_features(audio_dim=8, text_dim=8, now=t, domain_id=2)
        f["audio_tokens"] = torch.randn(4, 8, generator=generator)
        f["audio_mask"] = torch.ones(4, dtype=torch.bool)
        f["audio_times"] = torch.linspace(t-.75, t, 4)
        f["text_tokens"] = torch.randn(2, 8, generator=generator)
        f["text_mask"] = torch.ones(2, dtype=torch.bool)
        f["text_times"] = torch.full((2,), t)
        f["text_roles"] = torch.tensor([index, 1-index])
        f["text_fresh_mask"] = torch.tensor([event, event])
        f["modality_mask"] = torch.tensor([True, False, True])
        f["fresh_observation"] = torch.tensor([True, False, event])
        f["event_present"] = torch.tensor(event)
        f["action_present"] = torch.tensor(True)
        f["action_duration"] = torch.tensor(1.)
        pair.append(collate_role_features([f]))
    return {"session_id": session, "roles": ("A", "B"), "time": t,
            "target_audio": torch.randn(1, 16000, generator=generator),
            "partner_audio": torch.randn(1, 16000, generator=generator),
            "target_speech_active": True, "partner_speech_active": True,
            "partner_blendshape": torch.randn(1, 25, 56, generator=generator),
            "target_features": pair[0], "partner_features": pair[1]}


def test_joint_last_loss_reaches_history_observer_and_state():
    torch.manual_seed(4)
    model = small_model()
    first = packet(event=True)
    first["target_features"]["audio_tokens"].requires_grad_(True)
    state = None
    for time in range(1, 7):
        output, state, _ = model(first if time == 1 else packet(time), state)
    output.square().mean().backward()
    # The first packet is beyond the generator's three-second window, so this
    # gradient must flow through persistent state, not direct FiLM history.
    assert first["target_features"]["audio_tokens"].grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.state_model.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.observer.parameters())
    assert state.emotion.fast.grad_fn is not None
    assert state.detach().emotion.fast.grad_fn is None


def test_current_target_flame_never_enters_prediction():
    model = small_model().eval()
    one, two = packet(), packet()
    one["target_blendshape"] = torch.randn(1, 25, 56)
    two["target_blendshape"] = torch.randn(1, 25, 56) * 100
    for item in (one, two):
        f = item["target_features"]
        f["flame_tokens"] = item["target_blendshape"]
        f["flame_mask"] = torch.ones(1, 25, dtype=torch.bool)
        f["flame_times"] = torch.ones(1, 25)
        f["modality_mask"][:, 1] = True
    a, _, da = model(one)
    b, _, db = model(two)
    torch.testing.assert_close(a, b)
    torch.testing.assert_close(da["target_aff"], db["target_aff"])
    assert not da["target_modalities"][0, 1]


def test_history_context_does_not_repeat_action_or_event():
    model = small_model().eval()
    first = packet(event=True)
    first["target_features"]["event_ids"] = ["sentence-1"]
    _, state, _ = model(first)
    second = packet(2., event=True)
    second["target_features"]["event_ids"] = ["sentence-1"]
    for prefix in ("target", "partner"):
        second[prefix + "_speech_active"] = False
        second[prefix + "_features"]["fresh_observation"].zero_()
    second["partner_visual_mask"] = torch.zeros(1, 25, dtype=torch.bool)
    _, _, d = model(second, state)
    own = d["observations"][0]
    assert not own.event_present.any()
    assert own.event.abs().sum() == 0
    assert own.action_duration.item() == 0
    assert not own.action_present.any()
    assert not own.fresh_observation.any()
    assert d["target_modalities"][0, 2]  # Old context remains readable.


def test_stream_prefix_causal_and_segmented_execution_equal():
    model = small_model().eval()
    state, first_outputs = None, []
    for time in range(1, 6):
        output, state, _ = model(packet(time), state)
        first_outputs.append(output.detach())
    state = None
    for time in range(1, 6):
        output, state, _ = model(packet(time), state)
        if time == 2:
            state = state.detach()
        torch.testing.assert_close(output, first_outputs[time-1])
        assert output.shape == (1, 25, 56)
        assert state.history[0]["context"].shape[1] <= 75
    # Appending radically different future inputs cannot mutate a previous result.
    saved = first_outputs[0].clone()
    future = packet(6.)
    future["partner_blendshape"] *= 500
    model(future, state)
    torch.testing.assert_close(first_outputs[0], saved)


def test_future_tokens_rejected_and_late_text_does_not_rewrite_state():
    model = small_model().eval()
    bad = packet()
    bad["target_features"]["text_times"][0, 0] = 2.
    with pytest.raises(ValueError, match="future"):
        model(bad)
    first = packet()
    _, previous, _ = model(first)
    second = packet(2.)
    second["words"] = [{"id": "late", "role": "A", "text": "hello", "start": .1, "end": .2, "available_at": .4}]
    _, current, _ = model(second, previous)
    assert "late" not in previous.words
    assert current.words["late"]["available_at"] == 2.


def test_sessions_roles_and_reset_are_explicit():
    model = small_model().eval()
    sessions = StreamSessionsV3(model)
    sessions(packet(session="one"))
    sessions(packet(session="two"))
    assert len(sessions.states) == 2
    swapped = packet(2., session="one")
    swapped["roles"] = ("B", "A")
    with pytest.raises(ValueError, match="reset"):
        sessions(swapped)
    sessions.reset("one")
    sessions(packet(session="one"))


def test_film_off_equals_same_backbone_and_equal_train_permissions():
    conditioned = small_model("dyadic").eval()
    none = copy.deepcopy(conditioned)
    none.variant = "none"
    p = packet()
    out, _, _ = none(p)
    expected = none.generator.baseline(p["partner_blendshape"])
    torch.testing.assert_close(out, expected)
    configure_generation_stage(conditioned)
    configure_generation_stage(none)
    assert all(p.requires_grad for p in conditioned.generator.baseline.parameters())
    assert all(p.requires_grad for p in none.generator.baseline.parameters())
    assert not any(p.requires_grad for p in none.generator.film.parameters())


def test_tbptt_single_backward_per_segment_and_checkpointed_generator():
    torch.manual_seed(3)
    model = small_model()
    configure_generation_stage(model, gradient_checkpointing=True)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3, foreach=False)
    packets = [(packet(time, event=time == 1), torch.zeros(1, 25, 56), torch.ones(1, 25, dtype=torch.bool))
               for time in range(1, 7)]
    before = model.state_model.fast_log_rate.detach().clone() if hasattr(model.state_model, "fast_log_rate") else None
    state, metrics = train_segment(model, packets, optimizer, tbptt_steps=3)
    assert metrics["global_valid_blocks"] == 6
    assert metrics["valid_frames"] == 150
    assert metrics["tbptt_segments_rank"] == 2
    assert state.emotion.fast.grad_fn is None
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.state_model.parameters())


def test_reconstruction_masked_nan_and_exact_sum():
    prediction = torch.randn(1, 25, 56, requires_grad=True)
    target = torch.randn(1, 25, 56)
    mask = torch.ones(1, 25, dtype=torch.bool)
    mask[:, -3:] = False
    target[:, -3:] = float("nan")
    losses = reconstruction_numerators(prediction, target, mask)
    sum(losses.values()).backward()
    assert torch.isfinite(prediction.grad).all()
    assert prediction.grad[:, -3:].abs().sum() == 0


def test_rank_cursor_no_padding_and_resume_exact():
    class Dataset:
        def __len__(self):
            return 5
        def packets(self, index):
            for t in range(1, 3):
                yield packet(t, session=str(index)), torch.zeros(1, 25, 56), torch.ones(1, 25, dtype=torch.bool)
    data = Dataset()
    a, b = SegmentCursor(data, rank=0, world_size=2), SegmentCursor(data, rank=1, world_size=2)
    names_a = {a.next()[0]["session_id"] for _ in range(6)}
    names_b = {b.next()[0]["session_id"] for _ in range(4)}
    assert names_a.isdisjoint(names_b)
    resume = SegmentCursor(data, rank=0, world_size=2, seen=a.seen)
    assert resume.next()[0]["session_id"] == a.next()[0]["session_id"]


def test_future_labels_train_forecast_without_entering_live_stream():
    model = small_model().eval()
    model.teacher = copy.deepcopy(model.observer).requires_grad_(False).eval()
    configure_generation_stage(model, frozen_teacher=model.teacher, gradient_checkpointing=True)
    settings = {"forecast_seconds": [1, 2, 4, 8], "coordinate_weight": .1, "future_weight": 1.}
    auxiliary = JointAuxiliary(model, settings)
    packets = [(packet(time), torch.zeros(1, 25, 56), torch.ones(1, 25, dtype=torch.bool)) for time in range(1, 11)]
    auxiliary.prepare(packets, "cpu")
    old_target = auxiliary.targets[("dialogue", 9.)][0].clone()
    changed = copy.deepcopy(packets)
    changed[8][0]["target_features"]["audio_tokens"] *= -20
    auxiliary.prepare(changed, "cpu")
    assert not torch.equal(old_target, auxiliary.targets[("dialogue", 9.)][0])
    # Only the first two input packets are consumed; labels extend to eight seconds.
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3, foreach=False)
    state, result = train_segment(model, packets[:2], optimizer, tbptt_steps=4, auxiliary_loss=auxiliary)
    assert state.time == 2.
    assert result["auxiliary_loss"] > 0
    assert result["state_grad_norm"] > 0
    assert all(p.grad is None for p in model.teacher.parameters())


def test_partial_action_duration_and_half_token_cache_are_preserved():
    model = small_model().eval()
    item = packet()
    for prefix in ("target", "partner"):
        values = item[prefix + "_features"]
        values["audio_tokens"] = values["audio_tokens"].half()
        values["text_tokens"] = values["text_tokens"].half()
        values["action_duration"] = torch.tensor([.12])
    item["partner_visual_mask"] = torch.zeros(1, 25, dtype=torch.bool)
    _, _, diagnostics = model(item)
    assert diagnostics["observations"][0].action_duration.item() == pytest.approx(.12)
    assert diagnostics["observations"][1].action_duration.item() == pytest.approx(.12)


def test_checkpoint_recomputation_replays_numpy_masks_without_advancing_rng():
    import numpy as np

    class NumpyGenerator(TinyGenerator):
        def forward(self, audio, partner_audio, visual, context, enable_film=True):
            mask = torch.as_tensor(np.random.uniform(size=visual.shape) > .3, device=visual.device)
            return super().forward(audio, partner_audio, visual * mask, context, enable_film)

    ordinary = small_model()
    ordinary.generator = NumpyGenerator(ordinary.state_model.context_dim)
    checkpointed = copy.deepcopy(ordinary)
    checkpointed.generator_checkpointing = True
    results = []
    for model in (ordinary, checkpointed):
        torch.manual_seed(77)
        np.random.seed(77)
        state, outputs = None, []
        for time in range(1, 4):
            generated, state, _ = model(packet(time), state)
            outputs.append(generated)
        loss = sum(value.square().mean() for value in outputs)
        loss.backward()
        results.append((loss.detach(), {name: p.grad.clone() for name,p in model.named_parameters() if p.grad is not None}, np.random.rand()))
    torch.testing.assert_close(results[0][0], results[1][0])
    assert results[0][2] == results[1][2]
    for name, gradient in results[0][1].items():
        torch.testing.assert_close(gradient, results[1][1][name], rtol=1e-5, atol=1e-6)


def test_amp_overflow_backs_off_without_consuming_optimizer_update():
    model = small_model()
    parameters = configure_generation_stage(model)
    optimizer = torch.optim.AdamW(parameters, lr=1e-3, foreach=False)
    scaler = torch.amp.GradScaler("cpu", enabled=True, init_scale=128.)
    packets = [(packet(time), torch.zeros(1,25,56), torch.ones(1,25,dtype=torch.bool)) for time in (1,2)]
    before = [p.detach().clone() for p in parameters]
    handle = parameters[0].register_hook(lambda gradient: gradient * float("inf"))
    with pytest.raises(AMPGradientOverflow, match="optimizer unchanged"):
        train_segment(model, packets, optimizer, scaler=scaler, tbptt_steps=4)
    handle.remove()
    assert scaler.get_scale() == 64.
    assert not optimizer.state
    for old, parameter in zip(before, parameters):
        torch.testing.assert_close(old, parameter, rtol=0, atol=0)
    _, metrics = train_segment(model, packets, optimizer, scaler=scaler, tbptt_steps=4)
    assert metrics["global_valid_blocks"] == 2
    assert optimizer.state


def test_real_huggingface_generator_two_optimizer_steps_with_nested_checkpoint():
    from transformers import Wav2Vec2Config
    from emotion_ssm.config_v3 import default_config
    from emotion_ssm.utils.checkpoint_v3 import build_avatar
    config = default_config()
    config["observer"].update(audio_dim=8,text_dim=8,model_dim=8,affect_dim=4,num_layers=1,num_heads=2,dropout=0.)
    config["generation"]["feature_dim"] = 8
    config["state"].update(observation_dim=4,relation_dim=2,hidden_dim=8)
    audio = Wav2Vec2Config(hidden_size=8,num_hidden_layers=1,num_attention_heads=2,intermediate_size=16,
        conv_dim=(8,8,8),conv_kernel=(10,3,3),conv_stride=(5,2,2),num_conv_pos_embedding_groups=2,
        num_conv_pos_embeddings=8,mask_time_prob=.2,mask_time_length=2,mask_feature_prob=0.,
        hidden_dropout=.1,attention_dropout=.1,feat_proj_dropout=.1)
    construction = {"observer":config["observer"],"state":UnifiedEmotionStateCore(**config["state"]).get_config(),
                    "generator_audio":audio.to_dict(),"features":None}
    model = build_avatar(config,construction=construction,initialize=False)
    params = configure_generation_stage(model,frozen_teacher=model.teacher,gradient_checkpointing=True)
    optimizer = torch.optim.AdamW(params,lr=1e-3,foreach=False)
    state = None
    for step in range(2):
        packets = [(packet(time),torch.zeros(1,25,56),torch.ones(1,25,dtype=torch.bool)) for time in range(2*step+1,2*step+3)]
        state, result = train_segment(model,packets,optimizer,state=state,tbptt_steps=4)
        assert result["generator_grad_norm"] > 0
        assert model.state_model.baseline.grad is not None
        # FiLM starts at zero, so its first update opens the state gradient path.
        if step:
            assert model.state_model.baseline.grad.abs().sum() > 0
    assert result["state_grad_norm"] > 0
    assert result["observer_grad_norm"] > 0
    for name in ("audio_encoder1","audio_encoder2"):
        encoder = getattr(model.generator.baseline.joint_encoder,name)
        assert encoder.encoder.gradient_checkpointing
        assert all(not p.requires_grad and p.grad is None for p in encoder.feature_extractor.parameters())
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in encoder.encoder.parameters())


@pytest.mark.parametrize("variant", ["none", "affect", "self", "dyadic"])
def test_control_specific_permissions_auxiliary_and_no_unused_state_training(variant, monkeypatch):
    model = small_model(variant)
    model.teacher = copy.deepcopy(model.observer).requires_grad_(False).eval()
    parameters = configure_generation_stage(model, frozen_teacher=model.teacher)
    assert any(p.requires_grad for p in model.observer.parameters()) == (variant != "none")
    assert any(p.requires_grad for p in model.state_model.parameters()) == (variant in ("self", "dyadic"))
    assert all(p.requires_grad for p in model.generator.baseline.parameters())
    if variant in ("none", "affect"):
        monkeypatch.setattr(model.state_model, "advance", lambda *a, **kw: pytest.fail("Unused state core was advanced"))
        monkeypatch.setattr(model.state_model, "forecast", lambda *a, **kw: pytest.fail("Unused future task was trained"))
        monkeypatch.setattr(model.state_model, "rebind_trainable_baseline", lambda *a, **kw: pytest.fail("Unused baseline was rebound"))
    if variant == "none":
        monkeypatch.setattr(model.teacher, "forward", lambda *a, **kw: pytest.fail("None should not evaluate an auxiliary teacher"))
    settings = {"forecast_seconds": [1,2], "coordinate_weight": .1, "future_weight": 1., "masked_weight": .1}
    auxiliary = JointAuxiliary(model, settings)
    packets = [(packet(t), torch.zeros(1,25,56), torch.ones(1,25,dtype=torch.bool)) for t in (1,2,3)]
    auxiliary.prepare(packets, "cpu", input_count=2)
    optimizer = torch.optim.AdamW(parameters, lr=1e-3, foreach=False)
    _, metrics = train_segment(model, packets[:2], optimizer, auxiliary_loss=auxiliary, tbptt_steps=4)
    assert metrics["generator_grad_norm"] > 0
    assert (metrics["observer_grad_norm"] > 0) == (variant != "none")
    assert (metrics["state_grad_norm"] > 0) == (variant in ("self", "dyadic"))
    assert (metrics["future_count"] > 0) == (variant in ("self", "dyadic"))
    assert (metrics["coordinate_count"] > 0) == (variant != "none")
    assert (metrics["masked_count"] > 0) == (variant != "none")
    if variant == "none":
        assert metrics["auxiliary_loss"] == 0


def test_large_state_gradient_cannot_rescale_generator_gradient_clip():
    first = small_model()
    second = copy.deepcopy(first)
    packets = [(packet(t), torch.zeros(1,25,56), torch.ones(1,25,dtype=torch.bool)) for t in (1,2)]
    results = []
    for index, model in enumerate((first, second)):
        parameters = configure_generation_stage(model)
        optimizer = torch.optim.SGD(parameters, lr=.01)
        auxiliary = (lambda records: 1e7 * model.state_model.rotation.square().sum()) if index else None
        _, metrics = train_segment(model, packets, optimizer, auxiliary_loss=auxiliary, tbptt_steps=4, grad_clip=.5)
        results.append((metrics, {name: p.grad.clone() for name,p in model.generator.named_parameters()}))
    assert results[1][0]["state_grad_norm_before_clip"] > 1e5
    assert results[1][0]["state_grad_norm"] <= .500001
    assert results[1][0]["component_grad_norm_stage"] == "after_independent_component_clip"
    for name, value in results[0][1].items():
        torch.testing.assert_close(value, results[1][1][name], rtol=0, atol=0)
    for first_parameter, second_parameter in zip(first.generator.parameters(), second.generator.parameters()):
        torch.testing.assert_close(first_parameter, second_parameter, rtol=0, atol=0)


@pytest.mark.parametrize("variant", ["self", "dyadic"])
def test_baseline_gradient_reaches_every_tbptt_segment_and_optimizer_step(variant):
    torch.manual_seed(53)
    model = small_model(variant)
    parameters = configure_generation_stage(model)
    optimizer = torch.optim.AdamW(parameters, lr=1e-3, foreach=False)
    baseline_gradients = []
    handle = model.state_model.baseline.register_hook(lambda grad: baseline_gradients.append(grad.detach().clone()))
    state = None
    for start in (1, 5):
        examples = [(packet(t, event=t == 1), torch.zeros(1,25,56), torch.ones(1,25,dtype=torch.bool))
                    for t in range(start, start+4)]
        state, metrics = train_segment(model, examples, optimizer, state=state, tbptt_steps=2)
        assert not state.emotion.baseline.requires_grad
        assert metrics["tbptt_segments_rank"] == 2
    handle.remove()
    assert len(baseline_gradients) == 4
    assert all(torch.isfinite(grad).all() and grad.abs().sum() > 0 for grad in baseline_gradients)


def test_training_baseline_rebind_preserves_detached_history_without_mutating_input():
    model = small_model()
    _, state, _ = model(packet())
    state = state.detach()
    custom = state.emotion.baseline.clone().fill_(.7)
    state = replace(state, emotion=replace(state.emotion, baseline=custom))
    rebound = rebind_training_baseline(model, state)
    assert rebound is not state and rebound.emotion is not state.emotion
    assert rebound.emotion.baseline.requires_grad
    torch.testing.assert_close(rebound.emotion.baseline, model.state_model.baseline[None,None].expand_as(custom))
    assert torch.equal(state.emotion.baseline, custom)
    for name in ("fast", "slow", "relation", "elapsed", "last_event_ids"):
        assert getattr(rebound.emotion, name) is getattr(state.emotion, name)
        assert not getattr(rebound.emotion, name).requires_grad


@pytest.mark.parametrize("train_mode", [False, True])
def test_streaming_custom_baseline_is_not_rebound_by_forward(train_mode, monkeypatch):
    model = small_model().train(train_mode)
    _, state, _ = model(packet())
    custom = state.emotion.baseline.detach().clone().fill_(.7)
    state = replace(state.detach(), emotion=replace(state.emotion.detach(), baseline=custom))
    monkeypatch.setattr(model.state_model, "rebind_trainable_baseline",
                        lambda *a, **kw: pytest.fail("Forward must not adopt training baseline semantics"))
    _, result, _ = model(packet(2), state)
    torch.testing.assert_close(result.emotion.baseline, custom, rtol=0, atol=0)


def test_frozen_state_training_keeps_custom_baseline_and_has_no_baseline_gradient(monkeypatch):
    model = small_model()
    _, state, _ = model(packet())
    custom = state.emotion.baseline.detach().clone().fill_(.7)
    state = replace(state.detach(), emotion=replace(state.emotion.detach(), baseline=custom))
    parameters = configure_generation_stage(model, train_state=False)
    optimizer = torch.optim.AdamW(parameters, lr=1e-3, foreach=False)
    monkeypatch.setattr(model.state_model, "rebind_trainable_baseline",
                        lambda *a, **kw: pytest.fail("A frozen baseline must not be rebound"))
    examples = [(packet(t), torch.zeros(1,25,56), torch.ones(1,25,dtype=torch.bool)) for t in (2,3,4,5)]
    result, _ = train_segment(model, examples, optimizer, state=state, tbptt_steps=2)
    torch.testing.assert_close(result.emotion.baseline, custom, rtol=0, atol=0)
    assert model.state_model.baseline.grad is None


@pytest.mark.parametrize("variant", ["self", "dyadic"])
def test_baseline_resume_matches_uninterrupted_optimizer_update(variant):
    from emotion_ssm.utils.checkpoint import capture_rng_state, restore_rng_state
    torch.manual_seed(71)
    model = small_model(variant)
    parameters = configure_generation_stage(model)
    optimizer = torch.optim.AdamW(parameters, lr=1e-3, foreach=False)
    first = [(packet(t), torch.zeros(1,25,56), torch.ones(1,25,dtype=torch.bool)) for t in (1,2,3,4)]
    state, _ = train_segment(model, first, optimizer, tbptt_steps=2)
    buffer = io.BytesIO()
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "state": state, "rng": capture_rng_state()}, buffer)
    second = [(packet(t), torch.zeros(1,25,56), torch.ones(1,25,dtype=torch.bool)) for t in (5,6,7,8)]
    continuous_state, continuous_metrics = train_segment(model, second, optimizer, state=state, tbptt_steps=2)
    buffer.seek(0)
    payload = torch.load(buffer, weights_only=False)
    resumed = small_model(variant)
    resumed_parameters = configure_generation_stage(resumed)
    resumed.load_state_dict(payload["model"])
    resumed_optimizer = torch.optim.AdamW(resumed_parameters, lr=1e-3, foreach=False)
    resumed_optimizer.load_state_dict(payload["optimizer"])
    # Deserialized memory no longer shares the newly constructed parameter's storage.
    assert payload["state"].emotion.baseline.data_ptr() != resumed.state_model.baseline.data_ptr()
    restore_rng_state(payload["rng"])
    resumed_state, resumed_metrics = train_segment(resumed, second, resumed_optimizer, state=payload["state"], tbptt_steps=2)
    assert continuous_metrics == resumed_metrics
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, resumed.state_dict()[name], rtol=0, atol=0)
    for name in ("fast", "slow", "relation", "baseline", "elapsed", "last_event_ids"):
        torch.testing.assert_close(getattr(continuous_state.emotion,name), getattr(resumed_state.emotion,name), rtol=0, atol=0)
    for index, entry in optimizer.state_dict()["state"].items():
        for name, value in entry.items():
            torch.testing.assert_close(value, resumed_optimizer.state_dict()["state"][index][name], rtol=0, atol=0)
    model.eval(), resumed.eval()
    with torch.no_grad():
        continuous_output, _, _ = model(packet(9), continuous_state)
        resumed_output, _, _ = resumed(packet(9), resumed_state)
    torch.testing.assert_close(continuous_output, resumed_output, rtol=0, atol=0)


@pytest.mark.parametrize("old_revision", [None, "v3.1-original-gradient"])
def test_generation_rejects_old_optimizer_revision_before_distributed_init(old_revision, monkeypatch):
    from emotion_ssm.config_v3 import default_config
    from emotion_ssm.train.generation_v3 import run
    from emotion_ssm.utils import checkpoint_v3, distributed as distributed_utils
    config = default_config()
    config["paths"]["resume"] = "virtual.pt"
    payload = {"kind": "streaming_avatar_v3", "config": copy.deepcopy(config)}
    payload["config"]["train"]["generation_revision"] = old_revision
    monkeypatch.setattr(checkpoint_v3, "read_checkpoint", lambda path: payload)
    monkeypatch.setattr(distributed_utils, "init_distributed",
                        lambda *a, **kw: pytest.fail("Legacy resume must fail before distributed initialization"))
    with pytest.raises(ValueError, match="Training revision changed"):
        run(config)
