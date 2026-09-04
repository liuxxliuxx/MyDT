import copy

import pytest
import torch
import torch.nn.functional as F

import emotion_ssm.models.conditioned_dualtalk as conditioned_module
from emotion_ssm.models.conditioned_dualtalk import StateFiLM
from emotion_ssm.models.counterfactual import match_counterfactuals
from emotion_ssm.models.dynamics import DyadicEmotionSSM
from emotion_ssm.models.observation import (
    ObservationEncoder,
    build_ema_teacher,
    grad_reverse,
    update_ema,
)
from emotion_ssm.schema import EventObservation


def observation_batch(batch_size=3):
    return {
        "audio": torch.randn(batch_size, 12),
        "face": torch.randn(batch_size, 5),
        "text": torch.randn(batch_size, 10),
        "dataset_id": torch.zeros(batch_size, dtype=torch.long),
        "modality_mask": torch.ones(batch_size, 3, dtype=torch.bool),
    }


def test_modality_subset_does_not_read_masked_values():
    torch.manual_seed(3)
    model = ObservationEncoder(
        audio_dim=12,
        face_dim=5,
        text_dim=10,
        model_dim=16,
        observation_dim=8,
        num_domains=2,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
    ).eval()
    first = observation_batch()
    second = {name: value.clone() for name, value in first.items()}
    second["face"] = torch.randn_like(second["face"]) * 100
    second["text"] = torch.randn_like(second["text"]) * 100
    mask = torch.tensor([[1, 0, 0]], dtype=torch.bool)
    with torch.no_grad():
        a = model(first, mask).aff
        b = model(second, mask).aff
    assert torch.equal(a, b)


def test_reliability_exposes_logits_and_probabilities():
    model = ObservationEncoder(
        audio_dim=12,
        face_dim=5,
        text_dim=10,
        model_dim=16,
        observation_dim=8,
        num_domains=2,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
    ).eval()
    with torch.no_grad():
        output = model(observation_batch(), torch.tensor([[1, 1, 1]]))
    assert output.reliability_logits is not None
    assert torch.allclose(
        output.reliability, output.reliability_logits.sigmoid()
    )
    assert torch.all(output.reliability >= 0)
    assert torch.all(output.reliability <= 1)


def test_ema_has_no_gradient_and_updates():
    student = ObservationEncoder(
        audio_dim=12,
        face_dim=5,
        text_dim=10,
        model_dim=16,
        observation_dim=8,
        num_layers=1,
        num_heads=4,
    )
    teacher = build_ema_teacher(student)
    assert all(not parameter.requires_grad for parameter in teacher.parameters())
    before = next(teacher.parameters()).clone()
    with torch.no_grad():
        next(student.parameters()).add_(1.0)
    update_ema(teacher, student, 0.5)
    assert not torch.equal(before, next(teacher.parameters()))


def test_grl_reverses_gradient():
    value = torch.tensor([1.0, 2.0], requires_grad=True)
    grad_reverse(value, 0.25).sum().backward()
    assert torch.allclose(value.grad, torch.full_like(value, -0.25))


def small_ssm(num_speakers=4):
    return DyadicEmotionSSM(
        state_dim=16,
        observation_dim=8,
        relation_dim=6,
        influence_dim=8,
        influence_channels=4,
        num_speakers=num_speakers,
        num_timescales=4,
        tau_min=0.5,
        tau_max=10.0,
    )


def random_event(batch_size=3):
    return EventObservation(
        aff=torch.randn(batch_size, 8),
        event=torch.randn(batch_size, 8),
        action=torch.randn(batch_size, 8),
        reliability=torch.rand(batch_size, 3),
        modality_mask=torch.ones(batch_size, 3, dtype=torch.bool),
    )


def test_tau_unknown_speaker_and_correction_gate():
    model = small_ssm()
    with torch.no_grad():
        model.personal.baseline_delta.weight[0].fill_(1.0)
    baseline, tau = model.personal(torch.tensor([[0, -1]]))
    assert not torch.allclose(baseline[:, 0], baseline[:, 1])
    assert torch.all(tau >= 0.5) and torch.all(tau <= 10.0)
    state = model.initialize(torch.tensor([[0, -1]]))
    output = model.step(
        state,
        random_event(1),
        torch.tensor([0]),
        torch.tensor([1.0]),
    )
    assert torch.all(output.correction_gate >= 0)
    assert torch.all(output.correction_gate <= 1)
    assert torch.all(output.auxiliary["decay"] > 0)
    assert torch.all(output.auxiliary["decay"] <= 1)


def test_both_coupling_directions_and_rollout_mask():
    model = small_ssm()
    state = model.initialize(torch.tensor([[0, 1], [2, 3]]))
    event = random_event(2)
    no_partner = model.step(
        state, event, torch.tensor([0, 1]), torch.ones(2), enable_partner=False
    )
    partner = model.step(
        state, event, torch.tensor([0, 1]), torch.ones(2), enable_partner=True
    )
    assert partner.influence.shape == (2, 16)
    assert not torch.allclose(no_partner.next_prior.z, partner.next_prior.z)

    events = [random_event(2) for _ in range(3)]
    result = model.rollout(
        state,
        events,
        torch.tensor([[0, 1, 0], [1, 0, 1]]),
        torch.ones(2, 3),
        valid_mask=torch.tensor([[1, 1, 1], [1, 0, 0]], dtype=torch.bool),
    )
    assert torch.equal(result["next_prior_z"][1, 0], result["next_prior_z"][1, 2])


def test_online_counterfactual_matcher_masks_invalid_rows():
    event = F.normalize(torch.randn(5, 8), dim=-1)
    action = F.normalize(torch.randn(5, 8), dim=-1)
    matches = match_counterfactuals(
        event,
        action,
        context_emotion=torch.tensor([1, 1, 1, 2, 1]),
        context_intensity=torch.tensor([0.5, 0.55, 0.52, 0.5, 1.0]),
        turn_position=torch.tensor([0.1, 0.12, 0.11, 0.1, 0.1]),
        dialogue_id=torch.arange(5),
        top_k=3,
        max_action_cosine=1.0,
    )
    assert matches.indices.shape == (5, 3)
    assert not matches.valid[3].any()
    assert not matches.valid[4].any()


def test_counterfactual_matcher_respects_sender_direction():
    event = F.normalize(torch.randn(4, 8), dim=-1)
    action = F.normalize(torch.randn(4, 8), dim=-1)
    matches = match_counterfactuals(
        event,
        action,
        context_emotion=torch.ones(4, dtype=torch.long),
        context_intensity=torch.full((4,), 0.5),
        turn_position=torch.full((4,), 0.5),
        dialogue_id=torch.arange(4),
        sender_role=torch.tensor([0, 0, 1, 1]),
        top_k=3,
        max_action_cosine=2.0,
    )
    for row in range(4):
        selected = matches.indices[row, matches.valid[row]]
        assert selected.numel() == 1
        assert torch.all(torch.tensor([0, 0, 1, 1])[selected] == (row >= 2))


def test_zero_initialized_film_is_identity():
    film = StateFiLM(20, 12)
    features = torch.randn(2, 4, 12)
    state = torch.randn(2, 20)
    assert torch.equal(film(features, state), features)


def test_conditioned_wrapper_matches_baseline_when_film_is_disabled(monkeypatch):
    class Joint(torch.nn.Module):
        def __init__(self, feature_dim):
            super().__init__()
            self.audio = torch.nn.Linear(4, feature_dim)
            self.blend = torch.nn.Linear(3, feature_dim)

        def forward(self, first, second, blendshape):
            return self.audio(first), self.audio(second), self.blend(blendshape)

    class Temporal(torch.nn.Module):
        def forward(self, audio, blendshape):
            return audio + blendshape

    class Interaction(torch.nn.Module):
        def forward(self, first, temporal):
            return torch.cat([first, temporal], dim=-1)

    class DummyBaseline(torch.nn.Module):
        def __init__(self, args):
            super().__init__()
            self.joint_encoder = Joint(args.feature_dim)
            self.temporal_enhancer = Temporal()
            self.interaction_module = Interaction()
            self.synthesis_module = torch.nn.Linear(args.feature_dim * 2, args.blendshape_dim)

        def forward(self, first, second, blendshape):
            a, b, face = self.joint_encoder(first, second, blendshape)
            return self.synthesis_module(
                self.interaction_module(a, self.temporal_enhancer(b, face))
            )

    monkeypatch.setattr(conditioned_module, "DualTalkModel", DummyBaseline)
    baseline = DummyBaseline(type("Args", (), {"feature_dim": 5, "blendshape_dim": 3})())
    conditioned = conditioned_module.EmotionConditionedDualTalk(
        blendshape_dim=3, feature_dim=5, state_dim=4, relation_dim=2
    )
    conditioned.load_baseline_state_dict(baseline.state_dict())
    first = torch.randn(2, 6, 4)
    second = torch.randn(2, 6, 4)
    blendshape = torch.randn(2, 6, 3)
    context = torch.randn(2, 10)
    baseline.eval()
    conditioned.eval()
    with torch.no_grad():
        expected = baseline(first, second, blendshape)
        actual = conditioned(first, second, blendshape, context, enable_film=False)
    assert (expected - actual).abs().max().item() < 1e-6
