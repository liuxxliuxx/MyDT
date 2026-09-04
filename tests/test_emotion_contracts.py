"""Small deterministic contracts for the emotion observation/state pipeline."""

import torch
import torch.nn.functional as F

from emotion_ssm.evaluate import linear_leakage_probe
from emotion_ssm.models.conditioned_dualtalk import DyadicAudioConditioner
from emotion_ssm.models.dynamics import DyadicEmotionSSM
from emotion_ssm.models.observation import (
    ObservationEncoder,
    ObservationSupervisionHeads,
    TemporalAUEncoder,
)
from emotion_ssm.train.dualtalk import ConditionedTrainingSystem
from emotion_ssm.train.dynamics_core import _flatten_event_windows
from emotion_ssm.data.common import pack_face_sequence
from emotion_ssm.schema import DyadicState, EventObservation


def _event(batch=2, dim=8):
    return EventObservation(
        aff=torch.randn(batch, dim),
        event=torch.randn(batch, dim),
        action=torch.randn(batch, dim),
        reliability=torch.ones(batch, 3),
        modality_mask=torch.ones(batch, 3, dtype=torch.bool),
    )


def _encoder():
    return ObservationEncoder(
        audio_dim=4,
        face_dim=5,
        text_dim=6,
        model_dim=16,
        observation_dim=8,
        num_domains=3,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
    ).eval()


def test_au_frame_order_changes_temporal_output():
    encoder = TemporalAUEncoder(5, 16, 0.0).eval()
    value = torch.randn(1, 5, 5)
    mask = torch.ones(1, 5, dtype=torch.bool)
    with torch.no_grad():
        first = encoder(value, mask)
        shuffled = encoder(value.flip(1), mask)
    assert not torch.allclose(first, shuffled)


def test_long_au_sequence_is_sampled_across_full_duration():
    values = torch.arange(10, dtype=torch.float32)[:, None].expand(10, 35)
    packed, confidence, mask = pack_face_sequence(
        values, torch.ones(10), torch.ones(10, dtype=torch.bool), max_frames=4
    )
    assert packed[:, 0].tolist() == [0.0, 3.0, 6.0, 9.0]
    assert confidence.tolist() == [1.0, 1.0, 1.0, 1.0]
    assert mask.all()


def test_au_padding_values_are_ignored():
    model = _encoder()
    batch = {
        "audio": torch.zeros(1, 4),
        "face": torch.randn(1, 5, 5),
        "text": torch.zeros(1, 6),
        "dataset_id": torch.zeros(1, dtype=torch.long),
        "modality_mask": torch.tensor([[0, 1, 0]], dtype=torch.bool),
        "face_frame_mask": torch.tensor([[1, 1, 0, 0, 0]], dtype=torch.bool),
        "face_confidence": torch.ones(1, 5),
    }
    changed = {name: value.clone() for name, value in batch.items()}
    changed["face"][0, 2:] = torch.randn(3, 5) * 1000
    with torch.no_grad():
        first = model(batch, torch.tensor([[0, 1, 0]])).aff
        second = model(changed, torch.tensor([[0, 1, 0]])).aff
    assert torch.equal(first, second)


def test_all_au_padding_is_finite_and_zero():
    encoder = TemporalAUEncoder(5, 16, 0.0).eval()
    face = torch.randn(2, 4, 5)
    mask = torch.zeros(2, 4, dtype=torch.bool)
    confidence = torch.zeros(2, 4)
    with torch.no_grad():
        output = encoder(face, mask, confidence)
    assert torch.isfinite(output).all()
    assert torch.equal(output, torch.zeros_like(output))


def test_same_utterance_modalities_are_more_similar_than_random():
    # Use deterministic adapters so this contract tests the architecture rather
    # than relying on an untrained model to discover cross-modal alignment.
    model = _encoder()
    class ScalarAdapter(torch.nn.Module):
        def forward(self, value, domain):
            return value[:, :1].expand(-1, 16)

    class FaceAdapter(torch.nn.Module):
        def forward(self, value, frame_mask=None, confidence=None):
            return value[:, :1, :1].squeeze(1).expand(-1, 16)

    model.audio_adapter = ScalarAdapter()
    model.text_adapter = ScalarAdapter()
    model.face_temporal = FaceAdapter()
    model.face_adapter = ScalarAdapter()
    common = {
        "audio": torch.ones(1, 4),
        "face": torch.ones(1, 3, 5),
        "text": torch.ones(1, 6),
        "dataset_id": torch.zeros(1, dtype=torch.long),
        "modality_mask": torch.ones(1, 3, dtype=torch.bool),
        "face_frame_mask": torch.ones(1, 3, dtype=torch.bool),
        "face_confidence": torch.ones(1, 3),
    }
    other = {name: value.clone() for name, value in common.items()}
    other["audio"] = torch.full_like(other["audio"], -1.0)
    other["face"] = torch.full_like(other["face"], -1.0)
    other["text"] = torch.full_like(other["text"], -1.0)
    with torch.no_grad():
        same = model(common, torch.tensor([[1, 1, 1]])).modality_aff[0]
        random = model(other, torch.tensor([[1, 1, 1]])).modality_aff[0]
    same_cosine = F.cosine_similarity(same[0], same[1], dim=0)
    random_cosine = F.cosine_similarity(same[0], random[0], dim=0)
    assert same.shape == (3, 8)
    assert same_cosine > random_cosine


def test_latent_dimensions_have_nonzero_variance_and_heads_work_for_avt_and_single():
    model = _encoder()
    batch = {
        "audio": torch.randn(32, 4),
        "face": torch.randn(32, 2, 5),
        "text": torch.randn(32, 6),
        "dataset_id": torch.zeros(32, dtype=torch.long),
        "modality_mask": torch.ones(32, 3, dtype=torch.bool),
        "face_frame_mask": torch.ones(32, 2, dtype=torch.bool),
        "face_confidence": torch.ones(32, 2),
    }
    output = model(batch)
    output.aff.retain_grad()
    heads = ObservationSupervisionHeads(8, 2, 3)
    predictions = heads(output.aff, 0.0)
    assert torch.all(output.aff[:, -1].var(0) > 1e-5)
    assert predictions["emotion"].shape[:2] == (32, 7)
    assert predictions["vad"].shape[:2] == (32, 7)
    assert torch.isfinite(predictions["emotion"]).all()
    assert torch.isfinite(predictions["vad"]).all()
    # Both the A-only and AVT paths are connected to the emotion/VAD heads.
    loss = predictions["emotion"][:, [0, -1]].square().mean()
    loss = loss + predictions["vad"][:, [0, -1]].square().mean()
    loss.backward()
    assert output.aff.grad is not None
    assert output.aff.grad[:, [0, -1]].abs().sum() > 0


def test_open_loop_does_not_read_future_affect():
    model = DyadicEmotionSSM(
        state_dim=8, observation_dim=8, relation_dim=4, influence_dim=8,
        influence_channels=3, num_speakers=1, num_timescales=2,
    ).eval()
    state = model.initialize(torch.full((1, 2), -1, dtype=torch.long))
    events = [_event(1, 8) for _ in range(4)]
    changed = [item for item in events]
    changed[2] = EventObservation(
        aff=events[2].aff + 1000,
        event=events[2].event,
        action=events[2].action,
        reliability=events[2].reliability,
        modality_mask=events[2].modality_mask,
    )
    roles = torch.tensor([[0, 1, 0, 1]])
    dt = torch.ones(1, 4)
    with torch.no_grad():
        first = model.rollout(state, events, roles, dt, correction_mode="none")
        second = model.rollout(state, changed, roles, dt, correction_mode="none")
    assert torch.equal(first["next_prior_z"], second["next_prior_z"])


def test_dt_accepts_scalar_and_column_shapes():
    model = DyadicEmotionSSM(
        state_dim=8, observation_dim=8, relation_dim=4, influence_dim=8,
        influence_channels=3, num_speakers=1, num_timescales=2,
    ).eval()
    state = model.initialize(torch.full((2, 2), -1, dtype=torch.long))
    event = _event(2, 8)
    role = torch.zeros(2, dtype=torch.long)
    with torch.no_grad():
        scalar = model.transition(state, event, role, torch.tensor(1.0))
        column = model.transition(state, event, role, torch.ones(2, 1))
        vector = model.transition(state, event, role, torch.ones(2))
    assert torch.allclose(scalar.next_prior.z, vector.next_prior.z)
    assert torch.allclose(column.next_prior.z, vector.next_prior.z)


def test_rollout_padding_does_not_change_posterior_or_relation():
    model = DyadicEmotionSSM(
        state_dim=8, observation_dim=8, relation_dim=4, influence_dim=8,
        influence_channels=3, num_speakers=1, num_timescales=2,
    ).eval()
    state = model.initialize(torch.full((1, 2), -1, dtype=torch.long))
    events = [_event(1, 8) for _ in range(3)]
    changed = list(events)
    changed[2] = _event(1, 8)
    valid = torch.tensor([[1, 1, 0]], dtype=torch.bool)
    roles = torch.tensor([[0, 1, 0]])
    dt = torch.ones(1, 3)
    with torch.no_grad():
        first = model.rollout(state, events, roles, dt, valid_mask=valid)
        second = model.rollout(state, changed, roles, dt, valid_mask=valid)
    assert torch.equal(first["posterior_z"], second["posterior_z"])
    assert torch.equal(first["posterior_relation"], second["posterior_relation"])


def test_horizon_windows_keep_each_start_chronological():
    values = torch.arange(2 * 4 * 1, dtype=torch.float32).reshape(2, 4, 1)
    event = EventObservation(
        aff=values,
        event=values + 100,
        action=values + 200,
        reliability=torch.ones(2, 4, 3),
        modality_mask=torch.ones(2, 4, 3, dtype=torch.bool),
    )
    flattened = _flatten_event_windows(event, starts=3, horizon=2)
    windows = flattened.aff.reshape(2 * 3, 2, 1).squeeze(-1)
    assert windows.tolist() == [[0, 1], [1, 2], [2, 3], [4, 5], [5, 6], [6, 7]]


def test_sender_action_changes_receiver_more_than_sender():
    model = DyadicEmotionSSM(
        state_dim=8, observation_dim=8, relation_dim=4, influence_dim=8,
        influence_channels=3, num_speakers=1, num_timescales=2,
    ).eval()
    state = model.initialize(torch.full((1, 2), -1, dtype=torch.long))
    base = _event(1, 8)
    altered = EventObservation(
        aff=base.aff, event=base.event, action=base.action + 3.0,
        reliability=base.reliability, modality_mask=base.modality_mask,
    )
    with torch.no_grad():
        first = model.transition(state, base, torch.zeros(1, dtype=torch.long), torch.ones(1))
        second = model.transition(state, altered, torch.zeros(1, dtype=torch.long), torch.ones(1))
    delta = (second.next_prior.z - first.next_prior.z).abs()[0]
    assert delta[1].mean() > delta[0].mean()


class _FakeObserver(torch.nn.Module):
    def forward(self, waveform):
        value = waveform.mean(dim=-1, keepdim=True).expand(-1, 8)
        return EventObservation(
            aff=value, event=value, action=value,
            reliability=torch.ones(len(value), 3),
            modality_mask=torch.ones(len(value), 3, dtype=torch.bool),
        )


def test_conditioner_preserves_explicit_chunk_state():
    model = DyadicEmotionSSM(
        state_dim=8, observation_dim=8, relation_dim=4, influence_dim=8,
        influence_channels=3, num_speakers=1, num_timescales=2,
    )
    conditioner = DyadicAudioConditioner(_FakeObserver(), model)
    target = torch.randn(2, 16)
    partner = torch.randn(2, 16)
    dt = torch.ones(2)
    _, first_state, _ = conditioner(target, partner, dt)
    context_a, next_state, _ = conditioner(target, partner, dt, state=first_state)
    context_b, _, _ = conditioner(target, partner, dt, state=first_state)
    assert torch.equal(context_a, context_b)
    assert torch.equal(next_state.speaker_ids, first_state.speaker_ids)


def test_chunk_rollout_final_state_is_next_chunk_initial_state():
    model = DyadicEmotionSSM(
        state_dim=8, observation_dim=8, relation_dim=4, influence_dim=8,
        influence_channels=3, num_speakers=1, num_timescales=2,
    )
    conditioner = DyadicAudioConditioner(_FakeObserver(), model)
    chunks = [
        {"target_audio": torch.randn(1, 8), "partner_audio": torch.randn(1, 8), "dt": torch.ones(1)},
        {"target_audio": torch.randn(1, 8), "partner_audio": torch.randn(1, 8), "dt": torch.ones(1)},
    ]
    _, states, _ = conditioner.rollout_chunks(chunks)
    _, explicit, _ = conditioner(
        chunks[1]["target_audio"], chunks[1]["partner_audio"], chunks[1]["dt"], state=states[0]
    )
    assert torch.allclose(states[1].z, explicit.z)


class _SequenceRecordingConditioner(torch.nn.Module):
    """Small stateful conditioner used to exercise the training sequence path."""

    def __init__(self):
        super().__init__()
        self.input_states = []
        self.output_states = []

    def forward(self, target, partner, dt, state=None):
        self.input_states.append(None if state is None else state.clone())
        batch = target.shape[0]
        if state is None:
            state = DyadicState(
                z=target.new_zeros(batch, 2, 4),
                relation=target.new_zeros(batch, 2),
                speaker_ids=torch.full((batch, 2), -1, dtype=torch.long, device=target.device),
            )
        next_state = DyadicState(
            z=state.z + 1.0,
            relation=state.relation + 1.0,
            speaker_ids=state.speaker_ids,
        )
        self.output_states.append(next_state.clone())
        context = target.new_zeros(batch, 10)
        evidence = {"target_state_aff": next_state.z[:, 0]}
        return context, next_state, evidence


class _SequenceGenerator(torch.nn.Module):
    def forward(self, target, partner, partner_blendshape, context, enable_film=True):
        return target.mean(dim=-1, keepdim=True).unsqueeze(-1).expand(-1, 3, 56)


class _SequenceProjector(torch.nn.Module):
    def forward(self, blendshape):
        return blendshape.mean(dim=1)[:, :4]


def test_training_sequence_carries_final_state_to_next_chunk():
    conditioner = _SequenceRecordingConditioner()
    system = ConditionedTrainingSystem(
        _SequenceGenerator(),
        conditioner,
        _SequenceProjector(),
        state_loss_weight=0.0,
    )
    batch = {
        "target_audio": torch.randn(1, 2, 12),
        "partner_audio": torch.randn(1, 2, 12),
        "target_blendshape": torch.randn(1, 2, 3, 56),
        "partner_blendshape": torch.randn(1, 2, 3, 56),
        "dt": torch.ones(1, 2),
        "chunk_mask": torch.tensor([[1, 1]], dtype=torch.bool),
    }
    losses = system.forward_sequence(batch, bptt_chunks=1)
    assert torch.isfinite(losses["total"])
    assert conditioner.input_states[0] is None
    assert torch.equal(conditioner.input_states[1].z, conditioner.output_states[0].z)
    assert torch.equal(
        conditioner.input_states[1].relation, conditioner.output_states[0].relation
    )


def test_probe_is_bounded_and_random_features_are_near_chance():
    torch.manual_seed(4)
    features = torch.randn(40, 8)
    target = torch.arange(40) % 4
    accuracy = linear_leakage_probe(features, target, seed=4)
    assert 0.0 <= accuracy <= 1.0
    assert accuracy < 0.7


def test_nuisance_grl_does_not_change_emotion_forward_path():
    heads = ObservationSupervisionHeads(8, num_speakers=5, num_domains=3).eval()
    value = torch.randn(16, 7, 8)
    with torch.no_grad():
        plain = heads(value, grl_alpha=0.0)
        adversarial = heads(value, grl_alpha=1.0)
    # GRL changes only the backward direction of nuisance heads.  The emotion
    # and VAD predictions used by evaluation must remain identical.
    assert torch.equal(plain["emotion"], adversarial["emotion"])
    assert torch.equal(plain["intensity"], adversarial["intensity"])
    assert torch.equal(plain["vad"], adversarial["vad"])
