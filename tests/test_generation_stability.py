import copy
import io
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from wav2vec import _budget_mask_indices, _compute_mask_indices
from emotion_ssm.config_v3 import default_config
from emotion_ssm.train.generation_sampling import InterleavedCursor, ConversationStates, session_key
from emotion_ssm.train.generation_prefetch import PrefetchSegmentCursor
from emotion_ssm.train.generation_stability import STABILITY_PROTOCOL, optimizer_groups, set_learning_rates
from emotion_ssm.train.generation_v3 import configure_generation_from_config, train_segment, evaluate
from emotion_ssm.utils.reconstruction import ExpressionDecomposition
from emotion_ssm.utils.checkpoint import capture_rng_state, restore_rng_state
from test_v3_generation import small_model, packet
from test_generation_throughput import assert_identical


def config():
    cfg = default_config()
    cfg['generation'].update(train_observer=False, train_state=False)
    cfg['generation_stability'] = dict(protocol=STABILITY_PROTOCOL, conversations_per_rank=2,
        blocks_per_conversation=2, speech_lr=1e-5, film_lr=1e-4, cosine_steps=10000,
        warmup_steps=200, minimum_lr_ratio=.1)
    cfg['train']['lr'] = 3e-5
    return cfg


class Dialogues:
    prefetch_rng_neutral = True

    def __init__(self):
        self.names = [f'video{i}_sub_video_{clip}_speaker1' for i in range(8) for clip in range(2)]
        self.lengths = [7 + i % 3 for i in range(len(self.names))]

    def __len__(self):
        return len(self.names)

    def packets(self, index):
        for tick in range(1, self.lengths[index] + 1):
            p = packet(tick, self.names[index], event=tick == 1)
            mask = torch.ones(1, 25, dtype=torch.bool)
            if tick == 2:
                mask.zero_()
            if tick == self.lengths[index]:
                mask[:, -3:] = False
            yield p, p['partner_blendshape'] * .3, mask


@pytest.mark.parametrize('length', [0, 1, 5, 25, 50, 100, 200])
def test_mask_budget_matches_probability_and_excludes_padding(length):
    np.random.seed(919)
    valid = torch.zeros(3, length + 7, dtype=torch.bool)
    valid[0, :length] = True
    valid[1, :length // 2] = True
    counts = np.zeros(3)
    for _ in range(500):
        mask = _budget_mask_indices(tuple(valid.shape), .05, 10, valid)
        assert not mask[~valid.numpy()].any()
        assert np.all(np.abs(mask.sum(1) - .05 * valid.sum(1).numpy()) <= 1)
        counts += mask.sum(1)
    np.testing.assert_allclose(counts / 500, .05 * valid.sum(1).numpy(), atol=.06)
    assert not _budget_mask_indices((1, length), 0., 10).any()


def test_legacy_zero_mask_is_safe_and_default_policy_remains_explicit():
    assert not _compute_mask_indices((2, 25), 0., 10, min_masks=0).any()


def test_source_shards_chronology_and_no_duplicate_packets():
    dataset = Dialogues()
    all_sources = []
    for rank in (0, 1):
        cursor = InterleavedCursor(dataset, rank=rank, world_size=2, conversations=2, blocks_per_conversation=2)
        sources, seen, previous = set(), set(), {}
        for _ in range(60):
            rows = cursor.take_valid(4)
            assert sum(bool(valid.any()) for _, _, valid in rows) == 4
            for p, _, _ in rows:
                key, tick = session_key(p), p['time']
                assert (key, tick) not in seen
                seen.add((key, tick))
                assert tick == previous.get(key, 0) + 1
                previous[key] = tick
                sources.add(p['training_source_id'])
        all_sources.append(sources)
    assert not all_sources[0] & all_sources[1]


def test_prefetch_serialized_cursor_restores_only_committed_conversations():
    dataset = Dialogues()
    reference = InterleavedCursor(dataset, conversations=2, blocks_per_conversation=2)
    initial_rng = capture_rng_state()
    with PrefetchSegmentCursor(InterleavedCursor(dataset, conversations=2, blocks_per_conversation=2), 4) as prefetch:
        for _ in range(12):
            assert_identical(reference.take_valid(4), prefetch.take_valid(4))
        prefetch._future.result()
        saved = prefetch.state_dict()
        assert_identical(saved, reference.state_dict())
        expected = reference.take_valid(4)
    restored = InterleavedCursor(dataset, conversations=2, blocks_per_conversation=2, saved=saved)
    assert_identical(expected, restored.take_valid(4))
    assert_identical(initial_rng, capture_rng_state())


def test_interleaved_state_preserves_full_history_and_step_boundaries():
    torch.set_num_threads(1)
    model = small_model()
    model.teacher = copy.deepcopy(model.observer)
    parameters = configure_generation_from_config(model, config())
    reference = copy.deepcopy(model).eval()
    optimizer = torch.optim.AdamW(parameters, lr=0.)
    bank = ConversationStates()
    for ticks in ((1, 2), (3, 4), (5, 6)):
        rows = [(packet(t, session=s, event=t == 1), torch.zeros(1, 25, 56), torch.ones(1, 25, dtype=torch.bool))
                for s in ('A', 'B') for t in ticks]
        bank, metrics = train_segment(model, rows, optimizer, state=bank, tbptt_steps=2)
        assert len(bank.states) == 2
        expected_edges = 2 if ticks[0] == 1 else 4
        assert metrics['error_elements']['boundary_velocity'] == expected_edges * 56
    for name in ('A', 'B'):
        state = None
        with torch.no_grad():
            for t in range(1, 7):
                _, state, _ = reference(packet(t, name, event=t == 1), state)
        assert_identical(bank.states[(name, ('A', 'B'))], state.detach())
        assert bank.states[(name, ('A', 'B'))].emotion.elapsed.item() == 6
    original = copy.deepcopy(bank)
    stream = io.BytesIO()
    torch.save(bank, stream)
    stream.seek(0)
    restored = torch.load(stream, weights_only=False)
    final_rows = []
    for name in ('A', 'B'):
        p = packet(7, name)
        p['training_session_end'] = True
        final_rows.append((p, torch.zeros(1, 25, 56), torch.ones(1, 25, dtype=torch.bool)))
    restored, _ = train_segment(model, final_rows, optimizer, state=restored, tbptt_steps=2)
    assert not restored.states and not restored.previous
    assert_identical(bank, original)  # Retry inputs must not be mutated.


def test_interleaving_rejects_accidental_trainable_dynamics():
    model = small_model()
    model.teacher = copy.deepcopy(model.observer)
    cfg = config()
    cfg['generation']['train_state'] = True
    with pytest.raises(ValueError, match='TBPTT'):
        configure_generation_from_config(model, cfg)


def test_parameter_groups_and_schedule_are_disjoint_complete_and_decaying():
    model = small_model()
    model.teacher = copy.deepcopy(model.observer)
    model.generator.baseline.joint_encoder = nn.Module()
    joint = model.generator.baseline.joint_encoder
    for name in ('audio_encoder1', 'audio_encoder2'):
        encoder = nn.Linear(2, 2)
        encoder.config = SimpleNamespace()
        setattr(joint, name, encoder)
    cfg = config()
    params = configure_generation_from_config(model, cfg)
    groups = optimizer_groups(model, cfg)
    ids = [id(p) for g in groups for p in g['params']]
    assert len(ids) == len(set(ids)) and set(ids) == {id(p) for p in params}
    rates = {g['name']: g['lr'] for g in groups}
    assert rates == dict(generator=3e-5, speech=1e-5, film=1e-4)
    optimizer = torch.optim.AdamW(groups)
    first = set_learning_rates(optimizer, cfg, 0)['speech']
    peak = set_learning_rates(optimizer, cfg, 200)['speech']
    late = set_learning_rates(optimizer, cfg, 7000)['speech']
    final = set_learning_rates(optimizer, cfg, 30000)['speech']
    assert first < peak and final < late < peak
    assert final == pytest.approx(1e-6)


def test_bias_decomposition_respects_padding_and_temporal_offset():
    p, t = torch.randn(1, 7, 56), torch.randn(1, 7, 56)
    valid = torch.tensor([[True, True, False, True, True, False, True]])
    p[:, ~valid[0]] = float('nan')
    t[:, ~valid[0]] = float('nan')
    stats = ExpressionDecomposition('cpu')
    stats.update(p, t, valid)
    m = stats.metrics()
    actual = (p[valid][:, :50].double() - t[valid][:, :50].double()).square().mean().item()
    assert m['expression_bias_mse'] + m['expression_centered_mse'] == pytest.approx(actual)


def test_decomposition_and_full_metrics_match_distributed_record_partitions():
    torch.set_num_threads(1)
    model, data = small_model().eval(), Dialogues()
    names = data.names[:4]
    full, _ = evaluate(model, data, 'cpu', selected_names=names, max_blocks=3, representation_diagnostics=False)
    parts = [evaluate(model, data, 'cpu', rank=r, world_size=2, selected_names=names,
                      max_blocks=3, representation_diagnostics=False)[0] for r in (0, 1)]
    for key in ('expression_mse', 'expression_bias_mse', 'expression_centered_mse'):
        weighted = sum(m[key] * m['valid_frames'] for m in parts) / sum(m['valid_frames'] for m in parts)
        assert weighted == pytest.approx(full[key], abs=1e-12)
