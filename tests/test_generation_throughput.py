import copy
from dataclasses import fields, is_dataclass
import random

import numpy as np
import pytest
import torch
from torch import nn

from emotion_ssm.train.generation_prefetch import PrefetchSegmentCursor
from emotion_ssm.train.generation_v3 import (
    SegmentCursor, configure_generation_from_config, generation_execution_settings, train_segment,
)
from emotion_ssm.utils.checkpoint import capture_rng_state, restore_rng_state
from emotion_ssm.utils.reconstruction import ReconstructionTotals, DeferredReconstructionTotals
from emotion_ssm.config_v3 import default_config
from test_v3_generation import small_model, packet, TinyGenerator


def assert_identical(a, b):
    if torch.is_tensor(a):
        assert torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b)
    elif is_dataclass(a):
        for field in fields(a):
            assert_identical(getattr(a, field.name), getattr(b, field.name))
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_identical(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_identical(x, y)
    else:
        assert a == b


class FixedPackets:
    prefetch_rng_neutral = True

    def __init__(self):
        self.rows = []
        for dialogue in range(5):
            rows = []
            for tick in range(1, 7):
                p = packet(tick, session=f"dialogue-{dialogue}", event=tick == 1)
                truth = p['partner_blendshape'].clone() * .25
                mask = torch.ones(1, 25, dtype=torch.bool)
                if tick == 2:
                    mask.zero_()
                elif tick == 6:
                    mask[:, -4:] = False
                rows.append((p, truth, mask))
            self.rows.append(rows)

    def __len__(self):
        return len(self.rows)

    def packets(self, index):
        yield from copy.deepcopy(self.rows[index])


class StochasticGenerator(TinyGenerator):
    def forward(self, *args, **kwargs):
        result = super().forward(*args, **kwargs)
        # Exercises the same two RNG families as checkpointed audio augmentation.
        result = result * float(np.random.uniform(.8, 1.2))
        return nn.functional.dropout(result, .2, self.training)


@pytest.mark.parametrize('previous_enabled', [False, True])
def test_deferred_totals_bitwise_match_original(previous_enabled):
    torch.manual_seed(42)
    old, new = ReconstructionTotals(), DeferredReconstructionTotals('cpu')
    previous = None
    for index in range(9):
        p, t = torch.randn(2, 25, 56), torch.randn(2, 25, 56)
        mask = torch.rand(2, 25) > .2
        for totals in [old, new]:
            totals.update(p, t, mask, previous)
        if previous_enabled:
            previous = (p[:, -1], t[:, -1], mask[:, -1])
    new.distributed_sum('cpu')
    assert_identical(old.metrics(), new.metrics())
    assert_identical(old.square_error, new.square_error)
    assert_identical(old.elements, new.elements)


@pytest.mark.parametrize('rank', [0, 1])
def test_prefetch_order_rng_peek_and_resume(rank):
    data = FixedPackets()
    old = SegmentCursor(data, rank=rank, world_size=2)
    rng = capture_rng_state()
    with PrefetchSegmentCursor(SegmentCursor(data, rank=rank, world_size=2), 4, 3) as new:
        for _ in range(12):
            assert_identical(old.take_valid(4), new.take_valid(4))
            assert_identical(old.peek(3), new.peek(3))
            assert new.seen == old.seen
        # Let a speculative read finish; it must not advance the saved cursor.
        new._future.result()
        committed = new.seen
        expected = old.take_valid(4)
    resumed = SegmentCursor(data, rank=rank, world_size=2, seen=committed)
    assert_identical(expected, resumed.take_valid(4))
    assert_identical(rng, capture_rng_state())


def test_prefetch_surfaces_worker_failure_and_rejects_stochastic_data():
    class Broken(FixedPackets):
        def packets(self, index):
            raise ValueError('broken source')
            yield
    with PrefetchSegmentCursor(SegmentCursor(Broken()), 2) as cursor:
        with pytest.raises(ValueError, match='broken source'):
            cursor.take_valid(2)
    data = FixedPackets()
    data.prefetch_rng_neutral = False
    with pytest.raises(ValueError, match='RNG-neutral'):
        PrefetchSegmentCursor(SegmentCursor(data), 2)


@pytest.mark.parametrize('prefetch,defer', [(0, True), (1, False), (1, True)])
def test_twenty_training_steps_preserve_weights_gradients_optimizer_state_and_rng(prefetch, defer):
    torch.set_num_threads(1)
    torch.manual_seed(29)
    random.seed(31)
    np.random.seed(37)
    original = small_model('dyadic')
    original.generator = StochasticGenerator(original.generator.context_dim)
    original.teacher = copy.deepcopy(original.observer)
    cfg = default_config()
    cfg['generation'].update(train_observer=False, train_state=False)
    data, initial_rng = FixedPackets(), capture_rng_state()
    reference = []
    for optimized in [False, True]:
        model = copy.deepcopy(original)
        params = configure_generation_from_config(model, cfg)
        optimizer = torch.optim.AdamW(params, lr=1e-3, foreach=False)
        cursor = SegmentCursor(data)
        if optimized and prefetch:
            cursor = PrefetchSegmentCursor(cursor, 4)
        restore_rng_state(initial_rng)
        state = None
        try:
            for step in range(20):
                rows = cursor.take_valid(4)
                state, metrics = train_segment(model, rows, optimizer, state=state, tbptt_steps=3,
                                               defer_metrics=optimized and defer)
                result = dict(model=copy.deepcopy(model.state_dict()),
                              grads=[p.grad.clone() for p in params],
                              optimizer=copy.deepcopy(optimizer.state_dict()),
                              stream=copy.deepcopy(state), metrics=metrics, seen=cursor.seen,
                              rng=capture_rng_state())
                if optimized:
                    assert_identical(reference[step], result)
                else:
                    reference.append(result)
        finally:
            if hasattr(cursor, 'close'):
                cursor.close()


def test_resume_execution_overrides_do_not_override_training_config():
    saved = default_config()
    request = {'train': {'lr': 100, 'global_chunks_per_step': 999},
               'generation_execution': {'defer_metrics': True, 'prefetch_batches': 1}}
    result = generation_execution_settings(saved, request)
    assert result == request['generation_execution']
    assert saved['train']['lr'] == 1e-4 and saved['train']['global_chunks_per_step'] == 32
    with pytest.raises(ValueError, match='Unknown'):
        generation_execution_settings(saved, {'generation_execution': {'batch_size': 64}})


def test_benchmark_gate_rejects_any_state_or_metric_difference(tmp_path):
    import json
    from scripts.resume_avatar_throughput import compare_reports
    base = dict(ranks=[{'model_hash': 'same', 'gradient_hash': 'same', 'rng_hash': 'same'}],
                rows=[{'metrics': {'loss': .5}}], steps=20, gradients_only=False,
                mean_step_seconds=10., mean_data_wait_seconds=.5)
    for mode in ['reference', 'combined']:
        directory = tmp_path/mode
        directory.mkdir()
        (directory/'report.json').write_text(json.dumps(base))
    result = compare_reports(tmp_path, ['reference', 'combined'])
    assert result['passed'] and result['speed']['combined']['speedup'] == 1.
    changed = copy.deepcopy(base)
    changed['rows'][0]['metrics']['loss'] += 1e-10
    (tmp_path/'combined/report.json').write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='metrics differ'):
        compare_reports(tmp_path, ['reference', 'combined'])
    changed = copy.deepcopy(base)
    changed['ranks'][0]['rng_hash'] = 'changed'
    (tmp_path/'combined/report.json').write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='equality failed'):
        compare_reports(tmp_path, ['reference', 'combined'])
