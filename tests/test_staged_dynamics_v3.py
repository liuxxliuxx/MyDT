import copy
import json
from pathlib import Path

import pytest
import torch

from emotion_ssm.models.state_core import UnifiedEmotionStateCore
from emotion_ssm.models.token_observer import TokenObserver
from emotion_ssm.train import dynamics_staged_v3 as staged
from emotion_ssm.train.dynamics_v3 import DialogueCollection, encode_pair
from emotion_ssm.train.staged_dynamics_support import (
    EncodedDialogueCache, assert_locked, build_origin_bank, cached_pair, lock_snapshot,
    memory_index, parameter_partition, prediction_loss, state_anchor_loss, weight_digest,
)
from emotion_ssm.utils.checkpoint_v3 import read_checkpoint, save_checkpoint, require_training_revision
from test_v3_dynamics_training import make_training
from test_v3_state_core import observation


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def setup_models(tmp_path):
    config = make_training(tmp_path)
    observer = TokenObserver(config['observer']).eval()
    teacher = copy.deepcopy(observer).requires_grad_(False).eval()
    core = UnifiedEmotionStateCore(observation_dim=4, **config['state'])
    parameter_partition(observer, core, 'calibration')
    cache = EncodedDialogueCache(observer, teacher, tmp_path/'cache', torch.device('cpu'), 4)
    collection = DialogueCollection(config['data']['token_roots'], 'train')
    return config, observer, teacher, core, cache, collection


def test_state_anchor_reaches_state_and_missing_evidence_cannot_train_it():
    core = UnifiedEmotionStateCore(observation_dim=4, relation_dim=3, hidden_dim=8)
    state = core.initialize(2)
    state.fast = torch.randn_like(state.fast, requires_grad=True)
    target = torch.ones_like(state.z, requires_grad=True)
    confidence = torch.ones(2, 2, requires_grad=True)
    loss = state_anchor_loss(state, target, torch.ones(2, 2, dtype=torch.bool), {'evidence': confidence})
    loss.backward()
    assert state.fast.grad.abs().sum() > 0
    assert target.grad is None and confidence.grad is None
    missing = state_anchor_loss(state, target, torch.ones(2, 2, dtype=torch.bool), {'evidence': confidence*0})
    assert missing.item() == 0


def test_phase_partitions_cannot_increase_observation_coverage(tmp_path):
    _, observer, _, core, _, _ = setup_models(tmp_path)
    locks = lock_snapshot(core)
    for phase in ('calibration', 'fixed', 'joint', 'readapt'):
        parameters = parameter_partition(observer, core, phase)
        assert parameters
        assert_locked(core, locks)
        assert not any(p.requires_grad for p in observer.affect_head.parameters())
        assert not any(p.requires_grad for p in observer.reliability_head.parameters())
        assert core.influence[0].weight.requires_grad == (phase in {'calibration', 'joint'})
        assert core.adaptive_flow.left.requires_grad == (phase in {'fixed', 'joint', 'readapt'})
    with torch.no_grad():
        core.fast_correction_logits.add_(.01)
    with pytest.raises(RuntimeError, match='coverage'):
        assert_locked(core, locks)


def test_diagnostics_preserve_forward_and_report_exact_gain():
    torch.manual_seed(9)
    core = UnifiedEmotionStateCore(observation_dim=4, relation_dim=3, hidden_dim=8)
    state = core.initialize(2)
    pair = [observation(present=True, fresh=True, value=.2)]*2
    diagnostics = {}
    actual = core.advance(state, pair, .37, diagnostics=diagnostics)
    expected = core.advance(state, pair, .37)
    torch.testing.assert_close(actual.z, expected.z, rtol=0, atol=0)
    prior = diagnostics['input_conditioned_prior']
    expected_z = prior+diagnostics['correction_gain']*(.2-prior)
    torch.testing.assert_close(actual.z, expected_z)


def test_cached_features_equal_direct_and_heads_remain_trainable(tmp_path):
    _, observer, _, _, cache, collection = setup_models(tmp_path)
    encoded = cache.get(collection, 0)
    before = weight_digest({'observer': observer}, exclude_heads=True)
    for tick in (0, 3):
        direct, _ = encode_pair(observer, collection[0]['packets'][tick], 'cpu', event_id=tick)
        saved = cached_pair(observer, encoded, tick, 'cpu')
        for role in (0, 1):
            for field in ('aff', 'event', 'action', 'reliability'):
                torch.testing.assert_close(getattr(saved[role], field), getattr(direct[role], field), atol=2e-6, rtol=2e-5)
    action = saved[0].action.clone()
    with torch.no_grad():
        next(observer.action_head.parameters()).add_(.2)
    assert weight_digest({'observer': observer}, exclude_heads=True) == before
    changed = cached_pair(observer, cache.get(collection, 0), 3, 'cpu')
    assert not torch.allclose(changed[0].action, action)
    changed[0].action.square().sum().backward()
    assert next(observer.action_head.parameters()).grad.abs().sum() > 0
    assert cache.hits == 1


def test_cache_rejects_changed_source_manifest_by_using_new_key(tmp_path):
    _, _, _, _, cache, collection = setup_models(tmp_path)
    first = cache.get(collection, 0)
    manifest = collection.datasets[0].root/'manifest.json'
    data = json.loads(manifest.read_text()); data['processing_revision'] = 'changed'
    manifest.write_text(json.dumps(data))
    second = cache.get(collection, 0)
    assert first['metadata']['manifest'] != second['metadata']['manifest']
    assert cache.misses == 2


def test_fixed_bank_is_immutable_and_replay_detects_deployment_shift(tmp_path):
    _, observer, _, core, cache, collection = setup_models(tmp_path)
    bank = build_origin_bank(observer, core, cache, collection, [0, 1], [1, 2], stride=1)
    before = bank['states'].z.clone(); producer = bank['producer']
    parameter_partition(observer, core, 'fixed')
    indices = torch.arange(3)
    loss, outputs = prediction_loss(core, bank, indices, 'cpu', observer, .1)
    loss.backward()
    assert core.adaptive_flow.left.grad.abs().sum() > 0
    assert not any(p.grad is not None for p in observer.parameters())
    torch.testing.assert_close(before, bank['states'].z, rtol=0, atol=0)
    with torch.no_grad():
        core.rotation.add_(.4)
    fresh = build_origin_bank(observer, core, cache, collection, [0, 1], [1, 2], stride=1)
    assert fresh['producer'] != producer and fresh['keys'] == bank['keys']
    assert not torch.allclose(fresh['states'].z, before)


def test_future_targets_cannot_change_forecast(tmp_path):
    _, observer, _, core, cache, collection = setup_models(tmp_path)
    bank = build_origin_bank(observer, core, cache, collection, [0], [1, 2], stride=1)
    indices = torch.arange(2)
    _, outputs = prediction_loss(core, bank, indices, 'cpu')
    bank['targets'].normal_(100, 5)
    _, changed = prediction_loss(core, bank, indices, 'cpu')
    for a, b in zip(outputs, changed):
        torch.testing.assert_close(a.z, b.z, rtol=0, atol=0)


def test_batched_forecast_and_weighted_metrics_match_different_batches(tmp_path):
    _, observer, _, core, cache, collection = setup_models(tmp_path)
    bank = build_origin_bank(observer, core, cache, collection, [0, 1], [1, 2], stride=1)
    mean, shrink = staged.fit_shrink(bank, torch.device('cpu'))
    one = staged.evaluate_bank(core, bank, mean, shrink, 'cpu', 1)
    many = staged.evaluate_bank(core, bank, mean, shrink, 'cpu', 6)
    for method in one['methods']:
        assert one['methods'][method]['elements'] == many['methods'][method]['elements']
        assert one['methods'][method]['mse'] == pytest.approx(many['methods'][method]['mse'], rel=2e-6)


def test_checkpoint_gate_rejects_posterior_only_or_stale_origin_improvement():
    def result(error, gain=.1):
        return {'methods': {'learned': {'macro_mse': error}}, 'gain_over_hold': gain, 'producer': 'current'}
    initial = {name: result(1.) for name in ('fixed', 'live', 'gap')}
    candidate = {name: result(.9) for name in ('fixed', 'live', 'gap')}
    candidate.update(candidate_hash='current', correction_parameters_unchanged=True)
    assert staged.deployment_gate(candidate, initial, staged.DEFAULTS)['passed']
    candidate['fixed'] = result(1.)
    candidate['posterior_mse'] = .00001
    assert not staged.deployment_gate(candidate, initial, staged.DEFAULTS)['passed']
    candidate['fixed'] = result(.9); candidate['live']['producer'] = 'stale'
    assert not staged.deployment_gate(candidate, initial, staged.DEFAULTS)['passed']


def test_small_complete_stages_and_self_contained_resume(tmp_path, monkeypatch):
    config, observer, teacher, core, _, _ = setup_models(tmp_path)
    config['train']['cpu_threads'] = 1
    source = tmp_path/'source_dynamics.pt'
    save_checkpoint(source, dict(observer=observer, teacher=teacher, state=core), config,
                    dict(observer=observer.construction(), state=core.get_config()), 'dynamics_v3')
    config['paths'].update(dynamics_checkpoint=str(source), feature_cache=str(tmp_path/'features'))
    config['staged'] = dict(calibration_steps=1, fixed_steps=1, bank_refresh_steps=1,
        joint_rounds=1, joint_steps_per_round=1, readapt_steps_per_round=1,
        bank_dialogues_per_domain=2, validation_dialogues_per_domain=2,
        origin_stride=1, cache_batch=4, prediction_batch=4, validation_every=1,
        checkpoint_every=1, log_every=1, block_gap_seconds=1, block_gap_period=4,
        sparse_subset_probability=0., input_dropout=0., endpoint_label_weight=.1)
    saved_step = tmp_path/'step2.pt'
    original_save = staged.save_checkpoint
    def save(path, *args, **kwargs):
        value = original_save(path, *args, **kwargs)
        if Path(path).name == 'last.pt' and value['global_step'] == 2:
            torch.save(value, saved_step)
        return value
    monkeypatch.setattr(staged, 'save_checkpoint', save)
    result = staged.run(config)
    assert result['step'] == 4
    complete = read_checkpoint(Path(config['paths']['output'])/'last.pt')
    require_training_revision(complete)
    assert complete['kind'] == 'dynamics_staged_v3'
    source.rename(tmp_path/'source_moved.pt')
    resumed_config = copy.deepcopy(config); resumed_config['paths']['resume'] = str(saved_step)
    resumed = staged.run(resumed_config)
    assert resumed['step'] == 4
    after = read_checkpoint(Path(config['paths']['output'])/'last.pt')
    for name in complete['models']:
        for key, tensor in complete['models'][name].items():
            torch.testing.assert_close(tensor, after['models'][name][key], rtol=0, atol=0)
    assert after['metrics']['live']['producer'] == after['metrics']['candidate_hash']
