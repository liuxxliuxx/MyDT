import copy
import pytest
import torch

from test_staged_v33 import case
from scripts.compare_staged_v33_checkpoints import (
    FrozenViews, horizon_bank, target_hash, require_coordinates, replay,
    semantic_counts, endpoint_predictions,
)
from emotion_ssm.train.staged_v33.baselines import fit
from emotion_ssm.train.staged_v33.evaluation import evaluate


def test_common_coordinate_guard_includes_labels_but_allows_event_head_updates(tmp_path):
    _, observer, teacher, core, _, _, _, _ = case(tmp_path)
    payload = dict(construction=dict(observer=observer.construction()),
        models=dict(observer=observer.state_dict(), teacher=teacher.state_dict()))
    candidate = copy.deepcopy(payload)
    event_weight = next(k for k in candidate['models']['observer'] if k.startswith('event_head.') and k.endswith('weight'))
    candidate['models']['observer'][event_weight] += .1
    require_coordinates(payload, candidate)
    candidate['models']['observer']['emotion_head.weight'] += .1
    with pytest.raises(ValueError, match='coordinates'):
        require_coordinates(payload, candidate)


def test_selected_horizon_preserves_true_times_and_hashes_labels_and_masks(tmp_path):
    _, _, _, _, _, _, bank, _ = case(tmp_path)
    selected = horizon_bank(bank, 2)
    for row, query in selected['endpoints']:
        assert query['seconds'] == query['endpoint'] - selected['keys'][row][1]
        assert query['seconds'] >= 2
        assert query['endpoint_weight'] == 1
    assert len(selected['endpoints']) == sum(v['endpoints'] for v in semantic_counts(selected).values())
    altered = copy.deepcopy(selected)
    altered['endpoints'][0][1]['label']['emotion'] = 6
    assert target_hash(selected) != target_hash(altered)
    altered = copy.deepcopy(selected)
    altered['valid'].logical_not_()
    assert target_hash(selected) != target_hash(altered)
    duplicated = copy.deepcopy(bank)
    duplicated['endpoints'].append(next((r,q) for r,q in duplicated['endpoints'] if q['nominal_horizon']==2))
    with pytest.raises(ValueError, match='duplicate'):
        horizon_bank(duplicated, 2)


def test_full_replay_reproduces_source_and_uses_candidate_state_weights(tmp_path):
    _, observer, _, core, _, _, bank, settings = case(tmp_path)
    with torch.no_grad():
        rebuilt = replay(observer, core, bank, torch.device('cpu'), settings, None)
        torch.testing.assert_close(rebuilt['states'].z, bank['states'].z)
        changed = copy.deepcopy(core)
        changed.baseline.add_(.2)
        other = replay(observer, changed, bank, torch.device('cpu'), settings, None)
    assert target_hash(bank) == target_hash(other)
    assert not torch.allclose(other['states'].z, bank['states'].z)
    assert other['producer'] != bank['producer']


def test_endpoint_records_agree_with_unified_evaluator_and_batch_size(tmp_path):
    _, observer, _, core, _, _, full, settings = case(tmp_path)
    bank = horizon_bank(full, 2)
    fitted = fit(bank, core, settings, torch.device('cpu'))
    one = evaluate(observer, core, bank, fitted, dict(settings,prediction_batch=1), torch.device('cpu'))
    many = evaluate(observer, core, bank, fitted, dict(settings,prediction_batch=4), torch.device('cpu'))
    torch.testing.assert_close(torch.tensor(one['vector_sums']), torch.tensor(many['vector_sums']),atol=1e-7,rtol=1e-5)
    records = endpoint_predictions(observer, core, bank, settings, torch.device('cpu'))
    valid = [x for x in records if 0 <= x['label'].get('emotion',-1) < 7]
    expected = one['semantic']['learned']['query_weighted']
    assert len(valid) == expected['emotion_count']
    correct = sum(x['predictions']['learned']['emotion']==x['label']['emotion'] for x in valid)
    assert expected['accuracy'] == correct/len(valid)
