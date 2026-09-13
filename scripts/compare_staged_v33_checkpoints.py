"""Re-evaluate checkpoint snapshots with common coordinates and causal queries.

Only evaluation outputs are written. Frozen observation features are reused after
hash checks; every online memory is rebuilt from the start of every dialogue.
The existing v3.3 evaluator supplies both vector and true-endpoint metrics.
"""
import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import torch

from emotion_ssm.models.token_observer import TokenObserver
from emotion_ssm.models.state_core import UnifiedEmotionStateCore
from emotion_ssm.train.dynamics_v3 import unit_state_readout
from emotion_ssm.train.staged_dynamics_support import memory_index, weight_digest
from emotion_ssm.train.staged_v33.baselines import fit, refit_output_calibration
from emotion_ssm.train.staged_v33.data import build_bank
from emotion_ssm.train.staged_v33.evaluation import evaluate
from emotion_ssm.train.staged_v33.missing import digest
from emotion_ssm.utils.checkpoint_v3 import read_checkpoint


REVISION = 'common-coordinates-full-replay-32s-v1'
LABEL_PREFIXES = ('emotion_head.', 'vad_head.', 'intensity_head.')


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def coordinate_hash(payload):
    return weight_digest({k: payload['models'][k] for k in ('observer', 'teacher')}, exclude_heads=True)


def require_coordinates(reference, candidate):
    if coordinate_hash(reference) != coordinate_hash(candidate):
        raise ValueError('Observer coordinates, label heads, or teacher changed')
    if reference['construction']['observer'] != candidate['construction']['observer']:
        raise ValueError('Observer construction changed despite equal tensor dimensions')


def horizon_bank(bank, horizon):
    """Retain the original origin set; each selected real endpoint has weight one."""
    col = bank['horizons'].index(horizon)
    out = dict(bank, horizons=[horizon], targets=bank['targets'][:, col:col+1],
               valid=bank['valid'][:, col:col+1])
    out['endpoints'] = [(row, dict(q, endpoint_weight=1.)) for row, q in bank['endpoints']
                        if q['nominal_horizon'] == horizon]
    keys = [(bank['keys'][row][0], q['role'], q['endpoint']) for row, q in out['endpoints']]
    if len(keys) != len(set(keys)):
        raise ValueError('Selected horizon contains duplicate real endpoints')
    return out


def target_hash(bank):
    h = hashlib.sha256()
    h.update(digest(dict(keys=bank['keys'], horizons=bank['horizons'],
        endpoints=bank['endpoints'])).encode())
    for name in ('domain', 'targets', 'valid', 'current_gold', 'current_valid'):
        value = bank[name].contiguous().cpu()
        h.update((name+str(value.dtype)+str(value.shape)).encode())
        h.update(value.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def verify_bank(bank, payload, cohort, settings):
    identities = [e['identity'] for e in bank['encoded_rows']]
    if identities != cohort:
        raise ValueError('Cached dialogue order differs from the fixed cohort')
    core = UnifiedEmotionStateCore.from_config(payload['construction']['state'])
    expected = digest(dict(weights=weight_digest({k:payload['models'][k] for k in ('observer','state')}),
        construction=core.get_config(), protocol='clean', partner=settings['enable_partner'],
        plans=[e['plan']['sha256'] for e in bank['encoded_rows']]))
    if expected != bank['producer']:
        raise ValueError('Common bank was not produced by the declared reference checkpoint')
    for encoded in bank['encoded_rows']:
        if encoded['metadata']['encoder'] != coordinate_hash(payload):
            raise ValueError('Frozen feature teacher/observer mismatch')
        if encoded['plan']['protocol'] != 'clean':
            raise ValueError('Online comparison requires clean causal replay')


class FrozenViews:
    """Read existing frozen observations; event/action heads run in pair_at()."""
    def __init__(self, bank, device, settings):
        self.rows = bank['encoded_rows']
        self.device, self.settings = device, settings

    def get(self, collection, index, protocol='clean'):
        if protocol != 'clean':
            raise ValueError('Only clean online replay is supported here')
        return self.rows[index]


def build_models(payload, device, execution):
    observer = TokenObserver(payload['construction']['observer']).to(device).eval().requires_grad_(False)
    core = UnifiedEmotionStateCore.from_config(payload['construction']['state']).to(device).eval().requires_grad_(False)
    observer.load_state_dict(payload['models']['observer'], strict=True)
    core.load_state_dict(payload['models']['state'], strict=True)
    core.configure_execution(execution)
    return observer, core


@torch.no_grad()
def replay(observer, core, frozen_bank, device, settings, progress):
    views = FrozenViews(frozen_bank, device, settings)
    return build_bank(observer, core, views, None, list(range(len(views.rows))), frozen_bank['horizons'],
                      settings['origin_stride'], 'clean', settings['enable_partner'], progress)


@torch.no_grad()
def endpoint_predictions(decoder, core, bank, settings, device):
    """Per-dialogue audit and paired uncertainty; never used to select weights."""
    result = []
    items = bank['endpoints']
    for start in range(0, len(items), settings['prediction_batch']):
        subset = items[start:start+settings['prediction_batch']]
        rows = torch.tensor([r for r, _ in subset])
        state = memory_index(bank['states'], rows).to(device)
        seconds = torch.tensor([q['seconds'] for _, q in subset], device=device)
        future = core._propagate(state, seconds, enable_partner=settings['enable_partner']).z
        role = torch.tensor([q['role'] for _, q in subset], device=device)
        indices = torch.arange(len(rows), device=device)
        decoded = {name: decoder.decode_affect(unit_state_readout(value[indices, role]))
                   for name, value in [('learned', future), ('hold', state.z)]}
        for i, (row, q) in enumerate(subset):
            result.append(dict(dialogue=bank['keys'][row][0], domain=int(bank['domain'][row]),
                origin=bank['keys'][row][1], endpoint=q['endpoint'], seconds=q['seconds'], role=q['role'],
                label=q['label'], predictions={name:dict(emotion=int(d['emotion_logits'][i].argmax()),
                    vad=d['vad'][i].cpu().tolist(), intensity=float(d['intensity'][i].reshape(())))
                    for name, d in decoded.items()}))
    return result


def selected_summary(metrics):
    return {name:dict(vector_mse=m['macro_mse'], vector_micro_mse=m['mse'],
                      vector_by_domain=m['by_domain'], **metrics['semantic'][name]['query_weighted'])
            for name, m in metrics['methods'].items()}


def semantic_counts(bank):
    rows = bank['endpoints']
    result = {}
    for domain in sorted(set(bank['domain'].tolist())):
        queries = [q for r,q in rows if int(bank['domain'][r]) == domain]
        seconds = [q['seconds'] for q in queries]
        result[str(domain)] = dict(endpoints=len(queries),
            emotion=sum(0 <= int(q['label'].get('emotion', -1)) < 7 for q in queries),
            vad_coordinates=sum(sum(q['label'].get('vad_mask', [])) for q in queries),
            intensity=sum(bool(q['label'].get('intensity_mask')) for q in queries),
            min_actual_seconds=min(seconds) if seconds else None,
            max_actual_seconds=max(seconds) if seconds else None,
            mean_actual_seconds=sum(seconds)/len(seconds) if seconds else None)
    return result


@torch.no_grad()
def run(args):
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    def emit(event, **values):
        print(json.dumps(dict(event=event, elapsed=time.monotonic()-started, **values)), flush=True)
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(6666)
    device = torch.device(args.device)
    settings_payload = read_checkpoint(args.protocol_checkpoint)
    settings = dict(settings_payload['config']['staged'], prediction_batch=args.batch_size)
    reference = read_checkpoint(args.reference)
    require_coordinates(reference, settings_payload)
    cohort = settings_payload['run_state']['cohort']
    for root, metadata in cohort['provenance'].items():
        current = hashlib.sha256((Path(root)/'manifest.json').read_bytes()).hexdigest()
        if current != metadata['sha256']:
            raise ValueError('Dataset manifest changed: '+root)
    canonical = settings_payload['config']['data']['token_roots']
    candidates, missing = [], []
    for entry in args.model:
        name, path = entry.split('=', 1)
        if not Path(path).is_file():
            missing.append(dict(name=name, path=path, status='checkpoint_not_available')); continue
        # Opening once remains coherent if training atomically replaces best.pt.
        raw = Path(path).read_bytes()
        snapshot = output/(name+'.snapshot.pt')
        snapshot.write_bytes(raw)
        payload = read_checkpoint(snapshot)
        require_coordinates(reference, payload)
        if payload['config']['data']['token_roots'] != canonical:
            raise ValueError('Candidate uses different data roots')
        candidate_cohort = payload['run_state'].get('cohort')
        if candidate_cohort and candidate_cohort['sha256'] != cohort['sha256']:
            raise ValueError('Candidate calibration/validation cohort differs')
        candidates.append((name, payload, dict(path=path, snapshot=str(snapshot),
            checkpoint_sha256=hashlib.sha256(raw).hexdigest(), weight_hash=weight_digest(payload['models']),
            step=payload['global_step'])))
        del raw
    reference_observer, reference_core = build_models(reference, device, args.execution)
    decoder = reference_observer
    full_fixed = torch.load(args.fixed_validation, map_location='cpu', weights_only=False)
    full_train = torch.load(args.fixed_calibration, map_location='cpu', weights_only=False)
    verify_bank(full_fixed, reference, cohort['validation'], settings)
    verify_bank(full_train, reference, cohort['train'], settings)
    for bank in (full_fixed, full_train):
        for e in bank['encoded_rows']:
            source = e['identity'].split('::', 1)[0]
            if e['metadata']['manifest'] != cohort['provenance'][source]['sha256']:
                raise ValueError('Cached feature manifest differs from current manifest')
    fixed, train = horizon_bank(full_fixed, args.horizon), horizon_bank(full_train, args.horizon)
    target_identity = target_hash(fixed)
    emit('fitting_common_baselines', origins=len(train['keys']))
    fixed_fit = fit(train, reference_core, settings, device)
    result = dict(revision=REVISION, time=time.strftime('%Y-%m-%d %H:%M:%S%z'), horizon=args.horizon,
        execution=args.execution, device=str(device), batch_size=args.batch_size,
        common_reference_step=reference['global_step'], coordinate_hash=coordinate_hash(reference),
        teacher_hash=weight_digest({'teacher': reference['models']['teacher']}),
        label_head_hash=weight_digest({'labels':{k:v for k,v in reference['models']['observer'].items()
            if k.startswith(LABEL_PREFIXES)}}),
        common_origin_producer=full_fixed['producer'], target_hash=target_identity,
        cohort_sha256=cohort['sha256'], validation_dialogues=cohort['validation'],
        calibration_dialogues=cohort['train'], manifest_hashes={k:v['sha256'] for k,v in cohort['provenance'].items()},
        endpoint_counts=semantic_counts(fixed), missing_models=missing, models={},
        protocol=dict(vector='Existing v3.3 stride origins; forecast exactly h seconds; teacher freshness mask',
            semantic='Latest complete origin <= real endpoint minus h; propagate actual elapsed seconds',
            semantic_weighting='Each distinct real endpoint at this horizon counted once',
            f1_classes='Macro average over classes present in pooled ground truth',
            fixed='Same old9750 full memory, including fast/slow/relation, for every candidate',
            online='Replay every dialogue from initialization with each candidate; no truncation/reset',
            baselines='Train cohort only; fixed baselines shared, online baselines refit to each origin producer',
            continuous_shrink='Rate fitted to this horizon only in this 32s-focused comparison',
            selection='Existing best checkpoint, selected by all-horizon fixed vector MSE; no selection by these results'))
    atomic_json(output/'comparison.json', result)
    for name, payload, metadata in candidates:
        emit('model_started', model=name, step=payload['global_step'])
        observer, core = build_models(payload, device, args.execution)
        row = dict(metadata, views={})
        for view in ('fixed', 'online'):
            if view == 'fixed':
                bank = fixed
                fitted = refit_output_calibration(fixed_fit, train, core, settings, device)
            else:
                def progress(done, total, origins):
                    if done % 4 == 0 or done == total:
                        emit('full_replay', model=name, subset=subset_name, dialogues=done, total=total, origins=origins)
                subset_name = 'validation'
                full_online = replay(observer, core, full_fixed, device, settings, progress)
                bank = horizon_bank(full_online, args.horizon)
                subset_name = 'train_calibration'
                online_train = horizon_bank(replay(observer, core, full_train, device, settings, progress), args.horizon)
                fitted = fit(online_train, core, settings, device)
            if target_hash(bank) != target_identity:
                raise ValueError('Causal origins, teacher targets, real endpoints, masks, or labels differ')
            emit('evaluating', model=name, view=view)
            metrics = evaluate(decoder, core, bank, fitted, settings, device)
            records = endpoint_predictions(decoder, core, bank, settings, device)
            # Two independent reductions must agree for the human-label counts.
            if len(records) != sum(x['endpoints'] for x in result['endpoint_counts'].values()):
                raise AssertionError('Per-endpoint predictions lost an annotation')
            metrics['endpoint_predictions'] = records
            metrics['target_hash'] = target_hash(bank)
            metrics['state_hash'] = weight_digest({'state_memory':vars(bank['states'])})
            atomic_json(output/(name+'_'+view+'.json'), metrics)
            row['views'][view] = dict(summary=selected_summary(metrics), query_hash=metrics['query_hash'],
                state_hash=metrics['state_hash'], producer=bank['producer'], current=bank['current'],
                file=name+'_'+view+'.json')
            result['models'][name] = row
            atomic_json(output/'comparison.json', result)
            emit('view_complete', model=name, view=view, learned=row['views'][view]['summary']['learned'])
            if view == 'online':
                del full_online, online_train
        del observer, core
    result['elapsed_seconds'] = time.monotonic()-started
    result['status'] = 'complete_available_checkpoints'
    atomic_json(output/'comparison.json', result)
    emit('complete', models=list(result['models']), missing=missing)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--protocol-checkpoint', required=True)
    parser.add_argument('--fixed-validation', required=True)
    parser.add_argument('--fixed-calibration', required=True)
    parser.add_argument('--model', action='append', required=True, help='name=checkpoint; unavailable files recorded')
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--execution', default='optimized', choices=['reference','optimized','compiled'])
    parser.add_argument('--horizon', type=float, default=32.)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--cpu-threads', type=int, default=2)
    run(parser.parse_args())
