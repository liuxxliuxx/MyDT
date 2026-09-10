"""State calibration -> fixed-origin prediction -> joint/replay adaptation.

The architecture and affect coordinates are unchanged. Only the learning
protocol changes. Deployment acceptance requires a new full-dialogue replay
under the exact candidate weights, in addition to a permanently fixed-origin
control. Current reconstruction cannot select a checkpoint.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.distributed as dist

from emotion_ssm.config_v3 import STAGED_DYNAMICS_REVISION, read_config, write_config
from emotion_ssm.models.state_core import EmotionMemory, UnifiedEmotionStateCore
from emotion_ssm.models.token_observer import TokenObserver, SUBSETS
from emotion_ssm.train.dynamics_v3 import DialogueCollection, _distributed_device, synchronize_gradients
from emotion_ssm.train.staged_dynamics_support import (
    EncodedDialogueCache, assert_locked, bank_from_rows, build_origin_bank,
    cached_pair, lock_snapshot, memory_index, parameter_partition,
    prediction_loss, state_anchor_loss, weight_digest,
)
from emotion_ssm.utils.checkpoint import capture_rng_state, restore_rng_state
from emotion_ssm.utils.checkpoint_v3 import read_checkpoint, save_checkpoint, manifest_provenance
from emotion_ssm.utils.dynamics_execution import EXECUTION_REVISION, synchronize_gradients_batched


DEFAULTS = dict(calibration_steps=1000, fixed_steps=4000, joint_rounds=4,
                joint_steps_per_round=250, readapt_steps_per_round=1000,
                bank_refresh_steps=1000, bank_dialogues_per_domain=64,
                validation_dialogues_per_domain=8, origin_stride=4,
                prediction_batch=64, cache_batch=32, state_anchor_weight=.1,
                calibration_lr=3e-5, prediction_lr=1e-4, joint_lr=1e-5,
                joint_prediction_lr=3e-5, endpoint_label_weight=1.,
                block_gap_seconds=8, block_gap_period=32,
                sparse_subset_probability=.2, input_dropout=.1,
                acceptance_tolerance=.02, minimum_fixed_improvement=.005,
                validation_every=250, checkpoint_every=250, log_every=10,
                clip_grad=5., weight_decay=.001, compile_flow=False)


def stage_plan(settings):
    blocks = [('calibration', int(settings['calibration_steps']))]
    remaining = int(settings['fixed_steps'])
    while remaining:
        amount = min(remaining, int(settings['bank_refresh_steps']))
        blocks.append(('fixed', amount)); remaining -= amount
    for _ in range(int(settings['joint_rounds'])):
        blocks.extend([('joint', int(settings['joint_steps_per_round'])),
                       ('readapt', int(settings['readapt_steps_per_round']))])
    if any(steps < 1 for _, steps in blocks):
        raise ValueError('Each staged training block must contain positive steps')
    return blocks


def source_balanced_indices(collection, per_domain, seed, cycle=0):
    selected = []
    for domain in range(len(collection.datasets)):
        rows = [i for i, item in enumerate(collection.index) if item[0] == domain]
        random.Random(seed+domain).shuffle(rows)
        if not rows:
            continue
        count = min(per_domain, len(rows))
        offset = (cycle*count) % len(rows)
        selected.extend(rows[(offset+j) % len(rows)] for j in range(count))
    return selected


def all_reduce(value):
    if dist.is_initialized():
        dist.all_reduce(value)
    return value


def cpu_models(models):
    return {name: {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            for name, model in models.items()}


class OnlineDialogueCursor:
    def __init__(self, collection, cache, seed, rank, world):
        self.collection, self.cache = collection, cache
        self.seed, self.rank, self.world = seed, rank, world
        self.epoch = self.position = self.tick = 0
        self.memory = None
        self._encoded = {}

    def order(self):
        values = list(range(self.rank, len(self.collection), self.world))
        random.Random(self.seed+self.epoch*104729+self.rank).shuffle(values)
        return values

    def state_dict(self):
        return dict(epoch=self.epoch, position=self.position, tick=self.tick,
                    memory=None if self.memory is None else self.memory.detach().to('cpu').state_dict())

    def load(self, state, device):
        self.epoch, self.position, self.tick = state['epoch'], state['position'], state['tick']
        self.memory = None if state['memory'] is None else EmotionMemory.from_state_dict(state['memory']).to(device)
        self._encoded = {}

    def invalidate_history(self):
        # Recreate the consumed prefix with the new weights at stage changes.
        self.memory = None

    def encoded(self, subset):
        if subset not in self._encoded:
            self._encoded[subset] = self.cache.get(self.collection, self.order()[self.position], subset)
        return self._encoded[subset]

    def advance_cursor(self):
        self.tick += 1
        if self.tick == len(next(iter(self._encoded.values()))['dt']):
            self.tick = 0; self.position += 1
            self.memory = None; self._encoded = {}
            if self.position == len(self.order()):
                self.position = 0; self.epoch += 1

    def window(self, observer, core, count, horizons, settings):
        device = self.cache.device
        memories, records, encoded_rows, anchors = [], [], [], []
        correction_sum = torch.zeros((), device=device)
        correction_count = 0
        # One sampled view per window; chronological state carry is retained.
        subset = random.choice([s for s in SUBSETS if s != 'AVT']) if random.random() < settings['sparse_subset_probability'] else 'AVT'
        for _ in range(count):
            encoded = self.encoded(subset)
            if self.memory is None:
                self.memory = core.initialize(1, device)
                # No stale state survives a stage/producer change. The prefix
                # is causal; future packet tensors never enter this replay.
                with torch.no_grad():
                    clean = self.encoded('AVT')
                    for past in range(self.tick):
                        pair = cached_pair(observer, clean, past, device)
                        self.memory = core.advance(self.memory, pair, clean['dt'][past])
            tick = self.tick
            seconds = encoded['times'][tick]
            in_gap = seconds % settings['block_gap_period'] >= settings['block_gap_period']-settings['block_gap_seconds']
            missing = in_gap or random.random() < settings['input_dropout']
            pair = cached_pair(observer, encoded, tick, device, missing)
            diagnostics = {}
            current = core.advance(self.memory, pair, encoded['dt'][tick], diagnostics=diagnostics)
            teacher = encoded['gold'][tick:tick+1].to(device)
            valid = encoded['valid'][tick:tick+1].to(device)
            anchors.append(state_anchor_loss(current, teacher, valid, diagnostics))
            correction_sum += diagnostics['correction_gain'].detach().sum()
            correction_count += diagnostics['correction_gain'].numel()
            records.append((len(encoded_rows), tick)); encoded_rows.append(encoded)
            memories.append(current)
            self.memory = current
            self.advance_cursor()
        bank = bank_from_rows(memories, records, encoded_rows, list(horizons),
                              'differentiable_online_window', 'online_tbptt', cpu=False)
        return bank, torch.stack(anchors).mean(), correction_sum/correction_count


def fit_shrink(bank, device):
    totals = torch.cat([bank['teacher_sum'].to(device), torch.tensor([bank['teacher_count']], device=device)])
    all_reduce(totals)
    mean = totals[:-1]/totals[-1].clamp_min(1)
    coefficients = []
    current = bank['states'].z.to(device).double()
    for column in range(len(bank['horizons'])):
        valid = bank['valid'][:, column].to(device)
        x = current[valid]-mean
        y = bank['targets'][:, column].to(device).double()[valid]-mean
        sums = all_reduce(torch.stack([(x*y).sum(), x.square().sum()]))
        coefficients.append((sums[0]/sums[1].clamp_min(1e-12)).clamp(0, 1))
    return mean.float(), torch.stack(coefficients).float()


@torch.no_grad()
def evaluate_bank(core, bank, mean, coefficients, device, batch_size):
    # Error sums/counts remain on GPU until the one final collective/copy.
    names = ['learned', 'hold', 'shrink', 'mean']
    h = len(bank['horizons'])
    sums = torch.zeros(4, 3, h, 2, device=device, dtype=torch.float64)
    for lo in range(0, len(bank['states'].fast), batch_size):
        indices = torch.arange(lo, min(lo+batch_size, len(bank['states'].fast)))
        state = memory_index(bank['states'], indices).to(device)
        outputs = core.forecast(state, bank['horizons'])
        truth = bank['targets'][indices].to(device)
        valid = bank['valid'][indices].to(device)
        domains = bank['domain'][indices].to(device)
        for column, predicted in enumerate(outputs):
            candidates = [predicted.z, state.z, mean+coefficients[column]*(state.z-mean), mean.expand_as(state.z)]
            for method, prediction in enumerate(candidates):
                errors = (prediction.double()-truth[:, column].double()).square().sum(-1)
                for domain in range(3):
                    mask = valid[:, column] & (domains == domain)[:, None]
                    sums[method, domain, column, 0] += (errors*mask).sum()
                    sums[method, domain, column, 1] += mask.sum()*truth.shape[-1]
    all_reduce(sums)
    values = sums.cpu()
    result = {'sums': values.tolist(), 'methods': {}}
    for index, name in enumerate(names):
        error, count = values[index, ..., 0], values[index, ..., 1]
        macro_h = []
        for column in range(h):
            mask = count[:, column] > 0
            if mask.any():
                macro_h.append((error[mask, column]/count[mask, column]).mean())
        result['methods'][name] = {'mse': float(error.sum()/count.sum()),
                                   'macro_mse': float(torch.stack(macro_h).mean()),
                                   'elements': int(count.sum())}
    hold = result['methods']['hold']['macro_mse']
    result['gain_over_hold'] = 1-result['methods']['learned']['macro_mse']/hold
    result['producer'] = bank['producer']
    result['current'] = None
    if bank.get('current'):
        keys = list(bank['current'])
        current = all_reduce(torch.tensor([bank['current'][k] for k in keys], device=device, dtype=torch.float64))
        result['current'] = dict(zip(keys, current.cpu().tolist()))
    return result


def deployment_gate(metrics, initial, settings):
    tolerance = settings['acceptance_tolerance']
    fixed = metrics['fixed']['methods']['learned']['macro_mse']
    old_fixed = initial['fixed']['methods']['learned']['macro_mse']
    checks = {
        'fixed_predictor_improved': fixed <= old_fixed*(1-settings['minimum_fixed_improvement']),
        'fixed_predictor_beats_hold': metrics['fixed']['gain_over_hold'] > 0,
        'live_predictor_beats_hold': metrics['live']['gain_over_hold'] > 0,
        'live_absolute_not_worse': metrics['live']['methods']['learned']['macro_mse'] <= initial['live']['methods']['learned']['macro_mse']*(1+tolerance),
        'gap_absolute_not_worse': metrics['gap']['methods']['learned']['macro_mse'] <= initial['gap']['methods']['learned']['macro_mse']*(1+tolerance),
        'fresh_deployment_producer': metrics['live']['producer'] == metrics['candidate_hash'],
        'correction_parameters_unchanged': metrics['correction_parameters_unchanged'],
    }
    # Neither posterior reconstruction nor a worsened hold denominator can
    # replace the two absolute forecast controls in this gate.
    score = .5*fixed/old_fixed + .5*metrics['live']['methods']['learned']['macro_mse']/initial['live']['methods']['learned']['macro_mse']
    return dict(passed=all(checks.values()), checks=checks, score=score)


@torch.no_grad()
def endpoint_metrics(observer, core, bank, device, batch_size):
    """True endpoint/query metrics, kept separate from raw affect MSE."""
    totals = torch.zeros(7, device=device, dtype=torch.float64)
    records = bank['endpoints']
    for lo in range(0, len(records), batch_size):
        rows = records[lo:lo+batch_size]
        indices = torch.tensor([row for row, _ in rows])
        state = memory_index(bank['states'], indices).to(device)
        seconds = torch.tensor([query['seconds'] for _, query in rows], device=device)
        state = core._propagate(state, seconds)
        roles = torch.tensor([query['role'] for _, query in rows], device=device)
        from emotion_ssm.train.dynamics_v3 import unit_state_readout
        values = observer.decode_affect(unit_state_readout(state.z[torch.arange(len(rows), device=device), roles]))
        labels = [query['label'] for _, query in rows]
        emotion = torch.tensor([int(l.get('emotion', -1)) for l in labels], device=device)
        valid = (emotion >= 0) & (emotion < 7)
        if valid.any():
            totals[0] += torch.nn.functional.cross_entropy(values['emotion_logits'][valid], emotion[valid], reduction='sum')
            totals[1] += valid.sum()
            totals[2] += (values['emotion_logits'][valid].argmax(-1) == emotion[valid]).sum()
        truth = torch.tensor([l.get('vad', [0., 0., 0.]) for l in labels], device=device)
        mask = torch.tensor([l.get('vad_mask', [False]*3) for l in labels], device=device)
        totals[3] += (values['vad'][mask]-truth[mask]).square().sum()
        totals[4] += mask.sum()
        mask = torch.tensor([bool(l.get('intensity_mask', False)) for l in labels], device=device)
        truth = torch.tensor([float(l.get('intensity', 0.)) for l in labels], device=device)
        totals[5] += (values['intensity'][mask]-truth[mask]).square().sum()
        totals[6] += mask.sum()
    numbers = all_reduce(totals).cpu().tolist()
    return dict(emotion_ce=numbers[0]/numbers[1] if numbers[1] else None,
                emotion_accuracy=numbers[2]/numbers[1] if numbers[1] else None,
                emotion_query_count=numbers[1], vad_mse=numbers[3]/numbers[4] if numbers[4] else None,
                vad_coordinate_count=numbers[4], intensity_mse=numbers[5]/numbers[6] if numbers[6] else None,
                intensity_query_count=numbers[6], protocol='true_endpoint_per_origin_horizon_query')


def run(config, execution_override=None):
    config = copy.deepcopy(config)
    settings = {**DEFAULTS, **config.get('staged', {})}
    if min(settings['bank_refresh_steps'], settings['bank_dialogues_per_domain'], settings['validation_dialogues_per_domain']) < 1:
        raise ValueError('Positive replay/validation budgets are required')
    device, rank, world = _distributed_device(config)
    torch.set_num_threads(int(config['train'].get('cpu_threads', 4)))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    seed = int(config['train']['seed'])
    random.seed(seed+rank); torch.manual_seed(seed+rank)
    output = Path(config['paths']['output']); output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    def emit(event, **values):
        data = dict(event=event, rank=rank, elapsed=time.monotonic()-started, **values)
        with (output/f'rank{rank}.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(data, ensure_ascii=False, allow_nan=False)+'\n')
        if rank == 0:
            print(json.dumps(data, ensure_ascii=False, allow_nan=False), flush=True)

    resume_path = config['paths'].get('resume')
    source = read_checkpoint(resume_path or config['paths']['dynamics_checkpoint'])
    if resume_path:
        if source['kind'] != 'dynamics_staged_v3' or source['config']['train'].get('staged_revision') != STAGED_DYNAMICS_REVISION:
            raise ValueError('Only this staged protocol can resume its optimizer')
        if source['provenance'] != manifest_provenance(config):
            raise ValueError('Resume data/provenance changed')
        if source['run_state']['world_size'] != world:
            raise ValueError('Exact resume requires the same rank budget/world size')
        config = copy.deepcopy(source['config']); settings = {**DEFAULTS, **config['staged']}
    observer = TokenObserver(source['construction']['observer']).to(device).eval()
    teacher = TokenObserver(source['construction']['observer']).to(device).eval()
    core = UnifiedEmotionStateCore.from_config(source['construction']['state']).to(device)
    models = dict(observer=observer, teacher=teacher, state=core)
    for name, model in models.items():
        model.load_state_dict(source['models'][name], strict=True)
    # Identical initial weights on every rank even if local RNG states differ.
    if dist.is_initialized():
        for model in models.values():
            for value in model.state_dict().values():
                dist.broadcast(value, 0)
    teacher.requires_grad_(False)
    execution = execution_override or config.get('execution', {}).get('mode', 'reference')
    if execution == 'compiled' and device.type != 'cuda':
        raise ValueError('Compiled production dynamics requires CUDA')
    core.configure_execution(execution)
    config['execution'] = dict(mode=execution, revision=EXECUTION_REVISION)
    synchronize = synchronize_gradients if execution == 'reference' else synchronize_gradients_batched
    parameter_partition(observer, core, 'calibration')
    locks = lock_snapshot(core)
    initial_models = cpu_models(models) if not resume_path else source['run_state']['initial_models']
    if resume_path:
        locks = source['run_state']['locks']
    config['train']['staged_revision'] = STAGED_DYNAMICS_REVISION
    config['staged'] = settings
    config['state'] = {k: v for k, v in core.get_config().items() if k != 'observation_dim'}
    config['observer'] = observer.construction()
    config['train']['max_steps'] = sum(n for _, n in stage_plan(settings))
    config['train']['amp'] = False
    write_config(output/'config.json', config) if rank == 0 else None
    cache = EncodedDialogueCache(observer, teacher, config['paths']['feature_cache'], device, settings['cache_batch'])
    training = DialogueCollection(config['data']['token_roots'], 'train')
    validation = DialogueCollection(config['data']['token_roots'], 'val')
    selected_val = source_balanced_indices(validation, settings['validation_dialogues_per_domain'], seed+17011)
    local_val = selected_val[rank::world]
    cursor = OnlineDialogueCursor(training, cache, seed, rank, world)
    plan = stage_plan(settings)
    budget = int(config['train']['global_chunks_per_step'])
    if budget % world or budget // world > 32:
        raise ValueError('Use equal rank budgets and at most 32 chronological packets per rank')
    quota = budget // world
    if settings['compile_flow']:
        if device.type != 'cuda' or core.adaptive_flow is None:
            raise ValueError('Compiled flow requires adaptive CUDA dynamics')
        core.adaptive_flow.coefficients = torch.compile(core.adaptive_flow.coefficients, fullgraph=True)
    emit('initialized', protocol=STAGED_DYNAMICS_REVISION, plan=plan, world_size=world,
         execution=config['execution'],
         precision='fp32_no_tf32', global_origins_or_new_packets=budget,
         cache_fingerprint=cache.fingerprint, train_dialogues=len(training),
         validation_indices=selected_val, initial_checkpoint_step=source['global_step'])

    def bank(collection, indices, label, gap=0):
        last_progress = [time.monotonic()]
        def progress(done, total, origins):
            if done == total or time.monotonic()-last_progress[0] > 30:
                emit('bank_progress', label=label, dialogues=done, total=total, origins=origins)
                last_progress[0] = time.monotonic()
        return build_origin_bank(observer, core, cache, collection, indices, config['train']['forecast_seconds'],
                                 settings['origin_stride'], gap, progress)

    reference_path = output/f'reference_val_rank{rank}.pt'
    if resume_path and reference_path.exists():
        reference = torch.load(reference_path, map_location='cpu', weights_only=False)
    else:
        if resume_path:
            saved = cpu_models(models)
            for name in models:
                models[name].load_state_dict(initial_models[name])
        reference = bank(validation, local_val, 'immutable_reference_validation')
        torch.save(reference, reference_path)
        if resume_path:
            for name in models:
                models[name].load_state_dict(saved[name])
    expected_reference = weight_digest({'observer': initial_models['observer'], 'state': initial_models['state']})
    if reference['producer'] != expected_reference:
        raise ValueError('Fixed validation reference belongs to another initial model')
    global_step = 0 if not resume_path else source['global_step']
    best_score = math.inf if not resume_path else source['run_state']['best_score']
    initial_metrics = None if not resume_path else source['run_state']['initial_metrics']
    resume_block = 0 if not resume_path else source['run_state']['block']
    resume_step = 0 if not resume_path else source['run_state']['block_step']
    if resume_path:
        cursor.load(source['run_state']['ranks'][rank]['cursor'], device)

    def validate(train_bank, block, phase):
        assert_locked(core, locks)
        candidate_hash = weight_digest({'observer': observer, 'state': core})
        live = bank(validation, local_val, f'live_validation_step{global_step}')
        gap = bank(validation, local_val, f'gap_validation_step{global_step}', settings['block_gap_seconds'])
        mean, shrink = fit_shrink(train_bank, device)
        metrics = {'candidate_hash': candidate_hash, 'correction_parameters_unchanged': True,
                   'step': global_step, 'phase': phase, 'block': block}
        for label, value in [('fixed', reference), ('live', live), ('gap', gap)]:
            metrics[label] = evaluate_bank(core, value, mean, shrink, device, settings['prediction_batch'])
        metrics['live']['semantic'] = endpoint_metrics(observer, core, live, device, settings['prediction_batch'])
        metrics['bank_producer'] = train_bank['producer']
        delta = live['states'].z-reference['states'].z
        drift = all_reduce(torch.tensor([float(delta.double().square().sum()), delta.numel()], device=device, dtype=torch.float64))
        metrics['live_origin_drift_from_initial_rms'] = float((drift[0]/drift[1]).sqrt())
        if initial_metrics is not None:
            metrics['acceptance'] = deployment_gate(metrics, initial_metrics, settings)
        if rank == 0:
            (output/'validation.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
        emit('validation', metrics=metrics)
        return metrics

    for block_index, (phase, length) in enumerate(plan):
        if block_index < resume_block:
            continue
        parameters = parameter_partition(observer, core, phase)
        lr = settings['calibration_lr'] if phase == 'calibration' else settings['joint_lr'] if phase == 'joint' else settings['prediction_lr']
        if phase == 'readapt':
            lr = settings['joint_prediction_lr']
        optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=settings['weight_decay'])
        continuing = bool(resume_path and block_index == resume_block)
        begin = resume_step if continuing else 0
        bank_indices = source_balanced_indices(training, settings['bank_dialogues_per_domain'], seed+9001, block_index)
        local_train = bank_indices[rank::world]
        bank_path = output/f'origins_block{block_index}_rank{rank}.pt'
        producer_models = cpu_models(models)
        if continuing:
            producer_models = source['run_state']['bank_producer_models']
            if bank_path.exists():
                train_bank = torch.load(bank_path, map_location='cpu', weights_only=False)
            else:
                current_models = cpu_models(models)
                for name in models:
                    models[name].load_state_dict(producer_models[name])
                train_bank = bank(training, local_train, f'restored_bank_block{block_index}')
                for name in models:
                    models[name].load_state_dict(current_models[name])
                torch.save(train_bank, bank_path)
            expected = weight_digest({'observer': producer_models['observer'], 'state': producer_models['state']})
            if train_bank['producer'] != expected:
                raise ValueError('Resume origin bank producer differs from the saved snapshot')
            optimizer.load_state_dict(source['optimizer'])
            restore_rng_state(source['run_state']['ranks'][rank]['rng'])
        else:
            cursor.invalidate_history()
            train_bank = bank(training, local_train, f'fresh_train_bank_block{block_index}')
            torch.save(train_bank, bank_path)
        emit('phase_started', phase=phase, block=block_index, steps=length,
             bank_producer=train_bank['producer'], train_indices=bank_indices,
             bank_origins=len(train_bank['states'].fast), learning_rate=lr)
        if initial_metrics is None:
            initial_metrics = validate(train_bank, block_index, 'initial')
        step_seconds = []
        for block_step in range(begin+1, length+1):
            then = time.monotonic(); optimizer.zero_grad(set_to_none=True)
            if phase in {'calibration', 'joint'}:
                active, anchor, gain = cursor.window(observer, core, quota, config['train']['forecast_seconds'], settings)
                indices = torch.arange(quota)
            else:
                active = train_bank
                indices = torch.randint(len(active['states'].fast), (quota,))
                anchor = torch.zeros((), device=device); gain = torch.zeros((), device=device)
            future, _ = prediction_loss(core, active, indices, device, observer, settings['endpoint_label_weight'])
            loss = future + settings['state_anchor_weight']*anchor
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite staged loss; optimizer was not updated')
            loss.backward()
            # Gradients of rank-local means require equal rank weighting.
            synchronize(parameters, 1, device)
            norm = torch.nn.utils.clip_grad_norm_(parameters, settings['clip_grad'])
            optimizer.step()
            if cursor.memory is not None:
                cursor.memory = cursor.memory.detach()
            global_step += 1
            elapsed_step = time.monotonic()-then; step_seconds.append(elapsed_step)
            if global_step % settings['log_every'] == 0 or block_step == 1:
                packed = all_reduce(torch.stack([loss.detach(), future.detach(), anchor.detach(), gain.detach()]))/world
                values = packed.cpu().tolist()
                emit('train', step=global_step, total_steps=config['train']['max_steps'], phase=phase,
                     block_step=block_step, phase_steps=length, loss=values[0], future_loss=values[1],
                     state_anchor_loss=values[2], correction_gain=values[3], gradient_norm=float(norm),
                     mean_step_seconds=sum(step_seconds)/len(step_seconds),
                     cache_hits=cache.hits, cache_misses=cache.misses)
                step_seconds = []
                if rank == 0:
                    (output/'training_status.json').write_text(json.dumps(dict(status='running', phase=phase,
                        step=global_step, max_steps=config['train']['max_steps'], block=block_index)), encoding='utf-8')
            should_validate = global_step % settings['validation_every'] == 0 or block_step == length
            improved = False
            metrics = {}
            if should_validate:
                metrics = validate(train_bank, block_index, phase)
                gate = metrics['acceptance']
                # Calibration is intermediate; only a trained and freshly
                # replayed propagation checkpoint may be deployment-ready.
                improved = phase in {'fixed', 'readapt'} and gate['passed'] and gate['score'] < best_score
                if improved:
                    best_score = gate['score']
            if should_validate or global_step % settings['checkpoint_every'] == 0:
                assert_locked(core, locks)
                local = dict(cursor=cursor.state_dict(), rng=capture_rng_state())
                ranks = [local]
                if dist.is_initialized():
                    ranks = [None]*world; dist.all_gather_object(ranks, local)
                if rank == 0:
                    state = dict(block=block_index, block_step=block_step, best_score=best_score,
                                 ranks=ranks, initial_models=initial_models, initial_metrics=initial_metrics,
                                 locks=locks, bank_producer_models=producer_models, world_size=world,
                                 deployment_ready=bool(improved), prediction_origin_protocol='candidate_full_dialogue_replay')
                    save_checkpoint(output/'last.pt', models, config, source['construction'], 'dynamics_staged_v3',
                                    global_step, optimizer=optimizer, metrics=metrics, run_state=state)
                    if improved:
                        save_checkpoint(output/'best.pt', models, config, source['construction'], 'dynamics_staged_v3',
                                        global_step, optimizer=optimizer, metrics=metrics, run_state=state)
                if dist.is_initialized():
                    dist.barrier()
    if rank == 0:
        (output/'training_status.json').write_text(json.dumps(dict(status='complete', step=global_step,
            max_steps=config['train']['max_steps'], deployment_checkpoint_available=(output/'best.pt').exists())), encoding='utf-8')
    emit('complete', step=global_step, deployment_checkpoint_available=(output/'best.pt').exists())
    return dict(step=global_step, output=str(output))


def main():
    torch.set_num_interop_threads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--resume', default='')
    parser.add_argument('--execution', choices=['reference', 'optimized', 'compiled'], default=None)
    args = parser.parse_args()
    config = read_config(args.config)
    if args.resume:
        config['paths']['resume'] = args.resume
    try:
        run(config, execution_override=args.execution)
    except Exception as error:
        if int(os.environ.get('RANK', 0)) == 0:
            output = Path(config['paths']['output']); output.mkdir(parents=True, exist_ok=True)
            (output/'training_status.json').write_text(json.dumps(dict(status='failed', error=repr(error))), encoding='utf-8')
        raise


if __name__ == '__main__':
    main()
