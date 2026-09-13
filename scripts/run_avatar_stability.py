"""Replace the authorized Avatar run after tests; verify exact DDP resume, then retrain."""
from __future__ import annotations

import argparse
import copy
from dataclasses import fields, is_dataclass
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np
import torch

from emotion_ssm.config_v3 import write_config
from emotion_ssm.train.generation_stability import STABILITY_PROTOCOL
from emotion_ssm.train.staged_dynamics_support import weight_digest

GPUS = ['GPU-5b497823-4a84-bde7-5670-2172ee96245d', 'GPU-9117039b-5194-5d46-d4a7-088ff06ce552']
CONDITION_HASH = 'b1f656f069b3b86cdb18f9c59021d84746b606c84a93dd33d5e76d1283db860e'


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def same(a, b, path='root'):
    if torch.is_tensor(a):
        if not torch.equal(a, b):
            raise ValueError('Resume tensor differs: ' + path)
    elif isinstance(a, np.ndarray):
        if not np.array_equal(a, b):
            raise ValueError('Resume array differs: ' + path)
    elif is_dataclass(a):
        for item in fields(a):
            same(getattr(a, item.name), getattr(b, item.name), path + '.' + item.name)
    elif isinstance(a, dict):
        if a.keys() != b.keys():
            raise ValueError('Resume keys differ: ' + path)
        for key in a:
            same(a[key], b[key], path + '.' + str(key))
    elif isinstance(a, (list, tuple)):
        if len(a) != len(b):
            raise ValueError('Resume sequence length differs: ' + path)
        for i, (left, right) in enumerate(zip(a, b)):
            same(left, right, path + '.' + str(i))
    elif a != b:
        raise ValueError('Resume value differs: ' + path)


def condition_hash(payload):
    system = payload['models']['system']
    condition = {name: {key[len(prefix):]: value for key, value in system.items() if key.startswith(prefix)}
        for name, prefix in [('observer', 'observer.'), ('teacher', 'teacher.'), ('state', 'state_model.')]}
    return weight_digest(condition)


def stop_authorized_run(old, output):
    """Match owned process command arguments before sending any signal."""
    old = Path(old).resolve()
    if old.name != 'avatar_old9750_20260911_gpu23':
        raise ValueError('Replacement is restricted to the previously diagnosed Avatar run')
    targets = []
    for path in Path('/proc').iterdir():
        if not path.name.isdigit() or path.stat().st_uid != os.getuid():
            continue
        try:
            argv = path.joinpath('cmdline').read_bytes().decode().split('\0')
        except (OSError, UnicodeDecodeError):
            continue
        linked = any(value.startswith(str(old) + '/') or value == str(old) for value in argv)
        training = ('emotion_ssm.train.generation_v3' in argv or
                    any(value.endswith('/resume_avatar_throughput.py') or value.endswith('/run_avatar_old9750.py')
                        for value in argv))
        if linked and training:
            targets.append(dict(pid=int(path.name), argv=argv))
    if not targets:
        raise ValueError('No matching old Avatar process found; inspect server state before replacement')
    # Supervisors precede torchrun, which precedes its workers. No broad pkill.
    targets.sort(key=lambda row: ('emotion_ssm.train.generation_v3' in row['argv'], row['pid']))
    write_json(output / 'replacement_receipt.json', dict(time=time.time(), old_run=str(old), targets=targets,
        best_preserved=str(old / 'formal/best.pt'), last_preserved=str(old / 'formal/last.pt')))
    for row in targets:
        try:
            os.kill(row['pid'], signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline:
        alive = []
        for row in targets:
            path = Path('/proc') / str(row['pid']) / 'stat'
            if path.exists() and path.read_text().split(') ')[1][0] != 'Z':
                alive.append(row['pid'])
        if not alive:
            return
        time.sleep(.5)
    raise RuntimeError('Old training did not exit cleanly; new GPU training not launched: ' + str(alive))


def run(args):
    root, output = Path(args.project_root).resolve(), Path(args.output).resolve()
    old = root / 'runs/avatar_old9750_20260911_gpu23'
    code = Path(__file__).resolve().parents[1]
    if output.parent != root / 'runs' or code.parent != output:
        raise ValueError('Training must use its isolated project/runs code snapshot')
    if (output / 'pipeline_status.json').exists():
        raise ValueError('Refusing a duplicate training pipeline')
    torch.set_num_threads(2)
    payload = torch.load(old / 'formal/best.pt', map_location='cpu', weights_only=False, mmap=True)
    if payload['global_step'] != 1000 or condition_hash(payload) != CONDITION_HASH:
        raise ValueError('Source Avatar no longer matches step1000 / frozen old9750 condition')
    initial = output / 'initial_avatar_step001000.pt'
    # Preserve a complete inference checkpoint without copying the old optimizer.
    snapshot = dict(payload, optimizer=None, scaler=None, run_state={})
    torch.save(snapshot, initial)
    cfg = copy.deepcopy(payload['config'])
    cfg['generation'].update(train_observer=False, train_state=False, dynamics_execution='optimized')
    cfg['paths'].update(avatar_initialization=str(initial), resume='', output=str(output / 'formal'))
    cfg['train'].update(max_steps=30000, global_chunks_per_step=32, tbptt_seconds=32, seed=6666,
        lr=3e-5, observer_lr=1e-5, state_lr=1e-4, validate_every=1000, validation_max_dialogues=0,
        log_every=10, coordinate_weight=0., future_weight=0., masked_weight=0.,
        experiment_scope='old9750_frozen_condition_generator_stability_retrain')
    selection = json.loads((output / 'diagnostic_selection.json').read_text())
    groups = {name: dict(split='val' if name == 'val' else 'train',
                        names=[value.removeprefix('dualtalk:') for value in selection[key]])
              for name, key in [('reference_early_train', 'early_train'), ('reference_late_train', 'recent_train'), ('val', 'val')]}
    cfg['generation_stability'] = dict(protocol=STABILITY_PROTOCOL, conversations_per_rank=4,
        blocks_per_conversation=4, speech_lr=1e-5, film_lr=1e-4, freeze_speech=False,
        mask_time_prob=.05, mask_time_length=10, cosine_steps=10000, warmup_steps=200,
        minimum_lr_ratio=.1, diagnostic_every=250, diagnostic_max_blocks=24,
        diagnostic_groups=groups, retain_initial_candidate=True)
    cfg['generation_execution'] = dict(defer_metrics=True, prefetch_batches=1)
    write_config(output / 'formal_config.json', cfg)
    write_json(output / 'experiment_protocol.json', dict(protocol=STABILITY_PROTOCOL,
        initialization_avatar_step=1000, frozen_dynamics_step=9750, condition_hash=CONDITION_HASH,
        old_optimizer_restored=False, initial_validation=payload['metrics'], physical_gpus=[2, 3],
        valid_blocks_per_step=32, conversations_per_rank=4, blocks_per_conversation=4,
        learning_rates=dict(generator=3e-5, speech=1e-5, film=1e-4),
        warmup_steps=200, cosine_steps=10000, minimum_lr_ratio=.1,
        first_diagnostic_steps=1000, total_new_optimizer_steps=30000,
        probe_selection='Fixed old-run early/late training panels and validation; not rolling recent training',
        validation_selection='Full validation generation_total, including the retained initialization candidate',
        prior_best_preserved=str(old / 'formal/best.pt'), start_time=time.time()))
    del payload, snapshot
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(GPUS), PYTHONPATH=str(code),
        OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', PYTHONUNBUFFERED='1',
        PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        TOKENIZERS_PARALLELISM='false', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    stop_authorized_run(old, output)

    def stage(name, path, stop_after=None):
        command = [sys.executable, '-u', '-B', '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2',
                   '--module', 'emotion_ssm.train.generation_v3', '--config', str(path)]
        if stop_after is not None:
            command += ['--stop-after', str(stop_after)]
        with (output / (name + '.log')).open('wb') as log:
            child = subprocess.Popen(command, cwd=code, env=env, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT)
            write_json(output / 'pipeline_status.json', dict(status='running', stage=name, pid=os.getpid(),
                child_pid=child.pid, command=command, time=time.time(), log=str(output / (name + '.log'))))
            print(json.dumps(dict(stage=name, pid=child.pid)), flush=True)
            result = child.wait()
        if result:
            write_json(output / 'pipeline_status.json', dict(status='failed', stage=name, returncode=result,
                log=str(output / (name + '.log'))))
            raise RuntimeError('Training stage failed: ' + name)

    smoke = copy.deepcopy(cfg)
    smoke['train'].update(max_steps=4, validate_every=2, validation_max_dialogues=2, log_every=1)
    smoke['generation_stability'].update(diagnostic_groups={}, retain_initial_candidate=False)
    smoke['paths']['output'] = str(output / 'smoke_reference')
    write_config(output / 'smoke_reference.json', smoke)
    stage('smoke_reference', output / 'smoke_reference.json')
    smoke['paths']['output'] = str(output / 'smoke_resume')
    write_config(output / 'smoke_resume.json', smoke)
    stage('smoke_interrupted', output / 'smoke_resume.json', stop_after=2)
    smoke['paths']['resume'] = str(output / 'smoke_resume/last.pt')
    write_config(output / 'smoke_resume_config.json', smoke)
    stage('smoke_restored', output / 'smoke_resume_config.json')
    reference = torch.load(output / 'smoke_reference/last.pt', map_location='cpu', weights_only=False, mmap=True)
    restored = torch.load(output / 'smoke_resume/last.pt', map_location='cpu', weights_only=False, mmap=True)
    for field in ('models', 'optimizer', 'scaler', 'metrics'):
        same(reference[field], restored[field], field)
    same(reference['run_state']['ranks'], restored['run_state']['ranks'], 'rank_sampling_state_and_rng')
    if reference['global_step'] != 4 or condition_hash(reference) != CONDITION_HASH:
        raise ValueError('Incomplete training smoke or frozen condition changed')
    rows = [json.loads(line) for line in (output / 'smoke_reference/train_metrics.jsonl').read_text().splitlines()]
    for row in rows:
        if row['global_valid_blocks'] != 32 or row['sources_per_step'] < 8:
            raise ValueError('Smoke sampling coverage/budget failed')
        if row['observer_grad_norm'] or row['state_grad_norm'] or row['generator_grad_norm'] <= 0:
            raise ValueError('Smoke training permissions failed')
    write_json(output / 'smoke_gate.json', dict(passed=True, world_size=2, steps=4,
        comparison='4 uninterrupted updates == 2 updates + checkpoint restore + 2 updates; bitwise',
        verified=['model', 'optimizer', 'scaler', 'metrics', 'all conversation states', 'committed sampling cursor', 'RNG'],
        condition_hash=CONDITION_HASH, minimum_sources=min(row['sources_per_step'] for row in rows),
        mean_step_seconds=sum(row['step_seconds'] for row in rows[1:]) / max(1, len(rows)-1)))
    del reference, restored
    stage('formal', output / 'formal_config.json')
    write_json(output / 'pipeline_status.json', dict(status='complete', steps=30000, time=time.time(),
        best_checkpoint=str(output / 'formal/best.pt')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', required=True)
    parser.add_argument('--output', required=True)
    run(parser.parse_args())
