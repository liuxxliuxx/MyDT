"""Compare execution only, replaying the same complete Avatar checkpoint."""
import argparse
import copy
from dataclasses import fields, is_dataclass
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
import torch.distributed as dist

from emotion_ssm.train.generation_v3 import (
    GenerationTokenDataset, SegmentCursor, configure_generation_from_config,
    train_segment, OPTIMIZATION_PROTOCOL, EXECUTION_PROTOCOL,
)
from emotion_ssm.train.generation_prefetch import PrefetchSegmentCursor
from emotion_ssm.utils.checkpoint import capture_rng_state, restore_rng_state
from emotion_ssm.utils.checkpoint_v3 import load_avatar, read_checkpoint, save_checkpoint
from emotion_ssm.utils.distributed import init_distributed


MODES = {'reference': (False, 0), 'metrics': (True, 0), 'prefetch': (False, 1), 'combined': (True, 1)}


def digest(value):
    h = hashlib.sha256()
    def add(item):
        if torch.is_tensor(item):
            item = item.detach().cpu().contiguous()
            h.update(str((item.dtype, tuple(item.shape))).encode())
            h.update(item.numpy().tobytes())
        elif isinstance(item, np.ndarray):
            h.update(str((item.dtype, item.shape)).encode()); h.update(item.tobytes())
        elif is_dataclass(item):
            add({field.name: getattr(item, field.name) for field in fields(item)})
        elif isinstance(item, dict):
            for key in sorted(item, key=repr):
                add(key); add(item[key])
        elif isinstance(item, (list, tuple)):
            for part in item: add(part)
        elif isinstance(item, (set, frozenset)):
            add(sorted(item, key=repr))
        else:
            h.update(repr(item).encode())
        h.update(b'\0')
    add(value)
    return h.hexdigest()


def run(args):
    source = read_checkpoint(args.checkpoint)
    cfg = copy.deepcopy(source['config'])
    train = cfg['train']
    context = init_distributed(train['device'], train['seed'], train['deterministic'])
    device = context.device
    torch.set_num_threads(int(train.get('cpu_threads', 2)))
    # CPU construction avoids temporarily putting the fixed raw extractor on GPU.
    model, cfg, source = load_avatar(source, 'cpu')
    features = model.features
    model.features = None
    model.to(device)
    model.features = features
    parameters = configure_generation_from_config(model, cfg)
    model.state_model.configure_execution(cfg['generation'].get('dynamics_execution', 'reference'))
    if any(p.requires_grad for m in (model.observer, model.state_model) for p in m.parameters()):
        raise ValueError('This benchmark expects the frozen old9750 upstream')
    if args.gradients_only:
        optimizer = torch.optim.SGD(parameters, lr=0.)
    else:
        optimizer = torch.optim.AdamW([dict(params=parameters, lr=train['lr'], initial_lr=train['lr'], name='generator')],
                                      weight_decay=train['weight_decay'], foreach=False)
        optimizer.load_state_dict(source['optimizer'])
    scaler = torch.amp.GradScaler('cuda', enabled=train['amp'])
    if source.get('scaler'):
        scaler.load_state_dict(source['scaler'])
    saved = source['run_state']['ranks'][context.rank]
    start = source['global_step']
    source_run_state = source['run_state']
    source_metrics = source['metrics']
    del source
    dataset = GenerationTokenDataset(cfg['data']['dualtalk_tokens'], cfg['data']['dualtalk_raw'], 'train')
    count = train['global_chunks_per_step'] // context.world_size
    if count * context.world_size != train['global_chunks_per_step']:
        raise ValueError('Wrong world size for this valid-block budget')
    cursor = SegmentCursor(dataset, train['seed'], context.rank, context.world_size, saved['packets_seen'])
    defer, prefetch = MODES[args.mode]
    if prefetch:
        cursor = PrefetchSegmentCursor(cursor, count)
    state = saved['stream_state'].to(device)
    restore_rng_state(saved['rng'])
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    rows, input_hashes = [], []
    context.barrier()
    began = time.perf_counter()
    try:
        for index in range(args.steps):
            torch.cuda.synchronize(device)
            step_began = time.perf_counter()
            packets = cursor.take_valid(count)
            data_wait = time.perf_counter() - step_began
            # Hash on CPU, before any inputs can be mutated, outside compute timing.
            hash_began = time.perf_counter()
            input_hashes.append(digest(packets))
            hash_seconds = time.perf_counter() - hash_began
            if args.gradients_only:
                optimizer.zero_grad(set_to_none=True)
            state, metrics = train_segment(model, packets, optimizer, state=state, device=device,
                tbptt_steps=train['tbptt_seconds'], scaler=scaler, amp=train['amp'],
                grad_clip=train['clip_grad'], defer_metrics=defer, optimize=not args.gradients_only)
            if args.gradients_only:
                scaler.update()
            else:
                for group in optimizer.param_groups:
                    group['lr'] = group['initial_lr'] * .5 * (1 + math.cos(math.pi * (start+index+1) / train['max_steps']))
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter()-step_began-hash_seconds
            times = torch.tensor([elapsed, data_wait], device=device, dtype=torch.float64)
            if context.enabled:
                dist.all_reduce(times, op=dist.ReduceOp.MAX)
            row = dict(step=start+index+1, benchmark_index=index+1, mode=args.mode,
                       step_seconds=float(times[0]), data_wait_seconds=float(times[1]), metrics=metrics)
            rows.append(row)
            if context.is_main:
                with (out/'steps.jsonl').open('a') as handle:
                    handle.write(json.dumps(row)+'\n')
                print(json.dumps(row), flush=True)
        state_rng = capture_rng_state()
        final = dict(rank=context.rank, inputs=input_hashes, seen=cursor.seen,
            model_hash=digest(model.state_dict()), gradient_hash=digest([p.grad for p in parameters]),
            optimizer_hash=digest(optimizer.state_dict()), scaler_hash=digest(scaler.state_dict()),
            stream_hash=digest(state), rng_hash=digest(state_rng))
        all_final = [final]
        if context.enabled:
            all_final = [None] * context.world_size
            dist.all_gather_object(all_final, final)
        if args.save_continuation:
            if args.gradients_only:
                raise ValueError('A gradients-only probe cannot become a training checkpoint')
            local = dict(packets_seen=cursor.seen, stream_state=state.detach().to('cpu'), rng=state_rng)
            ranks = [local]
            if context.enabled:
                ranks = [None] * context.world_size
                dist.all_gather_object(ranks, local)
            run_state = dict(source_run_state, ranks=ranks, world_size=context.world_size,
                             optimizer_budget=dict(global_new_blocks=train['global_chunks_per_step'], step=start+args.steps))
            cfg['generation_execution'] = dict(defer_metrics=defer, prefetch_batches=prefetch)
            if context.is_main:
                save_checkpoint(out/'continuation.pt', {'system':model}, cfg, model.construction_info,
                    'streaming_avatar_v3', step=start+args.steps, optimizer=optimizer, metrics=source_metrics,
                    run_state=run_state, scaler=scaler)
        if context.is_main:
            selected = rows[min(args.warmup, len(rows)-1):]
            report = dict(mode=args.mode, gradients_only=args.gradients_only, checkpoint=args.checkpoint,
                source_step=start, steps=args.steps, warmup_excluded=args.warmup,
                mean_step_seconds=sum(row['step_seconds'] for row in selected)/len(selected),
                mean_data_wait_seconds=sum(row['data_wait_seconds'] for row in selected)/len(selected),
                total_wall_seconds=time.perf_counter()-began, ranks=all_final,
                rows=rows, peak_reserved_mib=torch.cuda.max_memory_reserved(device)/2**20)
            (out/'report.json').write_text(json.dumps(report, indent=2))
            print(json.dumps({k:v for k,v in report.items() if k not in ('ranks','rows')}), flush=True)
    finally:
        if hasattr(cursor, 'close'):
            cursor.close()
    context.barrier()
    if context.enabled:
        dist.destroy_process_group()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--mode', choices=MODES, required=True)
    p.add_argument('--steps', type=int, default=20)
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--gradients-only', action='store_true')
    p.add_argument('--save-continuation', action='store_true')
    run(p.parse_args())
