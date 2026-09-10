"""Measure identical fixed-origin forecast work, serial versus batched CUDA.

Both paths use the same targets, masks, reductions, parameters and precision.
The report is a hot-path timing, not a promised whole-training speedup.
"""
import argparse
import copy
import json
import statistics
import time
from pathlib import Path

import torch

from emotion_ssm.config_v3 import read_config
from emotion_ssm.models.state_core import UnifiedEmotionStateCore
from emotion_ssm.models.token_observer import TokenObserver
from emotion_ssm.train.dynamics_v3 import DialogueCollection
from emotion_ssm.train.staged_dynamics_support import (
    EncodedDialogueCache, build_origin_bank, memory_index, parameter_partition,
)
from emotion_ssm.utils.checkpoint_v3 import read_checkpoint


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True); args = parser.parse_args()
    config = read_config(args.config)
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda:0')
    ck = read_checkpoint(config['paths']['dynamics_checkpoint'])
    observer = TokenObserver(ck['construction']['observer']).to(device).eval()
    teacher = TokenObserver(ck['construction']['observer']).to(device).eval()
    core = UnifiedEmotionStateCore.from_config(ck['construction']['state']).to(device)
    for name, model in [('observer', observer), ('teacher', teacher), ('state', core)]:
        model.load_state_dict(ck['models'][name])
    parameters = parameter_partition(observer, core, 'fixed')
    cache = EncodedDialogueCache(observer, teacher, config['paths']['feature_cache'], device, 32)
    data = DialogueCollection(config['data']['token_roots'], 'train')
    chosen = [next(i for i, (d, _) in enumerate(data.index) if d == domain) for domain in range(3)]
    began = time.perf_counter()
    bank = build_origin_bank(observer, core, cache, data, chosen, config['train']['forecast_seconds'], stride=1)
    extraction_seconds = time.perf_counter()-began
    n = min(16, len(bank['states'].fast))
    indices = torch.linspace(0, len(bank['states'].fast)-1, n).long()
    origins = memory_index(bank['states'], indices).to(device)
    target = bank['targets'][indices].to(device); mask = bank['valid'][indices].to(device)
    denominators = mask.sum((0, 2)).clamp_min(1)

    def compute(batched):
        if batched:
            states = core.forecast(origins, bank['horizons'])
            forecasts = torch.stack([s.z for s in states], 1)
        else:
            forecasts = torch.cat([torch.stack([s.z for s in core.forecast(memory_index(origins, slice(i, i+1)), bank['horizons'])], 1)
                                   for i in range(n)])
        error = (forecasts-target).square().sum(-1)
        loss = ((error*mask).sum((0, 2))/denominators).mean()
        return loss, forecasts

    report = {'origins': n, 'horizons': bank['horizons'], 'precision': 'fp32_no_tf32',
              'feature_and_origin_preparation_seconds': extraction_seconds, 'timings': {}}
    previous_grad = previous_output = None
    for name, batched in [('serial', False), ('batched', True)]:
        timings = []
        for repeat in range(3):
            core.zero_grad(set_to_none=True)
            torch.cuda.synchronize(); start = time.perf_counter()
            loss, forecasts = compute(batched); loss.backward()
            torch.cuda.synchronize(); seconds = time.perf_counter()-start
            if repeat:
                timings.append(seconds)
        gradients = torch.cat([torch.zeros_like(p).flatten() if p.grad is None else p.grad.flatten() for p in parameters])
        if previous_grad is not None:
            torch.testing.assert_close(forecasts, previous_output, rtol=5e-5, atol=3e-6)
            torch.testing.assert_close(gradients, previous_grad, rtol=3e-4, atol=2e-5)
            report['max_forward_difference'] = float((forecasts-previous_output).abs().max())
            report['max_gradient_difference'] = float((gradients-previous_grad).abs().max())
        previous_grad, previous_output = gradients.detach().clone(), forecasts.detach().clone()
        report['timings'][name] = {'seconds': timings, 'median_seconds': statistics.median(timings)}
        print(json.dumps({'stage': name, **report['timings'][name]}), flush=True)
    report['forecast_forward_backward_speedup'] = report['timings']['serial']['median_seconds']/report['timings']['batched']['median_seconds']
    report['equal_work_and_gradients_checked'] = True
    Path(args.output).write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
