"""Two-rank checks and four real-data staged steps with compiled execution."""
import argparse
import copy
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from emotion_ssm.train.dynamics_staged_v3 import run
from emotion_ssm.train.dynamics_v3 import synchronize_gradients, _distributed_device
from emotion_ssm.utils.dynamics_execution import synchronize_gradients_batched


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--run-dir', required=True)
    args = parser.parse_args(); root = Path(args.run_dir)
    torch.set_num_interop_threads(1)
    ck = torch.load(root/'execution_benchmark.source.pt', map_location='cpu', weights_only=False)
    config = copy.deepcopy(ck['config'])
    config['paths'].update(output=str(root/'execution_smoke'), resume='', dynamics_checkpoint=str(root/'execution_benchmark.source.pt'))
    config['staged'].update(calibration_steps=1, fixed_steps=1, bank_refresh_steps=1, joint_rounds=1,
        joint_steps_per_round=1, readapt_steps_per_round=1, bank_dialogues_per_domain=1,
        validation_dialogues_per_domain=1, validation_every=1, checkpoint_every=1, log_every=1)
    config['train']['cpu_threads'] = 2
    device, rank, world = _distributed_device(config)
    reference = [torch.nn.Parameter(torch.zeros(2, 3, device=device)), torch.nn.Parameter(torch.zeros(7, device=device)),
                 torch.nn.Parameter(torch.zeros(1, device=device))]
    reference[0].grad = torch.arange(6, device=device, dtype=torch.float32).reshape(2, 3)+rank
    reference[1].grad = torch.arange(7, device=device, dtype=torch.float32)*(rank+1)
    candidate = [torch.nn.Parameter(p.detach().clone()) for p in reference]
    for before, after in zip(reference, candidate):
        after.grad = None if before.grad is None else before.grad.clone()
    synchronize_gradients(reference, 1, device); synchronize_gradients_batched(candidate, 1, device)
    for before, after in zip(reference, candidate):
        torch.testing.assert_close(before.grad, after.grad, rtol=0, atol=0)
    print(json.dumps({'event': 'grouped_ddp_gradient_exact', 'rank': rank, 'world': world}), flush=True)
    result = run(config, execution_override='compiled')
    if rank == 0:
        (root/'execution_smoke_result.json').write_text(json.dumps(dict(passed=True, grouped_ddp_gradient_exact=True, **result), indent=2))
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
