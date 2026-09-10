"""Execution optimizations which preserve the staged training objective."""
from collections import defaultdict

import torch
import torch.distributed as dist

EXECUTION_REVISION = 'staged-execution-v1'


def synchronize_gradients_batched(parameters, new_chunks, device):
    """One reduction per dtype/device instead of one per parameter tensor."""
    count = torch.tensor(float(new_chunks), device=device)
    distributed = dist.is_initialized()
    if distributed:
        dist.all_reduce(count, op=dist.ReduceOp.SUM)
    groups = defaultdict(list)
    for parameter in parameters:
        if parameter.grad is None:
            if not distributed:
                continue
            parameter.grad = torch.zeros_like(parameter)
        groups[(parameter.device, parameter.dtype)].append(parameter)
    for values in groups.values():
        flat = torch.cat([parameter.grad.reshape(-1) for parameter in values])
        if distributed:
            dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        flat.div_(count.clamp_min(1))
        if not torch.isfinite(flat).all():
            raise FloatingPointError('Non-finite global dynamics gradient; no optimizer update applied')
        offset = 0
        for parameter in values:
            parameter.grad = flat[offset:offset+parameter.numel()].view_as(parameter)
            offset += parameter.numel()
    return int(count.item())
