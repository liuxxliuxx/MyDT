from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor


@dataclass
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def enabled(self) -> bool:
        return self.world_size > 1

    def barrier(self) -> None:
        if self.enabled:
            dist.barrier()

    def reduce_scalars(self, values: Dict[str, float]) -> Dict[str, float]:
        if not self.enabled:
            return values
        names = sorted(values)
        tensor = torch.tensor(
            [values[name] for name in names], device=self.device, dtype=torch.float64
        )
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= self.world_size
        return {name: tensor[i].item() for i, name in enumerate(names)}


def init_distributed(device_name: str, seed: int, deterministic: bool) -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl" if device_name.startswith("cuda") and torch.cuda.is_available() else "gloo")

    if device_name.startswith("cuda") and torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}" if world_size > 1 else device_name)
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    process_seed = seed + rank
    random.seed(process_seed)
    np.random.seed(process_seed)
    torch.manual_seed(process_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(process_seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = torch.cuda.is_available()
    return DistributedContext(rank, local_rank, world_size, device)


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model
