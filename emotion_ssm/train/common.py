from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from emotion_ssm.data import FeatureDialogueStore, build_speaker_vocabulary
from emotion_ssm.utils.distributed import DistributedContext
from emotion_ssm.utils.paths import ensure_output_directory


def build_feature_stores(cfg) -> Tuple[FeatureDialogueStore, ...]:
    sources = list(cfg.DATA.SOURCES)
    unknown = sorted(set(sources) - {"emotiontalk", "iemocap"})
    if unknown:
        raise ValueError(f"Unknown DATA.SOURCES values: {unknown}")
    if not sources:
        raise ValueError("DATA.SOURCES must contain at least one dataset")
    if len(sources) != len(set(sources)):
        raise ValueError(f"DATA.SOURCES contains duplicates: {sources}")

    stores = []
    for source in sources:
        if source == "emotiontalk":
            stores.append(
                FeatureDialogueStore(
                    Path(cfg.DATA.EMOTIONTALK_ROOT), "emotiontalk", 0
                )
            )
        else:
            stores.append(
                FeatureDialogueStore(
                    Path(cfg.DATA.IEMOCAP_FEATURE_ROOT),
                    "iemocap",
                    1,
                    fold=int(cfg.DATA.IEMOCAP_FOLD),
                )
            )
    return tuple(stores)


def readonly_roots(cfg) -> Sequence[Path]:
    return [
        Path(cfg.DATA.ROOT),
        Path(cfg.DATA.EMOTIONTALK_ROOT),
        Path(cfg.DATA.IEMOCAP_RAW_ROOT),
        Path(cfg.DATA.DUALTALK_ROOT),
    ]


def create_run_directory(cfg, stage_name: str) -> Path:
    experiment = cfg.TRAIN.EXPERIMENT_NAME
    return ensure_output_directory(
        Path(cfg.TRAIN.OUTPUT_ROOT) / stage_name / experiment,
        readonly_roots(cfg),
    )


def move_to_device(batch: Mapping[str, object], device: torch.device) -> Dict[str, object]:
    return {
        name: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for name, value in batch.items()
    }


def make_loader(
    dataset: Dataset,
    batch_size: int,
    context: DistributedContext,
    train: bool,
    num_workers: int,
    pin_memory: bool,
) -> Tuple[DataLoader, object]:
    sampler = None
    if context.enabled:
        sampler = DistributedSampler(
            dataset,
            num_replicas=context.world_size,
            rank=context.rank,
            shuffle=train,
            drop_last=train,
        )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train and sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory and context.device.type == "cuda",
        drop_last=train,
        persistent_workers=num_workers > 0,
    )
    return loader, sampler


def maybe_ddp(model: nn.Module, context: DistributedContext) -> nn.Module:
    if not context.enabled:
        return model
    kwargs = {}
    if context.device.type == "cuda":
        kwargs = {"device_ids": [context.local_rank], "output_device": context.local_rank}
    # Phase A intentionally disables coupling blocks, so some parameters are unused.
    return DistributedDataParallel(model, find_unused_parameters=True, **kwargs)


def grl_alpha(progress: float, warmup_fraction: float) -> float:
    if warmup_fraction <= 0:
        return 1.0
    value = min(progress / warmup_fraction, 1.0)
    return 2.0 / (1.0 + math.exp(-10.0 * value)) - 1.0


def append_metrics(path: Path, values: Mapping[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(values), ensure_ascii=False) + "\n")


class ObservationTrainingBundle(nn.Module):
    def __init__(self, encoder: nn.Module, heads: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.heads = heads

    def forward(self, batch, subset_masks, grl_strength: float):
        output = self.encoder(batch, subset_masks)
        predictions = self.heads(output.aff, grl_strength)
        return output, predictions
