from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, Sampler
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
                    Path(cfg.DATA.EMOTIONTALK_ROOT), "emotiontalk", 0,
                    require_v2=not cfg.DATA.ALLOW_LEGACY_FEATURES,
                )
            )
        else:
            stores.append(
                FeatureDialogueStore(
                    Path(cfg.DATA.IEMOCAP_FEATURE_ROOT),
                    "iemocap",
                    1,
                    fold=int(cfg.DATA.IEMOCAP_FOLD),
                    require_v2=not cfg.DATA.ALLOW_LEGACY_FEATURES,
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


class DistributedLengthBucketSampler(Sampler[List[int]]):
    """Shard dialogues and batch nearby sequence lengths together.

    The sampler preserves the same distributed sample semantics as
    ``DistributedSampler``. It only changes the order within each epoch so
    padding on the dialogue/chunk axis is reduced.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        num_replicas: int = 1,
        rank: int = 0,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 6666,
        bucket_multiplier: int = 20,
    ) -> None:
        if not hasattr(dataset, "lengths"):
            raise ValueError("length-bucketed sampling requires dataset.lengths")
        self.lengths = [int(value) for value in dataset.lengths]
        self.dataset_size = len(self.lengths)
        self.batch_size = max(int(batch_size), 1)
        self.num_replicas = max(int(num_replicas), 1)
        self.rank = int(rank)
        if not 0 <= self.rank < self.num_replicas:
            raise ValueError("rank must be within [0, num_replicas)")
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.bucket_size = max(self.batch_size * int(bucket_multiplier), self.batch_size)
        if self.drop_last:
            self.num_samples = self.dataset_size // self.num_replicas
        else:
            self.num_samples = int(math.ceil(self.dataset_size / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.epoch = 0

    def __len__(self) -> int:
        if not self.shuffle and not self.drop_last:
            local_size = len(range(self.rank, self.dataset_size, self.num_replicas))
            return int(math.ceil(local_size / self.batch_size))
        if self.drop_last:
            return self.num_samples // self.batch_size
        return int(math.ceil(self.num_samples / self.batch_size))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        indices = list(range(self.dataset_size))
        rng = random.Random(self.seed + self.epoch)
        if self.shuffle:
            rng.shuffle(indices)
        if self.drop_last:
            indices = indices[: self.total_size]
        elif self.shuffle and len(indices) < self.total_size:
            if not indices:
                return
            repeats = int(math.ceil((self.total_size - len(indices)) / len(indices)))
            indices.extend((indices * (repeats + 1))[: self.total_size - len(indices)])
        indices = indices[: self.total_size]
        local_indices = indices[self.rank : self.total_size : self.num_replicas]

        buckets = []
        for start in range(0, len(local_indices), self.bucket_size):
            bucket = local_indices[start : start + self.bucket_size]
            bucket.sort(key=lambda index: self.lengths[index], reverse=True)
            buckets.append(bucket)
        if self.shuffle:
            rng.shuffle(buckets)

        batches: List[List[int]] = []
        for bucket in buckets:
            for start in range(0, len(bucket), self.batch_size):
                batch = bucket[start : start + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)
        if self.shuffle:
            rng.shuffle(batches)
        for batch in batches:
            yield batch


class ExactEvaluationSampler(Sampler[int]):
    """No repetition/padding: each validation sample belongs to one rank."""
    def __init__(self, dataset, rank, world_size):
        self.indices = list(range(rank, len(dataset), world_size))

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def make_loader(
    dataset: Dataset,
    batch_size: int,
    context: DistributedContext,
    train: bool,
    num_workers: int,
    pin_memory: bool,
    collate_fn=None,
    bucket_by_length: bool = False,
    seed: int = 6666,
) -> Tuple[DataLoader, object]:
    if bucket_by_length and hasattr(dataset, "lengths"):
        batch_sampler = DistributedLengthBucketSampler(
            dataset,
            batch_size,
            num_replicas=context.world_size,
            rank=context.rank,
            shuffle=train,
            drop_last=train,
            seed=seed,
        )
        loader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory and context.device.type == "cuda",
            persistent_workers=num_workers > 0,
            collate_fn=collate_fn,
        )
        return loader, batch_sampler
    sampler = None
    if context.enabled:
        sampler = DistributedSampler(
            dataset,
            num_replicas=context.world_size,
            rank=context.rank,
            shuffle=train,
            drop_last=train,
        ) if train else ExactEvaluationSampler(dataset, context.rank, context.world_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train and sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory and context.device.type == "cuda",
        drop_last=train,
        persistent_workers=num_workers > 0,
        collate_fn=collate_fn,
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
