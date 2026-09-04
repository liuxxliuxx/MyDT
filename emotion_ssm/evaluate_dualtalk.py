from __future__ import annotations

import argparse
import inspect
import json
import sys
import time
from pathlib import Path
from typing import Dict, Mapping

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

from emotion_ssm.config import load_config
from emotion_ssm.data import DualTalkChunkDataset
from emotion_ssm.models import EmotionConditionedDualTalk
from emotion_ssm.train.dualtalk import _build_system
from emotion_ssm.utils.paths import ensure_output_directory


def _install_numpy_pickle_compatibility() -> None:
    """Allow NumPy 1.x to read RNG metadata pickled by NumPy 2.x."""
    import numpy as np

    if "numpy._core" in sys.modules:
        return
    sys.modules["numpy._core"] = np.core
    for module_name in ("multiarray", "numeric", "_multiarray_umath"):
        module = getattr(np.core, module_name, None)
        if module is not None:
            sys.modules[f"numpy._core.{module_name}"] = module


def _torch_load(path: Path):
    kwargs = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = False
    return torch.load(str(path), **kwargs)


def _move_batch(batch: Mapping[str, object], device: torch.device) -> Dict[str, object]:
    return {
        name: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for name, value in batch.items()
    }


class ReconstructionTotals:
    """Accumulate element-weighted errors so the last short batch is unbiased."""

    def __init__(self) -> None:
        self.square_error = {
            "expression": 0.0,
            "jaw": 0.0,
            "neck": 0.0,
            "velocity": 0.0,
        }
        self.elements = {name: 0 for name in self.square_error}
        self.chunks = 0

    def update(self, generated: torch.Tensor, target: torch.Tensor) -> None:
        length = min(generated.shape[1], target.shape[1])
        generated = generated[:, :length].float()
        target = target[:, :length].float()
        groups = {
            "expression": (generated[:, :, :50], target[:, :, :50]),
            "jaw": (generated[:, :, 50:53], target[:, :, 50:53]),
            "neck": (generated[:, :, 53:56], target[:, :, 53:56]),
            "velocity": (
                generated[:, 1:] - generated[:, :-1],
                target[:, 1:] - target[:, :-1],
            ),
        }
        for name, (prediction, truth) in groups.items():
            self.square_error[name] += float((prediction - truth).square().sum())
            self.elements[name] += prediction.numel()
        self.chunks += generated.shape[0]

    def metrics(self) -> Dict[str, float]:
        result = {
            f"{name}_mse": self.square_error[name] / max(self.elements[name], 1)
            for name in self.square_error
        }
        result["generation_total"] = sum(result.values())
        return result


def _checkpoint_metadata(payload) -> Dict[str, object]:
    if not isinstance(payload, Mapping):
        return {}
    metadata: Dict[str, object] = {}
    for name in ("epoch", "global_step"):
        if name in payload:
            metadata[name] = int(payload[name])
    if isinstance(payload.get("metrics"), Mapping):
        metadata["saved_metrics"] = {
            str(name): float(value)
            for name, value in payload["metrics"].items()
            if isinstance(value, (int, float))
        }
    return metadata


def _load_conditioned_checkpoint(system, path: Path) -> Dict[str, object]:
    payload = _torch_load(path)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Unsupported conditioned checkpoint: {path}")
    models = payload.get("models")
    if not isinstance(models, Mapping) or "system" not in models:
        raise KeyError(f"Checkpoint does not contain models.system: {path}")
    system.load_state_dict(models["system"], strict=True)
    metadata = _checkpoint_metadata(payload)
    del payload
    return metadata


def _evaluate_baseline(cfg, checkpoint: Path, loader, device, max_batches: int):
    wrapper = EmotionConditionedDualTalk.from_config(cfg)
    payload = _torch_load(checkpoint)
    wrapper.load_baseline_state_dict(payload, strict=True)
    metadata = _checkpoint_metadata(payload)
    del payload
    model = wrapper.baseline.to(device).eval()
    totals = ReconstructionTotals()
    with torch.no_grad():
        for index, raw_batch in enumerate(loader):
            if max_batches and index >= max_batches:
                break
            batch = _move_batch(raw_batch, device)
            with autocast(enabled=device.type == "cuda"):
                generated = model(
                    batch["target_audio"],
                    batch["partner_audio"],
                    batch["partner_blendshape"],
                )
            totals.update(generated, batch["target_blendshape"])
            if (index + 1) % 50 == 0:
                print(json.dumps({"batches": index + 1, "chunks": totals.chunks}))
    return totals, metadata


def _dialogue_name(sample: Mapping[str, object]) -> str:
    name = str(sample["name"])
    for suffix in ("speaker1", "speaker2"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _random_partner_permutation(dataset: DualTalkChunkDataset) -> torch.Tensor:
    """Build a deterministic cyclic permutation with no same-dialogue partners."""
    names = [_dialogue_name(sample) for sample in dataset.samples]
    count = len(names)
    for shift in range(max(count // 2, 1), count):
        if all(names[index] != names[(index + shift) % count] for index in range(count)):
            return (torch.arange(count) + shift) % count
    raise ValueError("Could not construct a different-dialogue partner permutation")


def _evaluate_conditioned(
    cfg, checkpoint: Path, loader, device, max_batches: int, ablation: str
):
    system = _build_system(cfg, device)
    metadata = _load_conditioned_checkpoint(system, checkpoint)
    system.eval()
    totals = ReconstructionTotals()
    consistency_sum = 0.0
    consistency_count = 0
    sample_offset = 0
    random_indices = (
        _random_partner_permutation(loader.dataset)
        if ablation == "random_partner"
        else None
    )
    with torch.no_grad():
        for index, raw_batch in enumerate(loader):
            if max_batches and index >= max_batches:
                break
            batch = _move_batch(raw_batch, device)
            state_partner_audio = batch["partner_audio"]
            if random_indices is not None:
                batch_indices = random_indices[
                    sample_offset : sample_offset + len(batch["target_audio"])
                ]
                state_partner_audio = torch.stack(
                    [
                        loader.dataset.samples[int(item)]["partner_audio"]
                        for item in batch_indices
                    ]
                ).to(device, non_blocking=True)
            with autocast(enabled=device.type == "cuda"):
                context, _, evidence = system.conditioner(
                    batch["target_audio"],
                    state_partner_audio,
                    batch["dt"],
                    enable_partner=ablation != "self_only",
                )
                generated = system.generator(
                    batch["target_audio"],
                    batch["partner_audio"],
                    batch["partner_blendshape"],
                    context,
                    enable_film=ablation != "film_off",
                )
                projected = system.projector(generated)
                consistency = 1.0 - F.cosine_similarity(
                    projected, evidence["target_aff"], dim=-1
                )
            totals.update(generated, batch["target_blendshape"])
            consistency_sum += float(consistency.float().sum())
            consistency_count += consistency.numel()
            sample_offset += len(batch["target_audio"])
            if (index + 1) % 50 == 0:
                print(json.dumps({"batches": index + 1, "chunks": totals.chunks}))
    metrics = totals.metrics()
    metrics["state_consistency"] = consistency_sum / max(consistency_count, 1)
    return totals, metadata, metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate official and emotion-conditioned DualTalk checkpoints"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", choices=("baseline", "conditioned"), required=True)
    parser.add_argument(
        "--ablation",
        choices=("full", "film_off", "self_only", "random_partner"),
        default="full",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path)
    parser.add_argument("--split", choices=("test", "ood"), default="test")
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    _install_numpy_pickle_compatibility()
    cfg = load_config(args.config, args.opts)
    cfg = cfg.clone()
    cfg.defrost()
    if args.baseline_checkpoint is not None:
        cfg.DUALTALK.BASELINE_CHECKPOINT = str(args.baseline_checkpoint)
    cfg.freeze()

    if args.model == "conditioned" and args.baseline_checkpoint is None:
        parser.error("--baseline-checkpoint is required for the conditioned model")
    if not torch.cuda.is_available():
        raise RuntimeError("DualTalk evaluation requires a CUDA GPU")
    if args.max_batches < 0:
        parser.error("--max-batches must be non-negative")

    torch.manual_seed(cfg.SEED)
    torch.cuda.manual_seed_all(cfg.SEED)
    device = torch.device("cuda:0")
    dataset = DualTalkChunkDataset(
        Path(cfg.DATA.DUALTALK_ROOT) / args.split,
        cfg.DUALTALK.CHUNK_FRAMES,
        cfg.DUALTALK.FPS,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.TRAIN.SEQUENCE_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=cfg.DATA.PIN_MEMORY,
    )
    started = time.time()
    if args.model == "baseline":
        totals, metadata = _evaluate_baseline(
            cfg, args.checkpoint, loader, device, args.max_batches
        )
        metrics = totals.metrics()
    else:
        totals, metadata, metrics = _evaluate_conditioned(
            cfg, args.checkpoint, loader, device, args.max_batches, args.ablation
        )

    result = {
        "model": args.model,
        "ablation": "official" if args.model == "baseline" else args.ablation,
        "checkpoint": str(args.checkpoint.resolve()),
        "split": args.split,
        "dataset_chunks": len(dataset),
        "evaluated_chunks": totals.chunks,
        "batch_size": cfg.TRAIN.SEQUENCE_BATCH_SIZE,
        "max_batches": args.max_batches,
        "elapsed_seconds": time.time() - started,
        **metadata,
        **metrics,
    }
    output_dir = ensure_output_directory(args.output.parent, [cfg.DATA.ROOT])
    output_path = output_dir / args.output.name
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
