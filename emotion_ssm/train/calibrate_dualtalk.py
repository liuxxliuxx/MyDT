"""Calibrate domain-2 and FLAME against a fixed, same-time A/T teacher."""
from __future__ import annotations

import copy
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

from emotion_ssm.config import parse_config_args
from emotion_ssm.data.timed_dualtalk import TimedDualTalk, to_device
from emotion_ssm.models.conditioned_dualtalk import BlendshapeAffectProjector
from emotion_ssm.models.multimodal import MultimodalObserver, representation_diagnostics
from emotion_ssm.models.observation import ObservationEncoder
from emotion_ssm.train.dynamics_core import load_component_state
from emotion_ssm.utils.generation_checkpoint import source_for_config, state_dict_digest


def calibration_pair(observer, teacher, packet, target_flame, valid, source_domain):
    values = {k: v for k, v in packet["target_features"].items() if torch.is_tensor(v)}
    values["modality_mask"] = values["modality_mask"].clone()
    values["modality_mask"][:, 1] = False
    values["face"] = target_flame.new_zeros(len(target_flame), 1, 35)
    values["dataset_id"] = torch.full((len(target_flame),), source_domain, dtype=torch.long, device=target_flame.device)
    with torch.no_grad():
        teacher_value = teacher(values, torch.tensor([[1, 0, 1]], device=target_flame.device)).aff[:, 0]
    present = values["modality_mask"].any(-1) & valid.any(-1)
    values["dataset_id"] = torch.full_like(values["dataset_id"], observer.domain_id)
    values["flame"], values["flame_mask"] = target_flame, valid
    values["modality_mask"][:, 1] = valid.any(-1)
    full = observer(values).aff
    visual = dict(values)
    visual["modality_mask"] = torch.zeros_like(values["modality_mask"])
    visual["modality_mask"][:, 1] = valid.any(-1)
    return full, observer(visual).aff, teacher_value, present


def calibration_loss(full, visual, projected, teacher, valid):
    def distance(x):
        return ((1-F.cosine_similarity(x, teacher, dim=-1)) * valid).sum() / valid.sum().clamp_min(1)
    selected = visual[valid]
    variance = F.relu(.02-selected.std(0, unbiased=False)).mean() if len(selected) > 1 else visual.sum()*0
    return distance(full) + distance(visual) + distance(projected) + .1 * variance


def distributed_calibration_loss(full, visual, projected, teacher, valid):
    if dist.is_available() and dist.is_initialized():
        from torch.distributed.nn.functional import all_gather
        full, visual, projected = [torch.cat(all_gather(value)) for value in (full, visual, projected)]
        targets = [torch.empty_like(teacher) for _ in range(dist.get_world_size())]
        masks = [torch.empty_like(valid) for _ in range(dist.get_world_size())]
        dist.all_gather(targets, teacher); dist.all_gather(masks, valid)
        teacher, valid = torch.cat(targets), torch.cat(masks)
    return calibration_loss(full, visual, projected, teacher, valid)


@torch.no_grad()
def validate_calibration(observer, teacher, projector, dataset, domain, device):
    observer.eval()
    projector.eval()
    full, visual, projected, targets = [], [], [], []
    rank = dist.get_rank() if dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_initialized() else 1
    for index in range(rank, len(dataset), world):
        for packet, truth, mask in dataset.packets(index):
            packet, truth, mask = to_device(packet, device), truth.to(device), mask.to(device)
            f, v, t, valid = calibration_pair(observer, teacher, packet, truth, mask, domain)
            if valid.any():
                full.append(f[valid].cpu()); visual.append(v[valid].cpu())
                projected.append(projector(truth)[valid].cpu()); targets.append(t[valid].cpu())
    if world > 1:
        shards = [None] * world
        dist.all_gather_object(shards, (full, visual, projected, targets))
        full, visual, projected, targets = [[value for shard in shards for value in shard[i]] for i in range(4)]
    if not targets:
        raise ValueError("Calibration validation has no valid aligned A/T targets")
    target, f, v, p = map(torch.cat, (targets, full, visual, projected))
    mean = F.normalize(target.mean(0, keepdim=True), dim=-1).expand_as(target)
    baseline = float((1-F.cosine_similarity(mean, target)).mean())
    errors = {name: float((1-F.cosine_similarity(value, target)).mean())
              for name, value in (("full", f), ("visual", v), ("projector", p))}
    return {**errors, "mean_baseline": baseline,
            "visual_representation": representation_diagnostics(v),
            "projector_representation": representation_diagnostics(p)}


def main():
    cfg, _ = parse_config_args(__doc__)
    from emotion_ssm.utils.distributed import init_distributed
    context = init_distributed(cfg.DEVICE, cfg.SEED, cfg.DETERMINISTIC)
    device = context.device
    domain, source = source_for_config(cfg)
    encoder = ObservationEncoder.from_config(cfg)
    phase_b = Path(cfg.DUALTALK.PHASE_B_CHECKPOINT or cfg.TRAIN.PHASE_B_CHECKPOINT)
    from emotion_ssm.utils.generation_checkpoint import read_checkpoint, checkpoint_config
    from emotion_ssm.data.protocol import require_compatible
    phase_payload = read_checkpoint(phase_b)
    source_name = "emotiontalk" if domain == 0 else "iemocap"
    if source_name not in checkpoint_config(phase_payload).DATA.SOURCES:
        raise ValueError("Calibration adapter domain was not trained in this Phase-B checkpoint")
    require_compatible(source, phase_payload.get("feature_sources", {}).get(source_name, {}))
    encoder.load_state_dict(load_component_state(phase_b, "encoder"), strict=True)
    teacher = copy.deepcopy(encoder).to(device).eval().requires_grad_(False)
    encoder.copy_domain_adapters(domain, 2)
    coordinate_id = state_dict_digest(encoder.state_dict(), exclude_domain=2)
    observer = MultimodalObserver(encoder, cfg.MODEL.MODEL_DIM, cfg.MODEL.AU_NUM_HEADS,
                                  cfg.MODEL.AU_NUM_LAYERS, cfg.MODEL.DROPOUT).to(device)
    encoder.requires_grad_(False)
    for adapters in (encoder.audio_adapter, encoder.text_adapter):
        adapters.adapters[2].requires_grad_(True)
    projector = BlendshapeAffectProjector(56, cfg.MODEL.OBSERVATION_DIM).to(device)
    if context.enabled:
        for module in (observer, projector):
            for value in list(module.parameters()) + list(module.buffers()):
                dist.broadcast(value.data, 0)
    parameters = [p for p in observer.parameters() if p.requires_grad] + list(projector.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=cfg.TRAIN.LR)
    data = {split: TimedDualTalk(cfg.DATA.DUALTALK_ROOT, cfg.DUALTALK.TIMED_FEATURE_ROOT,
                               cfg.DUALTALK.SPLIT_MANIFEST, split, threshold=cfg.DUALTALK.SPEECH_RMS_THRESHOLD,
                               expected_source=source) for split in ("train", "val")}
    from emotion_ssm.train.common import create_run_directory
    output = create_run_directory(cfg, "dualtalk_calibration")
    if not len(data["train"]):
        raise ValueError("Calibration training split is empty")
    if 8 % context.world_size:
        raise ValueError("Calibration world size must divide the global eight-block batch")
    from emotion_ssm.train.generation import PacketCursor
    cursor = PacketCursor(data["train"], cfg.SEED, context.rank, context.world_size)
    best = float("inf")
    for step in range(1, cfg.TRAIN.MAX_STEPS+1):
        accumulated, inspected = [], 0
        while len(accumulated) < 8 // context.world_size:
            packet, truth, mask = cursor.next()
            inspected += 1
            if inspected > 10000:
                raise ValueError("Calibration shard has no usable aligned FLAME and A/T targets")
            observer.eval(); projector.train()
            packet, truth, mask = to_device(packet, device), truth.to(device), mask.to(device)
            full, visual, target, valid = calibration_pair(observer, teacher, packet, truth, mask, domain)
            if valid.any():
                accumulated.append((full, visual, projector(truth), target, valid))
        loss = distributed_calibration_loss(*(torch.cat(values) for values in zip(*accumulated)))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if context.enabled:
            for parameter in parameters:
                if parameter.grad is None:
                    parameter.grad = torch.zeros_like(parameter)
                dist.all_reduce(parameter.grad)
                parameter.grad.div_(context.world_size)
        torch.nn.utils.clip_grad_norm_(parameters, cfg.TRAIN.GRAD_CLIP)
        optimizer.step()
        if step % cfg.TRAIN.VAL_EVERY_STEPS == 0 or step == cfg.TRAIN.MAX_STEPS:
            metrics = validate_calibration(observer, teacher, projector, data["val"], domain, device)
            score = metrics["full"] + metrics["visual"]
            if score < best:
                best = score
                if context.is_main:
                    common = {"format_version": 2, "feature_source": source, "coordinate_id": coordinate_id,
                              "metrics": metrics, "global_step": step, "config": cfg.dump(), "world_size":context.world_size}
                    torch.save({**common, "observer": observer.state_dict(),
                                "calibrated": metrics["visual"] < metrics["mean_baseline"] and
                                (metrics["visual_representation"]["mean_std"] or 0) > 1e-4}, output / "observer.pt")
                    torch.save({**common, "projector": projector.state_dict(),
                                "calibrated": metrics["projector"] < metrics["mean_baseline"] and
                                (metrics["projector_representation"]["mean_std"] or 0) > 1e-4}, output / "projector.pt")
            if context.is_main:
                print({"step": step, **metrics}, flush=True)
    context.barrier()


if __name__ == "__main__":
    main()
