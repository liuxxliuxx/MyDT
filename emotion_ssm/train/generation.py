"""One training protocol for none/affect/self/dyadic controlled experiments."""
from __future__ import annotations

import contextlib
import json
import random
from pathlib import Path

import torch
import torch.distributed as dist

from emotion_ssm.data.timed_dualtalk import TimedDualTalk, to_device
from emotion_ssm.models.multimodal import representation_diagnostics
from emotion_ssm.train.common import append_metrics
from emotion_ssm.utils.distributed import init_distributed
from emotion_ssm.utils.generation_checkpoint import build_streaming, load_generation, save_generation
from emotion_ssm.utils.reconstruction import ReconstructionTotals, reconstruction_loss


def make_dataset(cfg, split, source):
    return TimedDualTalk(cfg.DATA.DUALTALK_ROOT, cfg.DUALTALK.TIMED_FEATURE_ROOT,
                        cfg.DUALTALK.SPLIT_MANIFEST, split, cfg.DUALTALK.FPS,
                        cfg.DUALTALK.SPEECH_RMS_THRESHOLD, source)


class PacketCursor:
    """Deterministic rank-local cursor; a saved count reconstructs exact order."""
    def __init__(self, dataset, seed, rank=0, world_size=1, seen=0):
        self.dataset, self.seed, self.rank, self.world_size = dataset, seed, rank, world_size
        if len(dataset) < world_size:
            raise ValueError("Not enough dialogues for distributed training")
        self.seen = 0
        self.iterator = self._iterate()
        for _ in range(seen):
            self.next()

    def _iterate(self):
        epoch = 0
        while True:
            order = list(range(len(self.dataset)))
            random.Random(self.seed + epoch).shuffle(order)
            produced = False
            for index in order[self.rank::self.world_size]:
                for packet, truth, mask in self.dataset.packets(index):
                    produced = True
                    # An epoch starts a new episode even for the same clip.
                    packet["session_id"] = f"{epoch}:{packet['session_id']}"
                    yield packet, truth, mask
            if not produced:
                raise ValueError("Training shard contains no usable packets")
            epoch += 1

    def next(self):
        value = next(self.iterator)
        self.seen += 1
        return value


@torch.no_grad()
def evaluate(model, dataset, device, rank=0, world_size=1, max_dialogues=0, ablation="full"):
    model.eval()
    totals, records, affects = ReconstructionTotals(), [], []
    from emotion_ssm.utils.statistics import RepresentationTotals, WeightedStatistics
    from emotion_ssm.models.observation import SUBSET_NAMES
    representations, preservation = RepresentationTotals(), WeightedStatistics()
    indices = list(range(len(dataset)))
    if max_dialogues:
        indices = indices[:max_dialogues]
    indices = indices[rank::world_size]
    saved_variant = model.variant
    if ablation == "film_off":
        model.variant = "none"
    elif ablation == "self_only":
        model.variant = "self"
    for index in indices:
        state, previous = None, None
        local = ReconstructionTotals()
        for chunk_index, (packet, truth, mask) in enumerate(dataset.packets(index)):
            packet, truth, mask = to_device(packet, device), truth.to(device), mask.to(device)
            packet["collect_diagnostics"] = index < 16 and chunk_index < 32
            generated, state, diagnostics = model(packet, state)
            for role, (observed, teacher, teacher_valid) in diagnostics.get("subsets", {}).items():
                representations.update(observed, torch.full_like(teacher_valid, 2, dtype=torch.long),
                                       [f"{role}_{name}" for name in SUBSET_NAMES])
                for sub, name in enumerate(SUBSET_NAMES):
                    valid = observed.valid_subsets[:, sub] & teacher_valid
                    if valid.any():
                        error = 1-torch.nn.functional.cosine_similarity(observed.aff[:, sub], teacher, dim=-1)
                        preservation.add(f"domain2_{role}_{name}_teacher_error", error[valid].mean(), valid.sum())
            local.update(generated, truth, mask, previous)
            previous = (generated[:, -1].detach(), truth[:, -1].detach(), mask[:, -1])
            if index < 16 and len(affects) < 512:
                affects.append(diagnostics["partner_aff"].cpu())
        totals.merge(local)
        records.append({"dialogue": dataset.names[index], "sse": local.square_error,
                        "elements": local.elements, **local.metrics()})
    model.variant = saved_variant
    totals.distributed_sum(device)
    if dist.is_available() and dist.is_initialized():
        gathered = [None] * world_size
        dist.all_gather_object(gathered, records)
        records = [item for shard in gathered for item in shard]
        gathered_affect = [None] * world_size
        dist.all_gather_object(gathered_affect, affects)
        affects = [item for shard in gathered_affect for item in shard]
    metrics = totals.metrics()
    if not metrics["valid_frames"]:
        raise ValueError("Evaluation split contains no valid target frames")
    metrics["error_elements"] = dict(totals.elements)
    metrics["observation_diagnostics"] = {**representations.metrics(), **preservation.finalize()}
    metrics["diagnostic_sampling"] = "first 16 dialogue indices, first 32 blocks each"
    if affects:
        metrics["partner_representation"] = representation_diagnostics(torch.cat(affects))
    return metrics, records


def run_generation(cfg):
    if cfg.DUALTALK.CHUNK_FRAMES != 25 or cfg.DUALTALK.FPS != 25:
        raise ValueError("Protocol v2 requires 25 frames at 25 FPS; do not silently mix protocols")
    from emotion_ssm.utils.generation_checkpoint import read_checkpoint, checkpoint_config
    payload = read_checkpoint(cfg.TRAIN.RESUME) if cfg.TRAIN.RESUME else None
    authoritative = checkpoint_config(payload) if payload else cfg
    context = init_distributed(cfg.DEVICE, authoritative.SEED, authoritative.DETERMINISTIC)
    device = context.device
    if cfg.TRAIN.RESUME:
        requested = cfg
        model, cfg, payload = load_generation(payload, device)
        from emotion_ssm.utils.emotion_checkpoint import deployment_paths
        cfg = deployment_paths(cfg, requested)
    else:
        model = build_streaming(cfg, device)
    if context.enabled:
        for value in list(model.parameters()) + list(model.buffers()):
            dist.broadcast(value.data, 0)
    source = model.construction_info["feature_source"]
    train, validation = make_dataset(cfg, "train", source), make_dataset(cfg, "val", source)
    from emotion_ssm.train.common import create_run_directory
    run_dir = create_run_directory(cfg, "dualtalk_conditioned")
    if context.is_main:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.yaml").write_text(cfg.dump(), encoding="utf-8")
    context.barrier()
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.TRAIN.AMP and device.type == "cuda")
    start, best, saved = 0, float("inf"), {}
    if payload:
        run = payload["run_state"]
        if run["split_digest"] != train.manifest_digest or run["world_size"] != context.world_size:
            raise ValueError("Resume requires identical splits and world size; use a new experiment otherwise")
        if run.get("feature_digest") != getattr(train, "feature_digest", None):
            raise ValueError("Features changed since this experiment; start a new run")
        optimizer.load_state_dict(payload["optimizer"])
        for values in optimizer.state.values():
            for key, value in values.items():
                if torch.is_tensor(value):
                    values[key] = value.to(device)
        start, best = payload["global_step"], run["best_score"]
        saved = run["ranks"][context.rank]
        if run.get("scaler"):
            scaler.load_state_dict(run["scaler"])
    cursor = PacketCursor(train, cfg.SEED, context.rank, context.world_size, saved.get("packets_seen", 0))
    state = saved.get("stream_state")
    if state is not None:
        state.history = [to_device(x, device) for x in state.history]
        if state.emotion is not None:
            from emotion_ssm.schema import DyadicState
            e = state.emotion
            state.emotion = DyadicState(e.z.to(device), e.relation.to(device), e.speaker_ids.to(device))
    if saved.get("rng"):
        from emotion_ssm.utils.checkpoint import restore_rng_state
        restore_rng_state(saved["rng"])
    global_chunks = cfg.TRAIN.GLOBAL_CHUNKS_PER_STEP
    if global_chunks % context.world_size:
        raise ValueError("GLOBAL_CHUNKS_PER_STEP must divide evenly across ranks")
    limit = min(cfg.TRAIN.MAX_STEPS, start+2) if cfg.TRAIN.DRY_RUN else cfg.TRAIN.MAX_STEPS
    for step in range(start, limit):
        model.train()
        packets, valid_chunks = [], 0
        while valid_chunks < global_chunks // context.world_size:
            item = cursor.next()
            packets.append(item)
            valid_chunks += bool(item[2].any())
            if len(packets) > 10000:
                raise ValueError("Training shard has too few valid target chunks; inspect preprocessing report")
        counts = torch.tensor([sum(int(mask.sum()) for _, _, mask in packets),
                               sum(int((mask[:, 1:] & mask[:, :-1]).sum()) for _, _, mask in packets)],
                              dtype=torch.float64, device=device)
        if context.enabled:
            dist.all_reduce(counts)
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.
        for packet, truth, mask in packets:
            packet, truth, mask = to_device(packet, device), truth.to(device), mask.to(device)
            if state is not None and state.session_id != packet["session_id"]:
                state = None
            if not mask.any():
                with torch.no_grad():
                    _, state, _ = model(packet, state)
                continue
            with torch.autocast(device.type, enabled=cfg.TRAIN.AMP and device.type == "cuda"):
                generated, state, diagnostics = model(packet, state)
                losses = reconstruction_loss(generated, truth, mask)
                frame_weight = mask.sum() / counts[0].clamp_min(1)
                edge_weight = (mask[:, 1:] & mask[:, :-1]).sum() / counts[1].clamp_min(1)
                loss = sum(losses[n] for n in ("expression", "jaw", "neck")) * frame_weight + losses["velocity"] * edge_weight
                if model.state_loss_weight:
                    loss = loss + model.state_loss_weight * frame_weight * (
                        1-torch.nn.functional.cosine_similarity(model.projector(generated), diagnostics["target_aff"].detach())).mean()
            scaler.scale(loss).backward()
            loss_sum += float(loss.detach())
            state = state.detach()
        if context.enabled:
            # Loss numerators were divided by global valid element counts.
            # Sum gradients; do not average them a second time.
            for parameter in parameters:
                if parameter.grad is None:
                    parameter.grad = torch.zeros_like(parameter)
                dist.all_reduce(parameter.grad)
        # Unscale after summing, so all ranks detect the same AMP overflow.
        scaler.unscale_(optimizer)
        if not all(p.grad is None or torch.isfinite(p.grad).all() for p in parameters):
            raise FloatingPointError("Non-finite gradients; no optimizer update was counted. Use the same precision in every control.")
        torch.nn.utils.clip_grad_norm_(parameters, cfg.TRAIN.GRAD_CLIP)
        scaler.step(optimizer); scaler.update()
        learning_rate = cfg.TRAIN.LR * .5 * (1+__import__("math").cos(__import__("math").pi*(step+1)/cfg.TRAIN.MAX_STEPS))
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        if context.is_main and (step+1) % cfg.TRAIN.LOG_INTERVAL == 0:
            print({"step": step+1, "rank0_loss_contribution": loss_sum, "lr": learning_rate}, flush=True)
        if (step+1) % cfg.TRAIN.VAL_EVERY_STEPS == 0 or step+1 == limit:
            metrics, records = evaluate(model, validation, device, context.rank, context.world_size,
                                        1 if cfg.TRAIN.DRY_RUN else 0)
            improved = metrics["generation_total"] < best
            best = min(best, metrics["generation_total"])
            from emotion_ssm.utils.checkpoint import capture_rng_state
            local = {"packets_seen": cursor.seen, "stream_state": state, "rng": capture_rng_state()}
            ranks = [local]
            if context.enabled:
                ranks = [None] * context.world_size
                dist.all_gather_object(ranks, local)
            run = {"split_digest": train.manifest_digest, "world_size": context.world_size,
                   "feature_digest": getattr(train, "feature_digest", None),
                   "text_protocol": getattr(train, "text_protocol", None),
                   "ranks": ranks, "best_score": best, "scaler": scaler.state_dict()}
            if context.is_main:
                append_metrics(run_dir / "metrics.jsonl", {"step": step+1, **metrics})
                save_generation(run_dir / "last.pt", model, cfg, step+1, optimizer, metrics, run)
                if improved:
                    save_generation(run_dir / "best_generation.pt", model, cfg, step+1, optimizer, metrics, run)
                    (run_dir / "best_validation.json").write_text(json.dumps({"metrics": metrics, "dialogues": records}, indent=2), encoding="utf-8")
                print({"validation_step": step+1, **metrics}, flush=True)
            context.barrier()


if __name__ == "__main__":
    from emotion_ssm.config import parse_config_args
    configuration, _ = parse_config_args(__doc__)
    run_generation(configuration)
