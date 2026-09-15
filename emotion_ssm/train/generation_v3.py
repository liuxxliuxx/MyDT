"""Joint avatar training with segment BPTT and globally weighted supervision.

These functions are shared by the v3 command line pipeline and small-model
tests. No model owns a permanent freeze or truncation policy.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from collections import deque
import json
import random
from pathlib import Path

import torch
import torch.distributed as dist

from emotion_ssm.models.streaming_v3 import map_tensors
from emotion_ssm.utils.reconstruction import ReconstructionTotals, DeferredReconstructionTotals
from emotion_ssm.config_v3 import GENERATION_REVISION
from emotion_ssm.utils.generation_losses import (loss_settings, boundary_frame, matching_boundary,
    boundary_loss_terms, plan_objective_counts, visual_window_mask, visual_loss_terms, VISUAL_NAMES,required_visual_heads)
from emotion_ssm.train.generation_sampling import ConversationStates, InterleavedCursor, session_key
from emotion_ssm.train.generation_stability import (configure_stability, optimizer_groups,
    set_learning_rates, mask_diagnostics, run_fixed_probes, STABILITY_PROTOCOL)

OPTIMIZATION_PROTOCOL = "v3-variant-objectives-component-clipping-baseline-tbptt-v2"
EXECUTION_PROTOCOL = "generation-metrics-prefetch-v1"


def generation_execution_settings(config, requested=None):
    """The only runtime overrides on resume are these two execution switches."""
    value = dict(config.get("generation_execution", {}))
    if requested is not None:
        value.update(requested.get("generation_execution", {}))
    if set(value) - {"defer_metrics", "prefetch_batches"}:
        raise ValueError("Unknown generation execution override")
    result = {"defer_metrics": value.get("defer_metrics", False),
              "prefetch_batches": value.get("prefetch_batches", 0)}
    if type(result["defer_metrics"]) is not bool or type(result["prefetch_batches"]) is not int:
        raise ValueError("Execution switches require bool and int values")
    if result["prefetch_batches"] not in (0, 1):
        raise ValueError("Only a single speculative CPU batch is supported")
    return result


def distributed():
    return dist.is_available() and dist.is_initialized()


class AMPGradientOverflow(FloatingPointError):
    """Retry the same valid-block budget after GradScaler has reduced its scale."""


def configure_generation_stage(model, *, train_observer=True, train_state=True,
                               frozen_teacher=None, coordinate_modules=(), gradient_checkpointing=True):
    """Set training permission once at a stage boundary, never inside forward.

    Every control exposes the same original DualTalk backbone. The original
    waveform convolution stays frozen; downstream speech layers remain trainable.
    Explicit semantic coordinate modules and a teacher can stay anchored while
    the state core and adapters learn from expression supervision.
    """
    train_observer = bool(train_observer and model.variant != "none")
    train_state = bool(train_state and model.variant in ("self", "dyadic"))
    model.requires_grad_(True)
    model.observer.requires_grad_(train_observer)
    model.state_model.requires_grad_(train_state)
    baseline = getattr(model.generator, "baseline", None)
    if baseline is not None:
        baseline.requires_grad_(True)
        joint = getattr(baseline, "joint_encoder", None)
        for name in ("audio_encoder1", "audio_encoder2"):
            encoder = getattr(joint, name, None)
            convolution = getattr(encoder, "feature_extractor", None)
            if convolution is not None:
                convolution.requires_grad_(False)
            if gradient_checkpointing and hasattr(encoder, "gradient_checkpointing_enable"):
                encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    film = getattr(model.generator, "film", None)
    router = getattr(model, "condition_router", None)
    film_enabled = model.variant != "none" if router is None else router.mode != "film_off"
    if film is not None:
        film.requires_grad_(film_enabled)
    # Pretrained token extraction is a separate fixed component; adapters belong
    # to observer, so keeping this extractor fixed does not block adapter gradients.
    fixed = [model.features, frozen_teacher, getattr(model, "visual_teacher", None), *coordinate_modules,
             None if train_observer else model.observer, None if train_state else model.state_model]
    for module in fixed:
        if module is not None:
            module.requires_grad_(False)
            module.eval()
    model._v3_fixed_modules = tuple(module for module in fixed if module is not None)
    for parameter in model.parameters():
        if not parameter.requires_grad:
            parameter.grad = None
    model.generator_checkpointing = bool(gradient_checkpointing)
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def set_training_mode(model):
    model.train()
    for module in getattr(model, "_v3_fixed_modules", ()):
        module.eval()


def configure_generation_from_config(model, config):
    """Restore the declared upstream freeze policy before constructing AdamW."""
    settings = config['generation']
    configure_generation_stage(model,
        train_observer=settings.get('train_observer', True),
        train_state=settings.get('train_state', True),
        frozen_teacher=getattr(model, 'teacher', None),
        coordinate_modules=tuple(getattr(model.observer, name) for name in
            ('affect_head', 'emotion_head', 'intensity_head', 'vad_head')),
        gradient_checkpointing=settings.get('gradient_checkpointing', True))
    configure_stability(model, config)
    model._generation_loss_settings = loss_settings(config.get("generation_losses"))
    teacher = getattr(model, "visual_teacher", None)
    names = dict(visual_feature="affect", visual_class_distill="emotion",
                 visual_vad_distill="vad", visual_intensity_distill="intensity")
    required = required_visual_heads(model._generation_loss_settings)
    router = getattr(model, "condition_router", None)
    if router is not None and router.mode in ('mean','mean_memory'):
        from emotion_ssm.models.visual_affect_teacher import observer_digest
        identity=dict(observer=observer_digest(model.observer),state=observer_digest(model.state_model),
                      variant=model.variant,context_dim=model.state_model.context_dim)
        if router.mean_provenance.get('coordinate')!=identity:
            raise ValueError('Training mean belongs to a different observer/state producer; recompute on train')
    if router is not None and router.mode in ("actual_semantic", "oracle_visual_pseudo"):
        required.extend(router.heads)
    if required:
        if teacher is None:
            raise ValueError("Visual/semantic supervision requires a validated frozen FLAME teacher")
        teacher.require_validated(required)
        teacher.require_coordinate(model.observer)
    return [p for p in model.parameters() if p.requires_grad]


def generation_auxiliary_enabled(model, train_config):
    upstream_trainable = any(p.requires_grad for module in (model.observer, model.state_model)
                             for p in module.parameters())
    return upstream_trainable and any(float(train_config.get(key, 0.)) > 0
        for key in ('coordinate_weight', 'future_weight', 'masked_weight'))


def rebind_training_baseline(model, state):
    """Start a new baseline graph after truncation or checkpoint restoration.

    The trainer owns this operation. Streaming inference keeps the baseline in
    its supplied session memory, including a caller's custom baseline. History
    tensors stay detached at the TBPTT boundary; only the learned baseline gets
    a fresh parameter connection. Do not mutate the caller's state: AMP retries
    must be able to reuse the same numeric history.
    """
    if (state is None or state.emotion is None or model.variant not in ("self", "dyadic")
            or not model.state_model.baseline.requires_grad):
        return state
    return replace(state, emotion=model.state_model.rebind_trainable_baseline(state.emotion))


class SegmentCursor:
    """Exact, no-replacement rank shards with a restartable chronological cursor.

    Dialogues are shuffled between epochs, never their one-second packets. Rank
    shards do not pad by repeating examples. Optimizer steps count only packets
    having a valid new target, but missing-target packets still update memory.
    """
    def __init__(self, dataset, seed=6666, rank=0, world_size=1, seen=0):
        if len(dataset) < world_size:
            raise ValueError("There must be at least one dialogue per rank")
        self.dataset, self.seed = dataset, int(seed)
        self.rank, self.world_size = int(rank), int(world_size)
        self.seen = 0
        self.iterator = self._iterate()
        self._pending = deque()
        for _ in range(int(seen)):
            self.next()

    def _iterate(self):
        epoch = 0
        while True:
            order = list(range(len(self.dataset)))
            random.Random(self.seed + epoch).shuffle(order)
            produced = False
            for index in order[self.rank::self.world_size]:
                for packet, target, valid in self.dataset.packets(index):
                    produced = True
                    packet = dict(packet)
                    packet["session_id"] = f"{epoch}:{packet['session_id']}"
                    yield packet, target, valid
            if not produced:
                raise ValueError("Training shard has no usable packets")
            epoch += 1

    def next(self):
        result = self._pending.popleft() if self._pending else next(self.iterator)
        self.seen += 1
        return result

    def peek(self, count):
        """Read future LABEL packets without advancing the saved input cursor."""
        while len(self._pending) < count:
            self._pending.append(next(self.iterator))
        return list(self._pending)[:count]

    def take_valid(self, count, max_packets=10000):
        if count < 1:
            raise ValueError("An optimizer step needs at least one valid block per rank")
        packets, valid_chunks = [], 0
        while valid_chunks < count:
            item = self.next()
            packets.append(item)
            valid_chunks += bool(item[2].any())
            if len(packets) >= max_packets and valid_chunks < count:
                raise ValueError("Training shard has insufficient valid targets; inspect preparation diagnostics")
        return packets


@dataclass
class SegmentRecord:
    packet: dict
    generated: torch.Tensor
    target: torch.Tensor
    mask: torch.Tensor
    state: object
    diagnostics: dict

    @property
    def observations(self):
        return self.diagnostics["observations"]


class GenerationTokenDataset:
    """Compatibility constructor delegating packet semantics to the data layer."""
    def __new__(cls, token_root, raw_root, split="train"):
        from emotion_ssm.data.packets_v3 import TokenDualTalk
        dataset = TokenDualTalk(raw_root, token_root, split)
        if not len(dataset):
            raise ValueError(f"No DualTalk token dialogues in {split}")
        return dataset


class JointAuxiliary:
    """Fixed-teacher coordinate retention and time-indexed autonomous forecasts.

    Lookahead packets are labels only. They are never advanced through the live
    state, never supply events/actions to forecast, and never enter generation.
    """
    def __init__(self, model, train_config):
        self.model, self.config = model, train_config
        self.targets, self.masked_valid = {}, {}
        teacher = getattr(model, "teacher", None)
        if teacher is None and model.variant != "none":
            raise ValueError("Joint v3 training requires an embedded frozen coordinate teacher")
        self.teacher = teacher
        self.horizons = (tuple(float(value) for value in train_config["forecast_seconds"])
                         if model.variant in ("self", "dyadic") else ())

    @torch.no_grad()
    def prepare(self, packets, device, input_count=None):
        self.targets, self.masked_valid = {}, {}
        if self.model.variant == "none":
            self.global_counts = torch.zeros(3, dtype=torch.float64, device=device)
            return
        input_count = len(packets) if input_count is None else int(input_count)
        if not self.horizons:
            packets = packets[:input_count]
        self.teacher.eval()
        for packet, _, _ in packets:
            key = (str(packet["session_id"]), round(float(packet["time"]), 6))
            values, valid, available = [], [], []
            for prefix in ("target", "partner"):
                features = map_tensors(packet[prefix + "_features"], lambda value: value.to(device))
                for name, value in list(features.items()):
                    if name.endswith("_tokens"):
                        features[name] = value.to(dtype=next(self.teacher.parameters()).dtype)
                observation = self.teacher(features)
                values.append(observation.aff.detach())
                fresh = getattr(observation, "fresh_observation", observation.modality_mask)
                valid.append(fresh.any(-1))
                available.append(observation.modality_mask.any(-1))
            self.targets[key] = (torch.stack(values, 1), torch.stack(valid, 1))
            self.masked_valid[key] = torch.stack(available, 1)
        counts = [0, 0, 0]
        for packet, _, _ in packets[:input_count]:
            session, now = str(packet["session_id"]), float(packet["time"])
            counts[0] += int(self.targets[(session, round(now, 6))][1].sum())
            counts[2] += int(self.masked_valid[(session, round(now, 6))].sum())
            for horizon in self.horizons:
                key = (session, round(now + horizon, 6))
                if key in self.targets:
                    counts[1] += int(self.targets[key][1].sum())
        self.global_counts = torch.tensor(counts, dtype=torch.float64, device=device)
        if distributed():
            dist.all_reduce(self.global_counts)

    def __call__(self, records):
        from torch.nn import functional as F
        zero = records[0].generated.sum() * 0.
        if self.model.variant == "none":
            return {"total": zero, "coordinate": zero, "future": zero, "masked": zero,
                    "coordinate_count": 0, "future_count": 0, "masked_count": 0,
                    "normalization": "global_sum"}
        coordinate, future, coordinate_count, future_count = zero, zero, 0, 0
        masked, masked_count = zero, 0
        for record in records:
            session, now = str(record.packet["session_id"]), float(record.packet["time"])
            label, valid = self.targets[(session, round(now, 6))]
            actual = torch.stack([observation.aff for observation in record.observations], 1)
            if valid.any():
                error = 1-F.cosine_similarity(actual.float(), label.float(), dim=-1)
                coordinate = coordinate + error[valid].sum()
                coordinate_count += int(valid.sum())
            if float(self.config.get("masked_weight", 0.)):
                mask_valid = self.masked_valid[(session, round(now, 6))]
                for role, features in enumerate(record.diagnostics["observation_inputs"]):
                    if mask_valid[:, role].any():
                        # Roles/valid blocks differ per rank. Global statistical
                        # collectives belong to synchronous A0 batches only.
                        result = self.model.observer.masked_loss(features, teacher=self.teacher, distributed=False,
                                                                 masking=self.config.get("masking"))
                        masked = masked + result["total"]
                        masked_count += int(mask_valid[:, role].sum())
            available = [(horizon, self.targets[(session, round(now + horizon, 6))]) for horizon in self.horizons
                         if (session, round(now + horizon, 6)) in self.targets]
            if available:
                predicted = self.model.state_model.forecast(record.state.emotion, [entry[0] for entry in available],
                                                           enable_partner=self.model.variant == "dyadic")
                for state, (_, (target, present)) in zip(predicted, available):
                    if present.any():
                        affect = self.model.state_model.affect(state).float()
                        error = 1-F.cosine_similarity(affect, target.float(), dim=-1)
                        error = error + F.smooth_l1_loss(affect, target.float(), reduction="none").mean(-1)
                        future = future + error[present].sum()
                        future_count += int(present.sum())
        return {"total": float(self.config["coordinate_weight"]) * coordinate / self.global_counts[0].clamp_min(1)
                         + float(self.config["future_weight"]) * future / self.global_counts[1].clamp_min(1)
                         + float(self.config.get("masked_weight", 0.)) * masked / self.global_counts[2].clamp_min(1),
                "coordinate": coordinate / max(1, coordinate_count), "future": future / max(1, future_count),
                "masked": masked / max(1, masked_count),
                "coordinate_count": coordinate_count, "future_count": future_count, "masked_count": masked_count,
                "normalization": "global_sum"}


def reconstruction_numerators(generated, target, mask):
    """Differentiable SSE; masked NaN targets cannot contaminate a valid loss."""
    # Accumulate MSE in FP32 even when the generator runs under FP16 autocast.
    # Fractional tail masks must not round global denominators to FP16 integers.
    generated, target = generated.float(), target.float()
    target = torch.where(mask[..., None], target, generated.detach())
    difference = generated - target
    edges = mask[:, 1:] & mask[:, :-1]
    velocity_error = difference[:, 1:] - difference[:, :-1]
    return {
        "expression": torch.where(mask[..., None], difference[..., :50].square(), 0.).sum(),
        "jaw": torch.where(mask[..., None], difference[..., 50:53].square(), 0.).sum(),
        "neck": torch.where(mask[..., None], difference[..., 53:56].square(), 0.).sum(),
        "velocity": torch.where(edges[..., None], velocity_error.square(), 0.).sum(),
    }


def _counts(packets, device):
    frames = sum(int(valid.sum()) for _, _, valid in packets)
    edges = sum(int((valid[:, 1:] & valid[:, :-1]).sum()) for _, _, valid in packets)
    blocks = sum(int(valid.any(1).sum()) for _, _, valid in packets)
    values = torch.tensor([frames * 50, frames * 3, frames * 3, edges * 56, blocks],
                          dtype=torch.float64, device=device)
    if distributed():
        dist.all_reduce(values)
    if not values[4]:
        raise ValueError("Training step has no valid target blocks")
    return values


def _synchronize_gradients(parameters):
    """SUM because losses have already been divided by global element counts."""
    if distributed():
        for parameter in parameters:
            if parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter)
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)


def train_segment(model, packets, optimizer, *, state=None, device=None, tbptt_steps=32,
                  scaler=None, amp=False, grad_clip=1., auxiliary_loss=None,
                  auxiliary_weight=1., optimize=True, defer_metrics=False, generation_loss_config=None):
    """Run one equal-budget optimizer step with boundary-only graph truncation.

    ``packets`` is a list of (input, target, valid_frame_mask). Forward never sees
    target FLAME. ``auxiliary_loss(records)`` optionally returns a local *mean*
    scalar (or dictionary containing ``total``); it can use future records only
    as detached labels, not as additional inputs to earlier predictions. Its mean
    is weighted by the segment's valid blocks/global valid blocks. Each segment
    is backpropagated exactly once; optimizer updates happen after all segments.

    Distributed callers use an unwrapped model with initially broadcast weights.
    All ranks call this once per optimizer step, even when dialogue lengths differ.
    Gradient sums include all trainable parameters and no DDP duplicate examples.
    """
    if tbptt_steps < 2:
        raise ValueError("Joint v3 training needs TBPTT segments longer than one block")
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        raise ValueError("Use the unwrapped model: train_segment performs exact gradient SUM explicitly")
    packets = list(packets)
    if not packets:
        raise ValueError("Empty training step")
    if device is None:
        device = next(model.parameters()).device
    device = torch.device(device)
    counts = _counts(packets, device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if optimize:
        optimizer.zero_grad(set_to_none=True)
    set_training_mode(model)
    if state is not None:
        state = state.to(device).detach()
    bank = state if isinstance(state, ConversationStates) else None
    objective = loss_settings(generation_loss_config if generation_loss_config is not None else
                              getattr(model, "_generation_loss_settings", None))
    visual_teacher = getattr(model, "visual_teacher", None)
    router = getattr(model, "condition_router", None)
    oracle = router is not None and router.mode == "oracle_visual_pseudo"
    visual_enabled = any(objective[n+"_weight"]>0 for n in VISUAL_NAMES)
    semantic_mode=router is not None and router.mode in ('actual_semantic','oracle_visual_pseudo')
    if visual_enabled or semantic_mode:
        if visual_teacher is None:
            raise ValueError("Visual objectives require a frozen, validated teacher")
        required = required_visual_heads(objective)
        visual_teacher.require_validated(required + (list(router.heads) if semantic_mode else []))
        visual_teacher.require_coordinate(model.observer)
    boundary_cache = dict(getattr(bank,'boundary_for_loss',{})) if bank is not None else {}
    if bank is None and state is not None and getattr(state, 'boundary_for_loss', None) is not None:
        edge = state.boundary_for_loss
        boundary_cache[(edge.session, edge.roles)] = edge
    extra_counts = torch.tensor(plan_objective_counts(packets, boundary_cache, objective),dtype=torch.float64,device=device)
    if distributed():
        dist.all_reduce(extra_counts)
    objective_sums = torch.zeros(5, device=device, dtype=torch.float64)
    if bank is not None:
        if any(p.requires_grad for module in (model.observer, model.state_model) for p in module.parameters()):
            raise ValueError("Interleaved generation requires frozen upstream state; joint TBPTT is a separate protocol")
        state = None
    metrics = DeferredReconstructionTotals(device) if defer_metrics else ReconstructionTotals()
    accumulated, auxiliary_total, segment_count = 0., 0., 0
    auxiliary_counts = {"coordinate_count": 0., "future_count": 0., "masked_count": 0.}
    previous = None
    use_amp = bool(amp and device.type == "cuda")
    for start in range(0, len(packets), tbptt_steps):
        state = rebind_training_baseline(model, state)
        records, segment_loss = [], None
        for raw_packet, raw_target, raw_valid in packets[start:start + tbptt_steps]:
            prefix=raw_packet.get('training_prefix')
            packet = map_tensors({k:v for k,v in raw_packet.items() if k!='training_prefix'}, lambda value: value.to(device))
            target, valid = raw_target.to(device), raw_valid.to(device).bool()
            if bank is not None:
                key = session_key(packet)
                state, previous = bank.states.get(key), bank.previous.get(key)
            if prefix is not None:
                if bank is None or any(p.requires_grad for module in (model.observer,model.state_model) for p in module.parameters()):
                    raise ValueError('Prefix-warmed windows require a frozen upstream conversation bank')
                if oracle:
                    raise ValueError('Target-derived oracle histories cannot enter causal prefix prewarming')
                state=None
                with torch.no_grad():
                    for past in prefix:
                        _,state,_=model(map_tensors(past,lambda value:value.to(device)),state,observe_only=True)
            key = session_key(packet)
            edge = matching_boundary(packet, valid.shape[1], boundary_cache.get(key))
            previous = None if edge is None else (edge.prediction.detach(), edge.target, edge.valid)
            if state is not None and (state.session_id != str(packet["session_id"]) or state.roles != tuple(packet["roles"])):
                state, previous = None, None
            with torch.autocast(device.type, enabled=use_amp):
                reference = None
                visual_valid = visual_window_mask(target, valid, objective)
                if visual_enabled or oracle:
                    with torch.no_grad():
                        reference = visual_teacher(target, valid, domain_id=packet.get('source_domain',2), now=packet['time'])
                if oracle:
                    from emotion_ssm.models.condition_router import semantic_code
                    if not visual_valid.all() and valid.any():
                        raise ValueError("Oracle training requires complete qualified target windows")
                    override=semantic_code(reference,router.heads)
                    if not visual_valid.all():override=torch.zeros_like(override)
                    generated, state, diagnostics = model(packet,state,
                        condition_override=override,diagnostic=True)
                else:
                    generated, state, diagnostics = model(packet, state)
                sums = reconstruction_numerators(generated, target, valid)
                loss = sum(sums[name] / counts[index].clamp_min(1).to(sums[name].dtype)
                           for index, name in enumerate(("expression", "jaw", "neck", "velocity")))
                boundary_sse, _ = boundary_loss_terms(generated, target, valid, edge)
                objective_sums[0] += boundary_sse.detach().double()
                if objective['boundary_weight']:
                    loss = loss + objective['boundary_weight']*boundary_sse/extra_counts[0].clamp_min(1).float()
                if visual_enabled:
                    predicted = visual_teacher(generated, valid, domain_id=packet.get('source_domain',2), now=packet['time'])
                    visual_sums = visual_loss_terms(predicted,reference,visual_valid,objective['temperature'],objective['vad_dimensions'])
                    for i,name in enumerate(VISUAL_NAMES):
                        objective_sums[i+1] += visual_sums[name].detach().double()
                        if objective[name+'_weight']:
                            loss = loss+objective[name+'_weight']*visual_sums[name]/extra_counts[1].clamp_min(1).float()
            segment_loss = loss if segment_loss is None else segment_loss + loss
            records.append(SegmentRecord(packet, generated, target, valid, state, diagnostics))
            metrics.update(generated, torch.nan_to_num(target), valid, previous)
            previous = (generated[:, -1].detach(), torch.nan_to_num(target[:, -1]).detach(), valid[:, -1])
            edge = boundary_frame(packet,generated if objective['boundary_weight'] else generated.detach(),target,valid)
            if packet.get('training_session_end',False):
                boundary_cache.pop(key,None)
            else:
                boundary_cache[key] = edge
            if bank is not None:
                if packet.get("training_session_end", False):
                    bank.states.pop(key, None)
                    bank.previous.pop(key, None)
                else:
                    bank.states[key], bank.previous[key] = state, previous
        if auxiliary_loss is not None:
            with torch.autocast(device.type, enabled=use_amp):
                extra = auxiliary_loss(records)
                already_global = isinstance(extra, dict) and extra.get("normalization") == "global_sum"
                if isinstance(extra, dict):
                    for name in auxiliary_counts:
                        auxiliary_counts[name] += float(extra.get(name, 0))
                    extra = extra["total"]
                local_blocks = sum(int(record.mask.any(1).sum()) for record in records)
                weight = 1. if already_global else local_blocks / counts[4]
                weighted = extra * weight * float(auxiliary_weight)
                segment_loss = segment_loss + weighted
                auxiliary_total += float(weighted.detach())
        if not torch.isfinite(segment_loss):
            raise FloatingPointError("Nonfinite joint training loss; optimizer was not advanced")
        if segment_loss.requires_grad:
            if scaler is None:
                segment_loss.backward()
            else:
                scaler.scale(segment_loss).backward()
        accumulated += float(segment_loss.detach())
        segment_count += 1
        state = state.detach()
        if bank is not None:
            bank = bank.detach()
        boundary_cache = map_tensors(boundary_cache, lambda value: value.detach())
        # The next segment retains numeric history but no graph from this one.
        del records, segment_loss
    _synchronize_gradients(parameters)
    if scaler is not None:
        scaler.unscale_(optimizer)
    finite = [torch.isfinite(parameter.grad).all() for parameter in parameters if parameter.grad is not None]
    if finite and not bool(torch.stack(finite).all()):
        if scaler is not None and scaler.is_enabled():
            before = scaler.get_scale()
            # unscale_ recorded identical global overflow on every rank. step
            # therefore skips, and update performs the normal common backoff.
            scaler.step(optimizer)
            scaler.update()
            raise AMPGradientOverflow(f"Scaled gradient overflow: scale {before:g} -> {scaler.get_scale():g}; optimizer unchanged")
        raise FloatingPointError("Nonfinite synchronized gradients; optimizer was not advanced")
    component_gradients = {}
    film_parameters=[p for p in model.generator.film.parameters() if p.requires_grad and p.grad is not None]
    component_gradients['film_grad_norm_before_clip']=float(torch.stack([p.grad.detach().float().square().sum()
        for p in film_parameters]).sum().sqrt()) if film_parameters else 0.
    norms_before = []
    for name, module in (("generator", model.generator), ("observer", model.observer), ("state", model.state_model)):
        component_parameters = [p for p in module.parameters() if p.requires_grad and p.grad is not None]
        if name=='generator' and router is not None:
            component_parameters += [p for p in router.parameters() if p.requires_grad and p.grad is not None]
        if grad_clip and component_parameters:
            before = torch.nn.utils.clip_grad_norm_(component_parameters, grad_clip, error_if_nonfinite=True)
        else:
            terms = [p.grad.detach().float().square().sum() for p in component_parameters]
            before = torch.stack(terms).sum().sqrt() if terms else torch.zeros((), device=device)
        norms_before.append(before)
        component_gradients[name + "_grad_norm_before_clip"] = float(before)
        terms = [parameter.grad.detach().float().square().sum() for parameter in component_parameters]
        component_gradients[name + "_grad_norm"] = float(torch.stack(terms).sum().sqrt()) if terms else 0.
    grad_norm = torch.stack(norms_before).square().sum().sqrt()
    if optimize:
        if scaler is None:
            optimizer.step()
        else:
            scaler.step(optimizer)
            scaler.update()
    metrics.distributed_sum(device)
    if distributed():
        dist.all_reduce(objective_sums)
    scalar = torch.tensor([accumulated, auxiliary_total, auxiliary_counts["coordinate_count"], auxiliary_counts["future_count"],
                           auxiliary_counts["masked_count"]],
                          dtype=torch.float64, device=device)
    if distributed():
        dist.all_reduce(scalar)
    result = {**metrics.metrics(), "loss": float(scalar[0]), "auxiliary_loss": float(scalar[1]),
              "optimization_protocol": OPTIMIZATION_PROTOCOL,
              "generation_revision": GENERATION_REVISION,
              "global_valid_blocks": int(counts[4]), "tbptt_segments_rank": segment_count,
              "grad_norm": float(grad_norm), "component_grad_norm_stage": "after_independent_component_clip",
              "generator_clip_threshold": float(grad_clip),
              "coordinate_count": int(scalar[2]), "future_count": int(scalar[3]), "masked_count": int(scalar[4]),
              "error_elements": dict(metrics.elements), **component_gradients}
    result.update(train_total_loss=result['loss'], generation_total_with_boundary=result['generation_total']+
                  objective['boundary_weight']*result['boundary_velocity_mse'],
                  boundary_loss=float(objective_sums[0]/extra_counts[0].clamp_min(1)),
                  visual_valid_windows=int(extra_counts[1]) if visual_enabled else 0)
    for i,name in enumerate(VISUAL_NAMES):
        result[name+'_loss'] = float(objective_sums[i+1]/extra_counts[1].clamp_min(1)) if visual_enabled else None
    if bank is not None:
        bank.boundary_for_loss = boundary_cache
    elif state is not None and (objective['boundary_weight'] or visual_enabled):
        state.boundary_for_loss = boundary_cache.get((state.session_id,state.roles))
    return bank if bank is not None else state, result


@torch.no_grad()
def evaluate(model, dataset, device, rank=0, world_size=1, max_dialogues=0,
             selected_names=None, max_blocks=0, representation_diagnostics=True):
    """Full chronological validation; no padded/repeated distributed samples."""
    model.eval()
    indices = list(range(len(dataset)))
    if selected_names is not None:
        lookup = {name: index for index, name in enumerate(dataset.names)}
        if len(set(selected_names)) != len(selected_names) or set(selected_names) - set(lookup):
            raise ValueError("Diagnostic manifest has duplicate or unavailable conversations")
        indices = [lookup[name] for name in selected_names]
    if max_dialogues:
        indices = indices[:max_dialogues]
    totals, records, affect_samples = ReconstructionTotals(), [], []
    objective=loss_settings(getattr(model,'_generation_loss_settings',None))
    visual_teacher=getattr(model,'visual_teacher',None)
    visual_enabled=any(objective[n+'_weight']>0 for n in VISUAL_NAMES)
    router=getattr(model,'condition_router',None)
    oracle=router is not None and router.mode=='oracle_visual_pseudo'
    required=required_visual_heads(objective)
    if router is not None and router.mode in ('actual_semantic','oracle_visual_pseudo'):
        required+=list(router.heads)
    if required:
        if visual_teacher is None:
            raise ValueError('Visual evaluation requires a validated teacher')
        visual_teacher.require_validated(required)
        visual_teacher.require_coordinate(model.observer)
    visual_totals=torch.zeros(5,device=device,dtype=torch.float64)
    representation_samples, preservation = {}, {}
    from emotion_ssm.utils.reconstruction import ExpressionDecomposition
    for index in indices[rank::world_size]:
        local, state, previous = ReconstructionTotals(), None, None
        edge=None
        local_visual=torch.zeros(5,device=device,dtype=torch.float64)
        decomposition = ExpressionDecomposition(device)
        for chunk_index, (packet, target, valid) in enumerate(dataset.packets(index)):
            if max_blocks and chunk_index >= max_blocks:
                break
            packet = map_tensors(packet, lambda value: value.to(device))
            target, valid = target.to(device), valid.to(device).bool()
            if visual_enabled or oracle:
                reference=visual_teacher(target,valid,domain_id=packet.get('source_domain',2),now=packet['time'])
            if oracle:
                from emotion_ssm.models.condition_router import semantic_code
                eligible=visual_window_mask(target,valid,objective)
                if not eligible.all() and valid.any():
                    raise ValueError('Oracle evaluation requires qualified target windows')
                override=semantic_code(reference,router.heads)
                if not eligible.all():override=torch.zeros_like(override)
                generated,state,diagnostics=model(packet,state,
                    condition_override=override,diagnostic=True)
            else:
                generated, state, diagnostics = model(packet, state)
            edge=matching_boundary(packet,valid.shape[1],edge)
            previous=None if edge is None else (edge.prediction,edge.target,edge.valid)
            local.update(generated, target, valid, previous)
            if visual_enabled:
                mask=visual_window_mask(target,valid,objective)
                predicted=visual_teacher(generated,valid,domain_id=packet.get('source_domain',2),now=packet['time'])
                terms=visual_loss_terms(predicted,reference,mask,objective['temperature'],objective['vad_dimensions'])
                local_visual+=torch.stack([terms[n] for n in VISUAL_NAMES]+[mask.sum()]).double()
            decomposition.update(generated, target, valid)
            if representation_diagnostics and index < 16 and chunk_index < 32:
                affect_samples.append(torch.stack([diagnostics["target_aff"], diagnostics["partner_aff"]], 1).cpu())
                for role, features in zip(("target", "partner"), diagnostics["observation_inputs"]):
                    reference = model.teacher(features) if hasattr(model, "teacher") else None
                    domain = int(features.get("domain_id", torch.tensor([2]))[0])
                    for subset in ("A", "AT", "AVT"):
                        observed = model.observer(features, subset=subset)
                        present = observed.modality_mask.any(-1)
                        key = f"domain{domain}_{role}_{subset}"
                        if present.any():
                            representation_samples.setdefault(key, []).append(observed.aff[present].float().cpu())
                        if reference is not None:
                            compatible = present & reference.modality_mask.any(-1)
                            error = 1-torch.nn.functional.cosine_similarity(observed.aff.float(), reference.aff.float(), dim=-1)
                            old_sum, old_count = preservation.get(key, (0., 0))
                            preservation[key] = (old_sum + float(error[compatible].sum()), old_count + int(compatible.sum()))
            previous = (generated[:, -1], target[:, -1], valid[:, -1])
            edge=boundary_frame(packet,generated,target,valid)
        totals.merge(local)
        visual_totals+=local_visual
        names = getattr(dataset, "names", None)
        records.append({"dialogue": names[index] if names is not None else str(index),
                        "sse": dict(local.square_error), "elements": dict(local.elements),
                        **local.metrics(), **decomposition.metrics(),
                        'visual_error_sums':{n:float(local_visual[i]) for i,n in enumerate(VISUAL_NAMES)},
                        'visual_valid_windows':int(local_visual[-1])})
    totals.distributed_sum(device)
    if distributed():
        gathered = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, records)
        records = [record for shard in gathered for record in shard]
        gathered_affect = [None] * dist.get_world_size()
        dist.all_gather_object(gathered_affect, affect_samples)
        affect_samples = [sample for shard in gathered_affect for sample in shard]
        gathered_diagnostics = [None] * dist.get_world_size()
        dist.all_gather_object(gathered_diagnostics, (representation_samples, preservation))
        representation_samples, preservation = {}, {}
        for samples, errors in gathered_diagnostics:
            for key, values in samples.items():
                representation_samples.setdefault(key, []).extend(values)
            for key, (total, count) in errors.items():
                old_sum, old_count = preservation.get(key, (0., 0))
                preservation[key] = (old_sum + total, old_count + count)
    metrics = {**totals.metrics(), "error_elements": dict(totals.elements)}
    if distributed():
        dist.all_reduce(visual_totals)
    metrics['generation_total_with_boundary']=metrics['generation_total']+objective['boundary_weight']*metrics['boundary_velocity_mse']
    metrics['visual_valid_windows']=int(visual_totals[-1])
    for i,n in enumerate(VISUAL_NAMES):
        metrics[n+'_loss']=float(visual_totals[i]/visual_totals[-1]) if visual_totals[-1]>0 else None
    metrics['gold_emotion_metrics']=None
    for name in ("expression_bias_mse", "expression_centered_mse", "expression_pred_variance", "expression_target_variance"):
        metrics[name] = sum(row[name] * row["valid_frames"] for row in records) / max(1, metrics["valid_frames"])
    if affect_samples:
        values = torch.cat(affect_samples, 0).float()
        for role, name in enumerate(("target", "partner")):
            current = values[:, role]
            normalized = torch.nn.functional.normalize(current, dim=-1)
            count = len(current)
            similarity = ((normalized.sum(0).square().sum() - normalized.square().sum()) /
                          max(1, count * (count-1)))
            metrics[name + "_affect_std"] = float(current.std(0, unbiased=False).mean())
            metrics[name + "_affect_similarity"] = float(similarity)
    representations = {}
    for key, parts in representation_samples.items():
        values = torch.cat(parts).float()
        normalized = torch.nn.functional.normalize(values, dim=-1)
        count = len(values)
        error, error_count = preservation.get(key, (0., 0))
        representations[key] = {"std": float(values.std(0, unbiased=False).mean()),
            "cosine": float((normalized.sum(0).square().sum()-normalized.square().sum()) / max(1, count*(count-1))),
            "samples": count, "teacher_error": error/max(1, error_count), "teacher_samples": error_count}
    metrics["representation_diagnostics"] = representations
    metrics["diagnostic_sampling"] = "first 16 directed dialogues and at most 32 blocks per dialogue"
    if not metrics["valid_frames"]:
        raise ValueError("Validation has no valid target frames")
    return metrics, records


def run(config, stop_after=None):
    """Run a complete joint generation experiment under torchrun or one process."""
    destination=Path(config['paths']['output'])
    if not config['paths'].get('resume') and any((destination/name).exists() for name in ('last.pt','metrics.jsonl','train_metrics.jsonl')):
        raise ValueError('Output already contains training results; use a new experiment directory or explicit compatible resume')
    import math
    import os
    import shutil
    import time
    from emotion_ssm.config_v3 import validate_config, write_config
    from emotion_ssm.utils.checkpoint import capture_rng_state, restore_rng_state
    from emotion_ssm.utils.checkpoint_v3 import (build_avatar, load_avatar, read_checkpoint, save_checkpoint,
                                                  manifest_provenance, require_training_revision)
    from emotion_ssm.utils.distributed import init_distributed
    from emotion_ssm.train.staged_dynamics_support import weight_digest
    validate_config(config)
    requested_execution = dict(config.get("generation_execution", {}))
    payload = read_checkpoint(config["paths"]["resume"]) if config["paths"].get("resume") else None
    if payload:
        # Reject changed gradient semantics before allocating models or joining
        # distributed collectives. Loading old weights for inference is separate.
        require_training_revision(payload)
        from emotion_ssm.utils.checkpoint_v3 import require_generation_objective_match
        require_generation_objective_match(payload['config'],config)
        if payload["run_state"].get("optimization_protocol") != OPTIMIZATION_PROTOCOL:
            raise ValueError("Optimizer objective/clipping protocol changed; use old weights as explicit initialization, not resume")
        requested_stability = config.get("generation_stability")
        if requested_stability is not None and requested_stability != payload["config"].get("generation_stability"):
            raise ValueError("Sampling/masking/learning-rate protocol changed; initialize a new experiment")
        if payload["run_state"].get("stability_protocol") != payload["config"].get("generation_stability", {}).get("protocol"):
            raise ValueError("Checkpoint stability protocol is inconsistent")
    authoritative = payload["config"] if payload else config
    train_config = authoritative["train"]
    context = init_distributed(train_config["device"], train_config["seed"], train_config["deterministic"])
    device = context.device
    initial_metrics = {}
    if payload:
        model, config, payload = load_avatar(payload, device)
        if payload["provenance"] != manifest_provenance(config):
            raise ValueError("Data/source/split changed; start a new experiment instead of resuming")
    elif config["paths"].get("avatar_initialization"):
        initialization_payload = torch.load(config["paths"]["avatar_initialization"], map_location="cpu", weights_only=False, mmap=True)
        if initialization_payload["kind"] != "streaming_avatar_v3":
            raise ValueError("Avatar initialization must be a complete generator checkpoint")
        requested_provenance=manifest_provenance(config)
        same_base=lambda p:{k:v for k,v in p.items() if not k.startswith('continuous:')}
        timeline_adaptation=(bool(config['data'].get('continuous_timelines')) and
            same_base(initialization_payload['provenance'])==same_base(requested_provenance))
        if initialization_payload["provenance"] != requested_provenance and not timeline_adaptation:
            raise ValueError("Initialization data/source/split differs from the controlled experiment")
        if (initialization_payload["config"]["generation"]["variant"] != config["generation"]["variant"]
                and not config.get('experiment',{}).get('controlled_condition_initialization',False)):
            raise ValueError("Use a separate explicit protocol when changing initialization conditioning")
        model = build_avatar(config, device=device, construction=initialization_payload["construction"], initialize=False)
        if config['generation'].get('condition_routing') is not None or config['generation'].get('visual_teacher') is not None:
            from emotion_ssm.utils.checkpoint_v3 import initialize_avatar_weights
            initialize_avatar_weights(model,initialization_payload,config)
        else:
            model.load_state_dict(initialization_payload["models"]["system"], strict=True)
        same_route=(initialization_payload['config']['generation'].get('condition_routing')==config['generation'].get('condition_routing')
                    and initialization_payload['config']['generation']['variant']==config['generation']['variant'])
        initial_metrics = dict(initialization_payload["metrics"]) if same_route and initialization_payload['provenance']==requested_provenance else {}
        if initial_metrics:
            initial_metrics['generation_total_with_boundary']=initial_metrics['generation_total']+loss_settings(config.get('generation_losses'))['boundary_weight']*initial_metrics['boundary_velocity_mse']
        del initialization_payload
    else:
        model = build_avatar(config, device)
    execution = generation_execution_settings(config, {"generation_execution": requested_execution})
    config["generation_execution"] = execution
    teacher = model.teacher
    parameters = configure_generation_from_config(model, config)
    model.state_model.configure_execution(config['generation'].get('dynamics_execution', 'reference'))
    frozen_condition = not any(p.requires_grad for module in (model.observer, model.state_model)
                               for p in module.parameters())
    condition_models = {'observer': model.observer, 'teacher': teacher, 'state': model.state_model}
    if context.enabled:
        for value in list(model.parameters()) + list(model.buffers()):
            dist.broadcast(value.data, 0)
    initial_condition_hash = weight_digest(condition_models) if frozen_condition else None
    # Cached-token generation never invokes the heavyweight raw extractor.
    # Keep its weights in the complete checkpoint while freeing GPU residency.
    if model.features is not None and config["generation"].get("offload_extractor", True):
        model.features.to("cpu")
    data_config = config["data"]
    training = GenerationTokenDataset(data_config["dualtalk_tokens"], data_config["dualtalk_raw"], "train")
    validation = GenerationTokenDataset(data_config["dualtalk_tokens"], data_config["dualtalk_raw"], "val")
    if data_config.get('continuous_timelines'):
        from emotion_ssm.data.continuous_dialogues import ContinuousDialogueDataset
        timelines=data_config['continuous_timelines']
        if 'train' not in timelines or 'val' not in timelines:
            raise ValueError('Continuous training needs independent train and val timelines')
        training=ContinuousDialogueDataset(training,timelines['train'])
        validation=ContinuousDialogueDataset(validation,timelines['val'])
        if set(training.source_ids)&set(validation.source_ids):
            raise ValueError('Continuous train and validation source identities overlap')
    if config['generation'].get('condition_routing',{}).get('mode') in ('actual_semantic','oracle_visual_pseudo'):
        from emotion_ssm.data.semantic_windows import SemanticWindowDataset
        training=SemanticWindowDataset(training,config.get('generation_losses'))
        validation=SemanticWindowDataset(validation,config.get('generation_losses'))
    output = Path(config["paths"]["output"])
    if context.is_main:
        output.mkdir(parents=True, exist_ok=True)
        write_config(output / "config.json", config)
    context.barrier()
    initialization = dict(event='generation_initialized', rank=context.rank, pid=os.getpid(),
        device=str(device), world_size=context.world_size, variant=model.variant,
        frozen_condition=frozen_condition, condition_hash=initial_condition_hash,
        dynamics_checkpoint=config['paths']['dynamics_checkpoint'],
        trainable_parameters={name:sum(p.numel() for p in module.parameters() if p.requires_grad)
            for name,module in (('generator',model.generator),('observer',model.observer),('state',model.state_model))},
        train_dialogues=len(training), validation_dialogues=len(validation),
        execution_protocol=EXECUTION_PROTOCOL, generation_execution=execution)
    print(json.dumps(initialization), flush=True)
    if context.is_main:
        (output/'initialization.json').write_text(json.dumps(initialization,indent=2),encoding='utf-8')
    train_config = config["train"]
    final_step = int(train_config["max_steps"])
    if stop_after is not None:
        final_step = min(final_step, int(stop_after))
    optimizer = torch.optim.AdamW(optimizer_groups(model, config), weight_decay=train_config["weight_decay"], foreach=False)
    scaler = torch.amp.GradScaler("cuda", enabled=train_config["amp"] and device.type == "cuda")
    start, best, saved, state = 0, float("inf"), {}, None
    best_reconstruction=float('inf')
    if payload:
        run_state = payload["run_state"]
        if run_state["world_size"] != context.world_size or run_state["split_digest"] != training.manifest_digest:
            raise ValueError("Exact resume requires the same world size and training split")
        optimizer.load_state_dict(payload["optimizer"])
        for values in optimizer.state.values():
            for key, value in values.items():
                if torch.is_tensor(value) and key != "step":
                    values[key] = value.to(device)
        if payload.get("scaler"):
            scaler.load_state_dict(payload["scaler"])
        start, best = payload["global_step"], run_state["best_score"]
        best_reconstruction=run_state.get('best_reconstruction_score',float('inf'))
        saved = run_state["ranks"][context.rank]
        state = saved.get("stream_state")
        if state is not None:
            state = state.to(device)
    stability = config.get("generation_stability", {})
    if stability:
        if config.get('long_history',{}).get('enabled',False):
            from emotion_ssm.train.history_sampling import HistoryWindowCursor
            cursor=HistoryWindowCursor(training,train_config['seed'],context.rank,context.world_size,
                stability['conversations_per_rank'],stability['blocks_per_conversation'],
                config['long_history'],saved.get('sampling_state'))
            if context.is_main:
                (output/'history_sampling_coverage.json').write_text(json.dumps(cursor.coverage(),indent=2),encoding='utf-8')
        else:
            cursor = InterleavedCursor(training, train_config["seed"], context.rank, context.world_size,
                stability["conversations_per_rank"], stability["blocks_per_conversation"], saved.get("sampling_state"))
        state = state if state is not None else ConversationStates()
        if not isinstance(state, ConversationStates):
            raise ValueError("Interleaved resume requires a complete conversation state bank")
    else:
        cursor = SegmentCursor(training, train_config["seed"], context.rank, context.world_size, saved.get("packets_seen", 0))
    if saved.get("rng"):
        restore_rng_state(saved["rng"])
    validation_metrics = payload.get("metrics", {}) if payload else {}
    del payload
    global_blocks = int(train_config["global_chunks_per_step"])
    if global_blocks % context.world_size:
        raise ValueError("Global valid blocks must divide evenly over ranks")
    local_blocks = global_blocks // context.world_size
    if stability and local_blocks != stability["conversations_per_rank"] * stability["blocks_per_conversation"]:
        raise ValueError("World size and conversation slots do not match the valid-block budget")
    auxiliary = JointAuxiliary(model, train_config) if generation_auxiliary_enabled(model, train_config) else None
    if execution["prefetch_batches"]:
        from emotion_ssm.train.generation_prefetch import PrefetchSegmentCursor
        lookahead_count = (math.ceil(max(train_config["forecast_seconds"]))
                           if auxiliary is not None and model.variant in ("self", "dyadic") else 0)
        cursor = PrefetchSegmentCursor(cursor, local_blocks, lookahead_count)
    set_learning_rates(optimizer, config, start)
    if stability and not start:
        run_fixed_probes(model, training, validation, config, context, 0, output)
        # The source checkpoint's full validation is a retained candidate, not a
        # resumed optimizer. Probe replay and unchanged inference are tested by
        # the launch gate; report the provenance of this inherited score.
        if initial_metrics and stability.get("retain_initial_candidate", False):
            from emotion_ssm.utils.generation_selection import selection_score
            best = selection_score(initial_metrics,config.get('generation_selection'))
            best_reconstruction=float(initial_metrics['generation_total'])
            validation_metrics = initial_metrics
            if context.is_main:
                save_checkpoint(output / "best.pt", {"system": model}, config, model.construction_info,
                    "streaming_avatar_v3", step=0, metrics=initial_metrics,
                    run_state={"candidate_only": True, "score_origin": "initialization_checkpoint_full_validation",
                               "stability_protocol": STABILITY_PROTOCOL})
                shutil.copyfile(output/'best.pt',output/'best_reconstruction.pt')
                (output / "initial_candidate.json").write_text(json.dumps(dict(score=best,
                    checkpoint=config["paths"]["avatar_initialization"], new_optimizer_steps=0)), encoding="utf-8")
            context.barrier()
    began = time.monotonic()
    for step in range(start, final_step):
        step_began = time.monotonic()
        packets = cursor.take_valid(local_blocks)
        data_wait_seconds = time.monotonic() - step_began
        # Future token observations are detached labels. A peek leaves the live
        # input cursor and all state histories unchanged, including on resume.
        lookahead = (cursor.peek(math.ceil(max(train_config["forecast_seconds"])))
                     if auxiliary is not None and model.variant in ("self", "dyadic") else [])
        if auxiliary is not None:
            auxiliary.prepare(packets + lookahead, device, input_count=len(packets))
        step_state, attempt_rng = state, capture_rng_state()
        attempts = 0
        while True:
            restore_rng_state(attempt_rng)
            try:
                state, metrics = train_segment(model, packets, optimizer, state=step_state, device=device,
                                               tbptt_steps=int(train_config["tbptt_seconds"]), scaler=scaler,
                                               amp=train_config["amp"], grad_clip=train_config["clip_grad"], auxiliary_loss=auxiliary,
                                               defer_metrics=execution["defer_metrics"])
                metrics["amp_overflow_retries"] = attempts
                break
            except AMPGradientOverflow as error:
                attempts += 1
                if context.is_main:
                    print(json.dumps({"step": step+1, "amp_retry": attempts, "message": str(error)}), flush=True)
                if attempts >= int(train_config.get("max_amp_retries", 8)):
                    raise FloatingPointError("Repeated AMP overflow exhausted retries; no optimizer budget was counted") from error
        used_learning_rates = {group["name"]: group["lr"] for group in optimizer.param_groups}
        set_learning_rates(optimizer, config, step+1)
        if stability:
            from collections import Counter
            rows = [(row[0]["session_id"], row[0]["training_source_id"]) for row in packets if row[2].any()]
            if distributed():
                shards = [None] * context.world_size
                dist.all_gather_object(shards, rows)
                rows = [row for shard in shards for row in shard]
            counts = Counter(row[0] for row in rows)
            metrics.update(conversations_per_step=len(counts), sources_per_step=len({row[1] for row in rows}),
                dominant_conversation_fraction=max(counts.values()) / len(rows),
                active_states_rank=len(state.states), speech_mask=mask_diagnostics(model))
            if config.get('long_history',{}).get('enabled',False):
                metrics['history_ages_seconds']=[p['history_age_seconds'] for p,_,m in packets if m.any()]
        if context.is_main and ((step+1) % train_config["log_every"] == 0 or step == start):
            entry = {"step": step+1, **metrics, "lr": optimizer.param_groups[0]["lr"],
                     "learning_rates_used": used_learning_rates,
                     "step_seconds": time.monotonic()-step_began, "elapsed_seconds": time.monotonic()-began,
                     "frozen_condition": frozen_condition,
                     "execution_protocol": EXECUTION_PROTOCOL, "generation_execution": execution,
                     "resume_step": start, "data_wait_seconds": data_wait_seconds,
                     "peak_allocated_mib": torch.cuda.max_memory_allocated(device)/2**20 if device.type=='cuda' else 0.,
                     "peak_reserved_mib": torch.cuda.max_memory_reserved(device)/2**20 if device.type=='cuda' else 0.}
            with (output / "train_metrics.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
            print(json.dumps(entry), flush=True)
            (output / "training_status.json").write_text(json.dumps({"status": "running", "step": step+1,
                "max_steps": train_config["max_steps"], "variant": model.variant, "metrics": metrics}), encoding="utf-8")
        if stability and (step+1) % int(stability.get("diagnostic_every", 250)) == 0:
            run_fixed_probes(model, training, validation, config, context, step+1, output)
        if (step+1) % train_config["validate_every"] == 0 or step+1 == final_step:
            if frozen_condition and weight_digest(condition_models) != initial_condition_hash:
                raise RuntimeError('Frozen observer/teacher/dynamics weights changed during generation training')
            validation_metrics, records = evaluate(model, validation, device, context.rank, context.world_size,
                                                  train_config["validation_max_dialogues"])
            from emotion_ssm.utils.generation_selection import selection_score, conditioned_acceptance
            score = selection_score(validation_metrics,config.get('generation_selection'))
            validation_metrics['selection_score']=score
            reference_metrics=config.get('generation_selection',{}).get('reference_metrics',initial_metrics)
            validation_metrics['deployment_checks']=conditioned_acceptance(validation_metrics,reference_metrics,
                config.get('generation_selection',{}).get('conditioned_policy'))
            reconstruction_improved=validation_metrics['generation_total']<best_reconstruction
            best_reconstruction=min(best_reconstruction,validation_metrics['generation_total'])
            improved = score < best
            best = min(best, score)
            local = {"packets_seen": cursor.seen, "stream_state": state.detach().to("cpu"), "rng": capture_rng_state()}
            if stability:
                local["sampling_state"] = cursor.state_dict()
            ranks = [local]
            if context.enabled:
                ranks = [None] * context.world_size
                dist.all_gather_object(ranks, local)
            run_state = {"world_size": context.world_size, "ranks": ranks, "best_score": best,
                         'best_reconstruction_score':best_reconstruction,
                         "optimization_protocol": OPTIMIZATION_PROTOCOL,
                         "generation_revision": GENERATION_REVISION,
                         "split_digest": training.manifest_digest, "feature_digest": training.feature_digest,
                         "frozen_condition": frozen_condition, "condition_hash": initial_condition_hash,
                         "stability_protocol": stability.get("protocol"),
                         "optimizer_budget": {"global_new_blocks": global_blocks, "step": step+1}}
            if context.is_main:
                with (output / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({"step": step+1, **validation_metrics}, ensure_ascii=False) + "\n")
                from emotion_ssm.utils.generation_selection import reconstruction_frontier
                frontier_file=output/'reconstruction_frontier.json'
                frontier=json.loads(frontier_file.read_text(encoding='utf-8'))['candidates'] if frontier_file.exists() else []
                frontier=[row for row in frontier if row['step']!=step+1]
                frontier.append({'step':step+1,**{k:validation_metrics[k] for k in
                    ('expression_mse','jaw_mse','neck_mse','boundary_velocity_mse','generation_total')}})
                frontier_file.write_text(json.dumps(dict(candidates=reconstruction_frontier(frontier),
                    scope='validation metric frontier; weights retained by best/archive policy only; independent semantics pending'),indent=2),encoding='utf-8')
                save_checkpoint(output / "last.pt", {"system": model}, config, model.construction_info,
                                "streaming_avatar_v3", step=step+1, optimizer=optimizer,
                                metrics=validation_metrics, run_state=run_state, scaler=scaler)
                if reconstruction_improved:
                    shutil.copyfile(output/'last.pt',output/'best_reconstruction.pt')
                if improved:
                    shutil.copyfile(output / "last.pt", output / "best.pt")
                    (output / "validation.json").write_text(json.dumps({"step": step+1, "metrics": validation_metrics,
                                                                              "dialogues": records}, indent=2), encoding="utf-8")
                print(json.dumps({"validation_step": step+1, **validation_metrics}), flush=True)
            context.barrier()
    if context.is_main:
        (output / "training_status.json").write_text(json.dumps({"status": "complete" if final_step == train_config["max_steps"] else "paused",
            "step": final_step,
            "variant": model.variant, "best_generation_total": best, "metrics": validation_metrics}), encoding="utf-8")
    if hasattr(cursor, "close"):
        cursor.close()
    return {"steps": final_step, "best_generation_total": best, "output": str(output)}


if __name__ == "__main__":
    import argparse
    from emotion_ssm.config_v3 import read_config
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--stop-after", type=int, help="Save and pause at this absolute step without changing the training budget")
    arguments = parser.parse_args()
    run(read_config(arguments.config), stop_after=arguments.stop_after)
