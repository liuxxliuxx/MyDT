"""V3 second-clock dynamics training with complete-dialogue TBPTT.

Each optimizer step consumes exactly the declared number of NEW packets across
ranks. Future packets are visible only to a frozen target teacher. The learned
state forecast receives its origin state and query seconds, never future input.
"""
from __future__ import annotations

import argparse
import bisect
import copy
import json
import math
import os
import random
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn import functional as F

from emotion_ssm.config_v3 import (DYNAMICS_REVISION, FUTURE_LABEL_PROTOCOL,
                                   LEGACY_FUTURE_LABEL_PROTOCOL, UNIT_LABEL_DYNAMICS_REVISIONS,
                                   read_config, validate_config, write_config)
from emotion_ssm.data.packets_v3 import TokenPacketDataset, collate_role_features, prefix_role_features
from emotion_ssm.models.state_core import EmotionMemory, StateObservation, UnifiedEmotionStateCore
from emotion_ssm.models.token_observer import SUBSETS, TokenObserver
from emotion_ssm.utils.checkpoint import capture_rng_state, restore_rng_state
from emotion_ssm.utils.checkpoint_v3 import (read_checkpoint, restore_training, save_checkpoint,
                                           require_training_revision)


def _cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: _cpu(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_cpu(v) for v in value)
    return value


def split_observation(observation, row: int) -> StateObservation:
    """Preserve v3 dynamic fields that the legacy dataclass index method omits."""
    names = ("aff", "event", "action", "reliability", "modality_mask", "event_present",
             "action_duration", "fresh_observation", "action_present", "event_id")
    return StateObservation(**{name: (None if getattr(observation, name, None) is None else
                                     getattr(observation, name)[row:row + 1]) for name in names})


def encode_pair(observer, packet, device, subsets=("AVT", "AVT"), dropout=0., event_id=None):
    pair, encoded = [], []
    if subsets[0] == subsets[1]:
        features = collate_role_features(packet["roles"], device)
        result = observer.encode(features, subset=subsets[0])
        encoded = [{"raw_aff": result["raw_aff"][i:i+1], "valid": result["valid"][i:i+1]}
                   for i in (0, 1)]
        pair = [split_observation(result["observation"], i) for i in (0, 1)]
    else:
        for role in (0, 1):
            result = observer.encode(collate_role_features([packet["roles"][role]], device), subset=subsets[role])
            pair.append(split_observation(result["observation"], 0))
            encoded.append(result)
    for observation in pair:
        if event_id is not None:
            observation.event_id = torch.full((1,), event_id, device=device, dtype=torch.long)
        if dropout and bool(torch.rand(()) < dropout):
            observation.modality_mask = torch.zeros_like(observation.modality_mask)
            observation.fresh_observation = torch.zeros_like(observation.fresh_observation)
            observation.event_present = torch.zeros_like(observation.event_present)
            observation.action_present = torch.zeros_like(observation.action_present)
            observation.action_duration = torch.zeros_like(observation.action_duration)
    return pair, encoded


class SufficientStatistics:
    """Unrounded sums/counts; independent of batch size, sharding and padding."""
    def __init__(self):
        self.values = {}

    def add(self, name, total, count):
        total, count = float(total), float(count)
        old = self.values.setdefault(name, [0., 0.])
        old[0] += total
        old[1] += count

    def mse(self, name, prediction, target):
        self.add(name, (prediction.detach().float() - target.detach().float()).square().sum(), target.numel())

    def merge(self, other):
        for name, (total, count) in other.values.items():
            self.add(name, total, count)

    def distributed(self):
        if dist.is_initialized():
            gathered = [None] * dist.get_world_size()
            dist.all_gather_object(gathered, self.values)
            self.values = {}
            for value in gathered:
                for name, (total, count) in value.items():
                    self.add(name, total, count)
        return self

    def metrics(self):
        result = {}
        for name, (total, count) in self.values.items():
            result[name] = total / count if count else None
            result[name + "_count"] = count
        return result


def unit_state_readout(affect, minimum_norm=1e-5):
    """Use the A0 label heads' unit coordinates without changing state dynamics.

    A near-zero state has no direction. Its readout is explicitly zero, with no
    directional label gradient; the vector forecast objective still trains it.
    """
    norm = affect.float().norm(dim=-1, keepdim=True)
    unit = affect.float() / norm.clamp_min(minimum_norm)
    return torch.where(norm > minimum_norm, unit, torch.zeros_like(unit))


def state_statistics(statistics, prefix, core, state):
    """Expose cancellation instead of judging memory from the summed state alone."""
    fast, slow = state.fast.detach().float(), state.slow.detach().float()
    for name, value in (("fast_norm", fast.norm(dim=-1)), ("slow_norm", slow.norm(dim=-1)),
                        ("affect_norm", core.affect(state).detach().float().norm(dim=-1))):
        statistics.add(prefix + "/" + name, value.sum(), value.numel())
    present = (fast.norm(dim=-1) > 1e-6) & (slow.norm(dim=-1) > 1e-6)
    cosine = F.cosine_similarity(fast, slow, dim=-1)[present]
    statistics.add(prefix + "/fast_slow_cosine", cosine.sum(), cosine.numel())
    if core.adaptive_flow is not None:
        with torch.no_grad(), torch.autocast(device_type=fast.device.type, enabled=False):
            diagnostics = {}
            core.adaptive_flow.coefficients(state.detach(), core.rates(),
                core.max_autonomous_rotation*core.rotation.tanh(), diagnostics=diagnostics)
        for name, value in diagnostics.items():
            statistics.add(prefix + "/adaptive/" + name, value.sum(), value.numel())


def label_loss(observer, affect, labels, statistics=None, prefix="label", normalize=False):
    """Each actual endpoint label is consumed once; no label interpolation."""
    total = affect.sum() * 0.
    output = observer.decode_affect(unit_state_readout(affect) if normalize else affect)
    for role in (0, 1):
        for label in labels[role]:
            emotion = int(label.get("emotion", -1))
            if 0 <= emotion < 7:
                logits = output["emotion_logits"][:, role]
                target = torch.tensor([emotion], device=affect.device)
                value = F.cross_entropy(logits.float(), target, reduction="sum")
                total = total + value
                if statistics is not None:
                    statistics.add(prefix + "/emotion_ce", value.detach(), 1)
                    statistics.add(prefix + "/emotion_accuracy", (logits.argmax(-1) == target).sum(), 1)
            if bool(label.get("intensity_mask", False)):
                pred = output["intensity"][:, role].float()
                target = pred.new_tensor([float(label["intensity"])])
                value = (pred - target).square().sum()
                total = total + value
                if statistics is not None:
                    statistics.add(prefix + "/intensity_mse", value.detach(), 1)
            mask = torch.as_tensor(label.get("vad_mask", [False]*3), device=affect.device, dtype=torch.bool)
            if mask.any():
                pred = output["vad"][:, role, mask].float()
                target = pred.new_tensor(label["vad"])[mask][None]
                value = (pred - target).square().sum()
                total = total + value / int(mask.sum())
                if statistics is not None:
                    statistics.add(prefix + "/vad_mse", value.detach(), int(mask.sum()))
    return total


def packet_prefix(packet, endpoint):
    """A causal auxiliary prefix for a label ending inside a one-second packet."""
    delta = float(endpoint) - float(packet["start"])
    if not 0 < delta <= float(packet["dt"]) + 1e-5:
        raise ValueError("Endpoint supervision lies outside its packet")
    # A0 and dynamics must share every causal modality and freshness rule.
    # In particular a future acoustic atom's prosody must be clipped together
    # with audio, and old history must not become one more speaking action.
    return {**packet, "end": float(endpoint), "dt": delta,
            "roles": [prefix_role_features(source, float(endpoint)) for source in packet["roles"]]}


def endpoint_objective(observer, core, prior, current, packet, device,
                       subsets=("AVT", "AVT"), statistics=None, prefix="current_endpoint", event_id=None,
                       normalize_labels=False):
    """Read each label at its REAL endpoint; never peek to the packet's end."""
    grouped = {}
    for role in (0, 1):
        for label in packet.get("targets", [[], []])[role]:
            endpoint = float(label["end"])
            grouped.setdefault(endpoint, [[], []])[role].append(label)
    total = current.fast.sum() * 0.
    for endpoint, labels in grouped.items():
        if abs(endpoint - float(packet["end"])) <= 1e-5:
            state = current
        else:
            partial = packet_prefix(packet, endpoint)
            observations, _ = encode_pair(observer, partial, device, subsets, event_id=event_id)
            state = core.advance(prior, observations, partial["dt"])
        total = total + label_loss(observer, core.affect(state), labels, statistics, prefix,
                                   normalize=normalize_labels)
    return total


class DialogueCollection:
    def __init__(self, roots, split):
        self.datasets = [TokenPacketDataset(root, split) for root in roots]
        self.index = [(domain, index) for domain, dataset in enumerate(self.datasets)
                      for index in range(len(dataset))]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        dataset, sample = self.index[index]
        value = self.datasets[dataset][sample]
        return value

    def identity(self, index):
        dataset, sample = self.index[index]
        return str(self.datasets[dataset].root.resolve()) + "::" + self.datasets[dataset].ids[sample]

    def balanced_indices(self, maximum=0):
        if maximum <= 0 or maximum >= len(self):
            return list(range(len(self)))
        groups = [[i for i, item in enumerate(self.index) if item[0] == domain]
                  for domain in range(len(self.datasets))]
        ordered = [group[position] for position in range(max(map(len, groups)))
                   for group in groups if position < len(group)]
        return ordered[:maximum]


class DialogueCursor:
    """Serializable per-rank dialogue ordering; packet states never cross dialogues."""
    def __init__(self, collection, rank=0, world=1, seed=6666):
        self.collection, self.rank, self.world, self.seed = collection, rank, world, seed
        self.assigned = list(range(rank, len(collection), world))
        if not self.assigned:
            raise ValueError("Each training rank requires at least one dialogue")
        self.epoch, self.order_position, self.packet_index = 0, 0, 0
        self.order = self._order()
        self.memory = None
        self._dialogue = None

    def _order(self):
        result = list(self.assigned)
        random.Random(self.seed + self.epoch * 104729 + self.rank).shuffle(result)
        return result

    @property
    def dialogue_index(self):
        return self.order[self.order_position]

    def current(self):
        if self._dialogue is None:
            self._dialogue = self.collection[self.dialogue_index]
            if not self._dialogue["packets"]:
                raise ValueError("Empty dialogue in token training manifest")
        return self._dialogue

    def increment(self):
        self.packet_index += 1
        finished = self.packet_index >= len(self.current()["packets"])
        if finished:
            self.packet_index = 0
            self.order_position += 1
            if self.order_position == len(self.order):
                self.epoch += 1
                self.order_position = 0
                self.order = self._order()
            self._dialogue = None
            self.memory = None
        return finished

    def state_dict(self):
        return {"epoch": self.epoch, "order_position": self.order_position,
                "packet_index": self.packet_index, "order": self.order,
                "dialogue_identity": self.collection.identity(self.dialogue_index),
                "memory": None if self.memory is None else _cpu(self.memory.detach().serialize())}

    def load_state_dict(self, state, device):
        self.epoch, self.order_position, self.packet_index = (state[name] for name in
                                                           ("epoch", "order_position", "packet_index"))
        self.order = list(state["order"])
        if sorted(self.order) != sorted(self.assigned):
            raise ValueError("Resume rank/data assignment changed")
        if state["dialogue_identity"] != self.collection.identity(self.dialogue_index):
            raise ValueError("Resume dialogue identity changed")
        self.memory = (None if state["memory"] is None else
                       EmotionMemory.deserialize(state["memory"]).to(device))
        self._dialogue = None


class TeacherTargets:
    """Small CPU LRU of frozen targets; never passed as forecast inputs."""
    def __init__(self, teacher, device, batch_size=16, cache_size=24):
        self.teacher, self.device = teacher, device
        self.batch_size, self.cache_size = batch_size, cache_size
        self.cache = OrderedDict()

    @torch.no_grad()
    def get(self, key, dialogue, horizons=None):
        if key in self.cache:
            self.cache.move_to_end(key)
            result = self.cache[key]
            self._endpoint_plan(result, dialogue, horizons)
            return result
        features = [role for packet in dialogue["packets"] for role in packet["roles"]]
        affects, valid = [], []
        for begin in range(0, len(features), self.batch_size):
            batch = collate_role_features(features[begin:begin+self.batch_size], self.device)
            output = self.teacher.encode(batch, subset="AVT")
            fresh = batch.get("fresh_observation", output["observation"].modality_mask).bool()
            affects.append(output["observation"].aff.detach().cpu())
            valid.append((output["valid"] & fresh.any(-1)).cpu())
        result = {"affect": torch.cat(affects).reshape(-1, 2, self.teacher.config.affect_dim),
                  "valid": torch.cat(valid).reshape(-1, 2),
                  "times": [float(packet["end"]) for packet in dialogue["packets"]]}
        self._endpoint_plan(result, dialogue, horizons)
        self.cache[key] = result
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return result

    @staticmethod
    def _endpoint_plan(result, dialogue, horizons):
        if horizons is None:
            return
        horizons = tuple(horizons)
        if result.get("future_endpoint_horizons") == horizons:
            return
        queries = future_endpoint_queries(result["times"], dialogue["packets"], horizons)
        result["future_endpoint_horizons"] = horizons
        result["future_endpoint_queries"] = queries
        # This metadata shares the bounded teacher-target LRU lifetime.
        labels = [label for packet in dialogue["packets"] for group in packet.get("targets", [[], []])
                  for label in group if _has_gold_endpoint(label)]
        covered = {query["endpoint_key"]: query["label"] for group in queries.values() for query in group}
        count = lambda values: {"endpoints": len(values),
            "emotion_endpoints": sum(0 <= int(label.get("emotion", -1)) < 7 for label in values),
            "vad_coordinates": sum(sum(label.get("vad_mask", [False]*3)) for label in values),
            "intensity_endpoints": sum(bool(label.get("intensity_mask", False)) for label in values)}
        result["future_endpoint_coverage"] = {"available": count(labels), "covered": count(list(covered.values()))}


def future_matches(times, index, horizons):
    """Exact timestamp lookup, never floor seconds or assume one index per second."""
    matches = []
    for seconds in horizons:
        query = times[index] + float(seconds)
        candidate = bisect.bisect_left(times, query - 1e-5, lo=index+1)
        if candidate < len(times) and abs(times[candidate] - query) <= 1e-5:
            matches.append((float(seconds), candidate))
    return matches


def _has_gold_endpoint(label):
    return (0 <= int(label.get("emotion", -1)) < 7 or any(label.get("vad_mask", [False]*3))
            or bool(label.get("intensity_mask", False)))


def future_endpoint_queries(times, packets, horizons):
    """Map complete origin packet indices to sorted true-endpoint gold queries.

    Each record is one eligible label/nominal-horizon contribution, with
    ``seconds`` (also ``query_seconds``), ``nominal_horizon``, ``origin_time``,
    ``endpoint``, ``role`` and ``label``. The latest complete origin <=e-h is
    selected strictly, without rounding e to a packet boundary. Records contain
    no observation features. ``endpoint_weight`` averages repeated horizons for
    independent-endpoint reporting, not for the per-origin training reduction.
    Endpoint identity is (role, true end) within one dialogue. Duplicate valid
    endpoints are rejected, including identical annotations, rather than counted
    as independent supervision. Only target fields are copied into each record.
    """
    times, horizons = [float(value) for value in times], [float(value) for value in horizons]
    if len(times) != len(packets) or any(not math.isfinite(value) for value in times):
        raise ValueError("Endpoint planning requires one finite time per packet")
    if any(right <= left for left, right in zip(times, times[1:])):
        raise ValueError("Endpoint planning requires strictly increasing packet times")
    if not horizons or horizons != sorted(set(horizons)) or any(not math.isfinite(h) or h <= 0 for h in horizons):
        raise ValueError("Endpoint planning requires sorted unique positive horizons")
    result, endpoints = {}, set()
    target_fields = ("start", "end", "emotion", "intensity", "intensity_mask", "intensity_source",
                     "vad", "vad_mask", "utterance_id", "raw_emotion")
    for packet in packets:
        for role, labels in enumerate(packet.get("targets", [[], []])):
            for label in labels:
                if not _has_gold_endpoint(label):
                    continue
                endpoint = float(label["end"])
                if not math.isfinite(endpoint):
                    raise ValueError("Gold endpoint time must be finite")
                endpoint_key = (role, endpoint)
                if endpoint_key in endpoints:
                    raise ValueError("Duplicate valid gold endpoint for the same role and true end")
                endpoints.add(endpoint_key)
                target = {name: copy.deepcopy(label[name]) for name in target_fields if name in label}
                planned = []
                for nominal in horizons:
                    origin = bisect.bisect_right(times, endpoint-nominal)-1
                    if origin >= 0:
                        seconds = endpoint-times[origin]
                        planned.append((origin, {"seconds": seconds, "query_seconds": seconds,
                            "nominal_horizon": nominal, "origin_time": times[origin], "endpoint": endpoint,
                            "role": role, "label": target, "endpoint_key": endpoint_key}))
                for origin, query in planned:
                    query["endpoint_weight"] = 1./len(planned)
                    result.setdefault(origin, []).append(query)
    for queries in result.values():
        queries.sort(key=lambda query: (query["seconds"], query["role"], query["nominal_horizon"]))
    return result


def future_endpoint_objective(core, observer, state, queries, statistics=None, normalize_labels=True):
    """Mean gold loss per eligible label/horizon at this origin, no future inputs.

    ``queries`` is one list from future_endpoint_queries. Freshness of a future
    teacher is intentionally irrelevant to independently supplied gold labels.
    Statistics retain query-weighted sums/counts and additionally average each
    endpoint's available horizons through its planner-supplied endpoint_weight.
    """
    loss = state.fast.sum()*0.
    if queries:
        seconds = sorted({float(query["seconds"]) for query in queries})
        forecasts = dict(zip(seconds, core.forecast(state, seconds)))
        for query in queries:
            labels = [[], []]
            labels[query["role"]] = [query["label"]]
            local = SufficientStatistics() if statistics is not None else None
            loss = loss + label_loss(observer, core.affect(forecasts[query["seconds"]]), labels,
                                     local, "label", normalize=normalize_labels)
            if statistics is not None:
                for name, (total, count) in local.values.items():
                    name = name.removeprefix("label/")
                    statistics.add("future_endpoint/"+name, total, count)
                    statistics.add(f"future_endpoint/h{query['nominal_horizon']:g}/"+name, total, count)
                    weight = query["endpoint_weight"]
                    statistics.add("future_endpoint_unique/"+name, total*weight, count*weight)
                statistics.add("future_endpoint/actual_seconds", query["seconds"], 1)
        loss = loss/len(queries)
    if statistics is not None:
        statistics.add("future_endpoint/loss", loss.detach(), 1)
        statistics.add("future_endpoint/queries_per_origin", len(queries), 1)
        statistics.add("future_endpoint/origin_coverage", bool(queries), 1)
    return loss


def forecast_objective(core, observer, state, targets, dialogue, index, horizons,
                       statistics=None, training_mean=None, labels_weight=1.0,
                       vector_loss=False, normalize_labels=False, prediction_statistics=None):
    matches = [(h, j) for h, j in future_matches(targets["times"], index, horizons)
               if bool(targets["valid"][j].any())]
    loss = state.fast.sum() * 0.
    if not matches:
        return loss
    predicted = core.forecast(state, [h for h, _ in matches])
    for prediction, (seconds, future) in zip(predicted, matches):
        target = targets["affect"][future:future+1].to(state.fast.device)
        valid = targets["valid"][future:future+1].to(state.fast.device)
        affect = core.affect(prediction)
        coordinate_error = F.mse_loss(affect[valid].float(), target[valid].float())
        if prediction_statistics is not None:
            prediction_statistics.mse("future_affect_mse", affect[valid], target[valid])
        # A per-coordinate MSE shrinks the forecast objective by affect_dim
        # relative to endpoint CE. Preserve raw MSE reporting below, but optimize
        # mean squared L2 distance per valid affect vector in the new revision.
        loss = loss + coordinate_error * (affect.shape[-1] if vector_loss else 1)
        # These are the legacy exact-grid labels. Revised training sets their
        # loss weight to zero and uses future_endpoint_objective at true ends;
        # the old grid labels remain available as diagnostic metrics below.
        labels = [[label for label in group if abs(float(label["end"]) - targets["times"][future]) <= 1e-5]
                  for group in dialogue["packets"][future].get("targets", [[], []])]
        if labels_weight:
            loss = loss + labels_weight * label_loss(observer, affect, labels, normalize=normalize_labels)
        if statistics is not None:
            state_statistics(statistics, f"learned_open_loop/h{seconds:g}/state", core, prediction)
            estimates = {"learned_open_loop": affect, "last_state": core.affect(state),
                         "pure_decay": core.affect(core.decay_only(state, seconds))}
            if training_mean is not None:
                estimates["training_mean"] = training_mean[None, None].expand_as(affect)
            for name, estimate in estimates.items():
                statistics.mse(name + "/affect_mse", estimate[valid], target[valid])
                statistics.mse(name + f"/h{seconds:g}/affect_mse", estimate[valid], target[valid])
                label_loss(observer, estimate, labels, statistics, name + "/endpoint",
                           normalize=normalize_labels)
    return loss / len(matches)


def configure_parameters(observer, core, config):
    # Event/action and dynamics learn in the fixed calibrated affect coordinate.
    # Freezing only the final heads would still allow fusion/adapter drift.
    observer.requires_grad_(False)
    observer.event_head.requires_grad_(True)
    observer.action_head.requires_grad_(True)
    event = [p for p in observer.parameters() if p.requires_grad]
    return [{"params": list(core.parameters()), "lr": config["train"]["state_lr"]},
            {"params": event, "lr": config["train"]["lr"]}]


def synchronize_gradients(parameters, new_chunks, device):
    count = torch.tensor(float(new_chunks), device=device)
    if dist.is_initialized():
        dist.all_reduce(count, op=dist.ReduceOp.SUM)
        for parameter in parameters:
            if parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter)
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
    for parameter in parameters:
        if parameter.grad is not None:
            parameter.grad.div_(count.clamp_min(1))
            if not torch.isfinite(parameter.grad).all():
                raise FloatingPointError("Non-finite global dynamics gradient; no optimizer update applied")
    return int(count.item())


@torch.no_grad()
def validate(observer, teacher, core, collection, target_cache, config, device,
             training_mean, rank=0, world=1):
    was_training = observer.training
    observer.eval()
    core.eval()
    statistics = SufficientStatistics()
    maximum = int(config["train"].get("validation_max_dialogues", 0))
    revised = config["train"].get("dynamics_revision") in UNIT_LABEL_DYNAMICS_REVISIONS
    future_protocol = config["train"].get("future_label_protocol", LEGACY_FUTURE_LABEL_PROTOCOL)
    if future_protocol not in (FUTURE_LABEL_PROTOCOL, LEGACY_FUTURE_LABEL_PROTOCOL):
        raise ValueError("Unknown future label protocol")
    real_endpoints = future_protocol == FUTURE_LABEL_PROTOCOL
    horizons = config["train"]["forecast_seconds"]
    indices = collection.balanced_indices(maximum)
    for index in indices[rank::world]:
        dialogue = collection[index]
        targets = target_cache.get("val:" + collection.identity(index), dialogue,
                                   horizons=horizons if real_endpoints else None)
        if real_endpoints:
            coverage = targets["future_endpoint_coverage"]
            for name, available in coverage["available"].items():
                statistics.add("future_endpoint_coverage/"+name, coverage["covered"][name], available)
        state = core.initialize(1, device)
        for tick, packet in enumerate(dialogue["packets"]):
            pair, _ = encode_pair(observer, packet, device, event_id=tick)
            prior = state
            state = core.advance(prior, pair, float(packet["dt"]))
            state_statistics(statistics, "current_state", core, state)
            fresh = targets["valid"][tick:tick+1].to(device)
            observed = torch.stack([item.aff for item in pair], 1)
            statistics.mse("observer_teacher/affect_mse", observed[fresh],
                           targets["affect"][tick:tick+1].to(device)[fresh])
            endpoint_objective(observer, core, prior, state, packet, device,
                               statistics=statistics, event_id=tick, normalize_labels=revised)
            forecast_objective(core, observer, state, targets, dialogue, tick,
                               horizons, statistics, training_mean,
                                labels_weight=0. if real_endpoints else config["train"]["label_weight"],
                                vector_loss=revised, normalize_labels=revised)
            if real_endpoints:
                future_endpoint_objective(core, observer, state, targets["future_endpoint_queries"].get(tick, []),
                                          statistics, normalize_labels=True)
        statistics.add("dialogues", 1, 1)
        statistics.add("new_packets", len(dialogue["packets"]), 1)
    result = statistics.distributed().metrics()
    result["validation_loss"] = result.get("learned_open_loop/affect_mse")
    if result["validation_loss"] is None:
        raise ValueError("Validation has no exact-time fresh future targets; cannot choose best")
    result["protocol"] = "full_dialogue_exact_seconds_no_future_inputs"
    result["grid_label_metrics_protocol"] = LEGACY_FUTURE_LABEL_PROTOCOL
    result["future_label_protocol"] = future_protocol
    result["future_endpoint_metrics_protocol"] = FUTURE_LABEL_PROTOCOL if real_endpoints else None
    result["validation_loss_protocol"] = "exact_grid_fresh_teacher_raw_affect_mse"
    result["validation_dialogue_limit"] = maximum
    result["label_readout"] = "unit_affect_zero_for_undefined_direction" if revised else "legacy_raw_state"
    result["dynamics_revision"] = config["train"].get("dynamics_revision", "legacy")
    observer.train(was_training)
    core.train()
    return result


def _distributed_device(config):
    world = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    requested = config["train"].get("device", "cuda:0")
    device = torch.device(f"cuda:{local_rank}" if world > 1 and requested.startswith("cuda") else requested)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if world > 1 and not dist.is_initialized():
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    return device, rank, world


def _resume_configuration(payload, until_step=None, output=None, learning_rate=None):
    """Resume exactly, or explicitly fork the learning rate into another output."""
    config = copy.deepcopy(payload["config"])
    if until_step is not None:
        if isinstance(until_step, bool) or not isinstance(until_step, int):
            raise ValueError("resume_until_step must be an integer absolute step")
        previous_limit = int(config["train"]["max_steps"])
        if until_step <= max(previous_limit, int(payload["global_step"])):
            raise ValueError("resume_until_step must extend the checkpoint's existing step limit")
        config["train"]["max_steps"] = until_step
        config["train"]["dynamics_steps"] = until_step
    if output is not None:
        if not str(output).strip():
            raise ValueError("resume_output cannot be empty")
        config["paths"]["output"] = str(Path(output).resolve())
    if learning_rate is not None:
        if (isinstance(learning_rate, bool) or not isinstance(learning_rate, (int, float))
                or not math.isfinite(learning_rate) or learning_rate <= 0):
            raise ValueError("resume_lr must be a finite positive number")
        if output is None or Path(output).resolve() == Path(payload["config"]["paths"]["output"]).resolve():
            raise ValueError("A learning-rate branch requires a separate resume_output directory")
        if payload.get("optimizer") is None:
            raise ValueError("A learning-rate branch requires the source optimizer state")
        config["train"].update(lr=float(learning_rate), state_lr=float(learning_rate))
        config["learning_rate_branch"] = {
            "source_step": int(payload["global_step"]),
            "source_output": payload["config"]["paths"]["output"],
            "source_learning_rates": [float(group["lr"]) for group in payload["optimizer"]["param_groups"]],
            "learning_rate": float(learning_rate),
            "optimizer_moments_preserved": True,
        }
    return config


def run(config, *, resume_until_step=None, resume_output=None, resume_lr=None):
    config = copy.deepcopy(validate_config(config))
    resume_path = config["paths"].get("resume")
    resumed = read_checkpoint(resume_path) if resume_path else None
    if resumed is None and any(value is not None for value in (resume_until_step, resume_output, resume_lr)):
        raise ValueError("Resume overrides require a complete resume checkpoint")
    if resumed is not None:
        if resumed["kind"] != "dynamics_v3":
            raise ValueError("Only a complete dynamics_v3 checkpoint can resume this stage")
        require_training_revision(resumed)
        # Batch budget/data are authoritative; an explicit LR fork is recorded.
        config = _resume_configuration(resumed, resume_until_step, resume_output, resume_lr)
    else:
        # Generation's diagnostic max_steps is distinct from the upstream
        # dynamics budget. Record the effective budget in saved experiment data.
        config["train"]["max_steps"] = int(config["train"].get("dynamics_steps", config["train"]["max_steps"]))
    if config["train"].get("dynamics_revision") != DYNAMICS_REVISION:
        raise ValueError("New dynamics training requires the current dynamics_revision and a fresh optimizer")
    if config["train"].get("future_label_protocol") != FUTURE_LABEL_PROTOCOL:
        raise ValueError("New dynamics training requires the current future_label_protocol")
    if min(int(config["train"]["max_steps"]), int(config["train"]["validate_every"]),
           int(config["train"]["log_every"])) < 1:
        raise ValueError("Dynamics steps, validation and logging intervals must be positive")
    device, rank, world = _distributed_device(config)
    seed = int(config["train"]["seed"]) + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(bool(config["train"].get("deterministic", True)), warn_only=True)
    if resumed is None:
        source = read_checkpoint(config["paths"]["observation_checkpoint"])
        if source["construction"].get("adapter_source_binding") is not None:
            config["data"]["adapter_source_binding"] = copy.deepcopy(source["construction"]["adapter_source_binding"])
        observer = TokenObserver(source["construction"]["observer"])
        observer.load_state_dict(source["models"]["observer"], strict=True)
        teacher = TokenObserver(source["construction"]["observer"])
        teacher.load_state_dict(source["models"].get("teacher", source["models"]["observer"]), strict=True)
        core = UnifiedEmotionStateCore(**{"observation_dim": observer.config.affect_dim, **config["state"]})
    else:
        observer = TokenObserver(resumed["construction"]["observer"])
        teacher = TokenObserver(resumed["construction"]["observer"])
        core = UnifiedEmotionStateCore.from_config(resumed["construction"]["state"])
    observer.to(device)
    teacher.to(device).requires_grad_(False).eval()
    core.to(device)
    if config["train"].get("compile_adaptive_flow", False):
        if core.adaptive_flow is None or device.type != "cuda":
            raise ValueError("Compiled flow execution requires the adaptive CUDA core")
        # Compile a bound function, not the module: weight names/construction
        # remain unchanged in complete checkpoints. Keep default FP32 accuracy.
        core.adaptive_flow.coefficients = torch.compile(core.adaptive_flow.coefficients, fullgraph=True)
    groups = configure_parameters(observer, core, config)
    optimizer = torch.optim.AdamW(groups, weight_decay=config["train"]["weight_decay"])
    parameters = [p for group in groups for p in group["params"]]
    models = {"observer": observer, "teacher": teacher, "state": core}
    roots = config["data"]["token_roots"]
    if not roots:
        raise ValueError("Dynamics requires token_roots containing train/val dialogue manifests")
    training = DialogueCollection(roots, "train")
    validation = DialogueCollection(roots, "val")
    cursor = DialogueCursor(training, rank, world, config["train"]["seed"])
    target_cache = TeacherTargets(teacher, device, config["train"]["observer_batch_size"])
    mean_sum = torch.zeros(observer.config.affect_dim, device=device, dtype=torch.float64)
    mean_count = torch.zeros((), device=device, dtype=torch.float64)
    start_step, best = 0, math.inf
    if resumed is not None:
        restore_training(resumed, models, optimizer=optimizer, config=config, restore_rng=False)
        # load_state_dict restores source group options, including its old LR.
        # Override after loading, preserving every moment tensor and step counter.
        if resume_lr is not None:
            for group in optimizer.param_groups:
                group["lr"] = float(resume_lr)
        run_state = resumed["run_state"]
        if int(run_state["world_size"]) != world:
            raise ValueError("Exact resume requires the same DDP world size")
        rank_state = run_state["ranks"][rank]
        cursor.load_state_dict(rank_state["cursor"], device)
        mean_sum = run_state["training_mean_sum"].to(device)
        mean_count = run_state["training_mean_count"].to(device)
        best = float(run_state["best_validation"])
        start_step = int(resumed["global_step"])
        restore_rng_state(rank_state["rng"])
    if dist.is_initialized():
        for model in models.values():
            for value in model.state_dict().values():
                dist.broadcast(value, 0)
    budget = int(config["train"]["global_chunks_per_step"])
    if budget < world:
        raise ValueError("Global new-packet budget must be at least world size")
    quota = budget // world + int(rank < budget % world)
    output = Path(config["paths"]["output"])
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        write_config(output / "config.json", config)
        if resumed is not None:
            optimizer_steps = [int(value["step"].item()) for value in optimizer.state.values() if "step" in value]
            receipt = {"checkpoint": str(resume_path), "checkpoint_step": start_step,
                       "source_learning_rates": [float(g["lr"]) for g in resumed["optimizer"]["param_groups"]],
                       "actual_learning_rates": [float(g["lr"]) for g in optimizer.param_groups],
                       "explicit_lr_override": resume_lr, "world_size": world,
                       "optimizer_step_range": [min(optimizer_steps), max(optimizer_steps)] if optimizer_steps else [],
                       "cursor": {k: rank_state["cursor"][k] for k in ("epoch", "order_position", "packet_index", "dialogue_identity")}}
            (output / "resume_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        (output / "training_status.json").write_text(json.dumps(
            {"status": "running", "stage": "dynamics_v3", "step": start_step,
             "max_steps": int(config["train"]["max_steps"]), "world_size": world}), encoding="utf-8")
    use_amp = bool(config["train"].get("amp", True)) and device.type == "cuda" and torch.cuda.is_bf16_supported()
    maximum = int(config["train"]["max_steps"])
    construction = {"observer": observer.construction(), "state": core.get_config()}
    if config["data"].get("adapter_source_binding") is not None:
        construction["adapter_source_binding"] = copy.deepcopy(config["data"]["adapter_source_binding"])
    validation_metrics = {}
    # Frozen observation features must not acquire training-only dropout noise.
    # eval() does not disable gradients through the trainable event/action heads.
    observer.eval()
    core.train()
    for step in range(start_step + 1, maximum + 1):
        optimizer.zero_grad(set_to_none=True)
        statistics = SufficientStatistics()
        graph_losses, segment_seconds = [], 0.
        increment_sum = torch.zeros_like(mean_sum)
        increment_count = torch.zeros_like(mean_count)
        for consumed in range(quota):
            dialogue = cursor.current()
            tick = cursor.packet_index
            packet = dialogue["packets"][tick]
            targets = target_cache.get("train:" + training.identity(cursor.dialogue_index), dialogue,
                                       horizons=config["train"]["forecast_seconds"])
            if cursor.memory is None:
                cursor.memory = core.initialize(1, device)
            if not graph_losses:
                cursor.memory = core.rebind_trainable_baseline(cursor.memory)
            subsets = ("AVT", "AVT")
            if config["train"].get("modality_dropout", True):
                from emotion_ssm.models.context_masking import sample_training_subset
                subsets = tuple(sample_training_subset(config["train"].get("masking")) for _ in (0, 1))
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                pair, _ = encode_pair(observer, packet, device, subsets,
                                      float(config["train"].get("observation_dropout", 0.)), tick)
                current = core.advance(cursor.memory, pair, float(packet["dt"]))
                target = targets["affect"][tick:tick+1].to(device)
                valid = targets["valid"][tick:tick+1].to(device)
                observed = torch.stack([item.aff for item in pair], 1)
                retained = torch.stack([item.modality_mask.any(-1) for item in pair], 1) & valid
                anchor = (F.mse_loss(observed[retained].float(), target[retained].float()) if retained.any()
                          else observed.sum() * 0.)
                future = forecast_objective(core, observer, current, targets, dialogue, tick,
                                            config["train"]["forecast_seconds"], labels_weight=0.,
                                            vector_loss=True, normalize_labels=True, prediction_statistics=statistics)
                future_labels = future_endpoint_objective(
                    core, observer, current, targets["future_endpoint_queries"].get(tick, []), statistics,
                    normalize_labels=True)
                labels = endpoint_objective(observer, core, cursor.memory, current, packet, device,
                                            subsets=subsets, event_id=tick, normalize_labels=True)
                loss = (config["train"]["future_weight"] * (future + config["train"]["label_weight"] * future_labels) +
                        config["train"]["coordinate_weight"] * anchor + config["train"]["label_weight"] * labels)
            if not torch.isfinite(loss.detach()).all():
                if rank == 0:
                    (output / "training_status.json").write_text(json.dumps(
                        {"status": "failed", "stage": "dynamics_v3", "step": step,
                         "reason": "non_finite_loss_no_optimizer_update"}), encoding="utf-8")
                raise FloatingPointError("Non-finite dynamics loss; no optimizer update applied")
            graph_losses.append(loss)
            segment_seconds += float(packet["dt"])
            cursor.memory = current
            statistics.add("loss", loss.detach(), 1)
            statistics.add("future", future.detach(), 1)
            statistics.add("coordinate", anchor.detach(), 1)
            statistics.add("endpoint_label", labels.detach(), 1)
            state_statistics(statistics, "current_state", core, current)
            increment_sum += target[valid].double().sum(0)
            increment_count += valid.sum()
            finished = cursor.increment()
            boundary = (finished or consumed + 1 == quota or
                        segment_seconds >= float(config["train"]["tbptt_seconds"]))
            if boundary:
                torch.stack(graph_losses).sum().backward()
                graph_losses, segment_seconds = [], 0.
                if cursor.memory is not None:
                    cursor.memory = cursor.memory.detach()
        try:
            actual_chunks = synchronize_gradients(parameters, quota, device)
        except FloatingPointError:
            if rank == 0:
                (output / "training_status.json").write_text(json.dumps(
                    {"status": "failed", "stage": "dynamics_v3", "step": step,
                     "reason": "non_finite_global_gradient_no_optimizer_update"}), encoding="utf-8")
            raise
        if actual_chunks != budget:
            raise RuntimeError("Optimizer step did not consume its exact new-packet budget")
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, config["train"]["clip_grad"])
        optimizer.step()
        if dist.is_initialized():
            dist.all_reduce(increment_sum)
            dist.all_reduce(increment_count)
        mean_sum += increment_sum
        mean_count += increment_count
        metrics = statistics.distributed().metrics()
        metrics.update(step=step, new_chunks=actual_chunks, cumulative_new_chunks=step*budget,
                       learning_rates=[float(group["lr"]) for group in optimizer.param_groups],
                       gradient_norm=float(grad_norm), training_mean_samples=float(mean_count),
                       baseline_gradient_norm=float(core.baseline.grad.norm()) if core.baseline.grad is not None else 0.,
                        dynamics_revision=DYNAMICS_REVISION,
                        flow_execution="compiled" if config["train"].get("compile_adaptive_flow", False) else "eager",
                        future_loss_units="mean_squared_l2_per_affect",
                        future_label_protocol=FUTURE_LABEL_PROTOCOL,
                        future_endpoint_loss_units="mean_per_eligible_label_horizon_at_origin",
                       observer_shared_frozen=True, local_gradient_packet_limit=quota)
        should_validate = (step % int(config["train"]["validate_every"]) == 0 or step == maximum)
        improved = False
        if should_validate:
            if not float(mean_count):
                raise ValueError("No fresh training targets: training mean and validation are undefined")
            validation_metrics = validate(observer, teacher, core, validation, target_cache, config, device,
                                          (mean_sum / mean_count).float(), rank, world)
            improved = validation_metrics["validation_loss"] < best
            best = min(best, validation_metrics["validation_loss"])
            metrics["validation"] = validation_metrics
        if rank == 0 and (step % int(config["train"]["log_every"]) == 0 or should_validate or step == 1):
            with (output / "metrics.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(metrics, ensure_ascii=False) + "\n")
            print(json.dumps(metrics, ensure_ascii=False), flush=True)
            (output / "training_status.json").write_text(json.dumps(
                {"status": "running", "stage": "dynamics_v3", "step": step,
                 "max_steps": maximum, "world_size": world,
                 "best_validation": best if math.isfinite(best) else None}), encoding="utf-8")
        if rank == 0 and should_validate:
            (output / "validation.json").write_text(json.dumps(
                {"step": step, **validation_metrics}, ensure_ascii=False, indent=2), encoding="utf-8")
        # Persist at configured optimizer boundaries: saved cursor/state/RNG
        # never describes a partially consumed optimizer step.
        save_every = int(config["train"].get("checkpoint_every", config["train"]["validate_every"]))
        if should_validate or step % save_every == 0:
            local = {"cursor": cursor.state_dict(), "rng": capture_rng_state()}
            ranks = [local]
            if dist.is_initialized():
                ranks = [None] * world
                dist.all_gather_object(ranks, local)
            if rank == 0:
                run_state = {"world_size": world, "ranks": ranks, "best_validation": best,
                             "training_mean_sum": mean_sum.cpu(), "training_mean_count": mean_count.cpu(),
                             "mean_protocol": "all_fresh_training_packets_consumed_to_step"}
                save_checkpoint(output / "last.pt", models, config, construction, "dynamics_v3", step,
                                optimizer=optimizer, metrics=metrics, run_state=run_state)
                if improved:
                    save_checkpoint(output / "best.pt", models, config, construction, "dynamics_v3", step,
                                    optimizer=optimizer, metrics=metrics, run_state=run_state)
    if dist.is_initialized():
        dist.barrier()
    if rank == 0:
        (output / "training_status.json").write_text(json.dumps(
            {"status": "complete", "stage": "dynamics_v3", "step": maximum,
             "max_steps": maximum, "world_size": world, "best_validation": best}), encoding="utf-8")
    return {"step": maximum, "best_validation": best, "output": str(output),
            "validation": validation_metrics, "rank": rank}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default="")
    parser.add_argument("--resume-until-step", type=int, default=None,
                        help="Extend a resumed run to this absolute optimizer step; keep all learning settings")
    parser.add_argument("--resume-output", default=None,
                        help="Write a resumed run to this directory, preserving the source checkpoint")
    parser.add_argument("--resume-lr", type=float, default=None,
                        help="Fork both optimizer groups to this LR after restoring moments; requires separate output")
    arguments = parser.parse_args()
    config = read_config(arguments.config)
    if arguments.resume:
        config["paths"]["resume"] = arguments.resume
    run(config, resume_until_step=arguments.resume_until_step, resume_output=arguments.resume_output,
        resume_lr=arguments.resume_lr)


if __name__ == "__main__":
    main()
