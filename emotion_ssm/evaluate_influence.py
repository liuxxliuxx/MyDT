"""Paired history, event-response and partner-state probes for frozen v2 checkpoints.

Natural-target errors and intervention sensitivity are intentionally separate.
No intervention output is labelled as a correct counterfactual reaction.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from emotion_ssm.data.protocol import fingerprint, source_video_id
from emotion_ssm.data.timed_dualtalk import to_device
from emotion_ssm.schema import EventObservation
from emotion_ssm.train.generation import make_dataset
from emotion_ssm.utils.generation_checkpoint import load_generation
from emotion_ssm.utils.reconstruction import ReconstructionTotals


LEXICON = sorted({"happy", "happiness", "glad", "love", "loved", "wonderful", "excited",
    "great", "amazing", "sad", "sorry", "hate", "hated", "angry", "afraid", "scared",
    "terrible", "upset", "worried", "worry", "fear", "hurt", "cry", "crying"})
BRANCHES = ("full", "history_reset", "events_removed", "partner_shuffled", "coupling_off",
            "pulse_zero", "pulse_single", "pulse_repeated")


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def paired_name(name):
    for suffix in ("_speaker1", "_speaker2"):
        if name.endswith(suffix):
            return name[:-len(suffix)]
    raise ValueError("Expected an explicitly paired speaker name")


def prepare(feature_root, output, pairs=32, seed=6666, split="test"):
    """Freeze selection without loading any generator prediction or FLAME target."""
    import re
    root = Path(feature_root)
    dataset_manifest = json.loads((root / "dataset_manifest.json").read_text(encoding="utf-8"))
    candidates, inventory = {}, {"paired_clips": 0, "history_eligible_pairs": 0,
        "pairs_with_any_aligned_words": 0, "lexicon_eligible_pairs": 0}
    for path in sorted((root / split).glob("*_speaker1.pt")):
        cache = torch.load(path, map_location="cpu", weights_only=False)
        if cache["cache_id"] != dataset_manifest["cache_ids"][f"{split}/{path.stem}"]:
            raise ValueError("Cache manifest mismatch")
        count = sum(int(x["target"]["valid_frames"]) == 25 and
                    int(x["partner"]["valid_frames"]) == 25 for x in cache["chunks"])
        name = paired_name(path.stem)
        inventory["paired_clips"] += 1
        inventory["pairs_with_any_aligned_words"] += bool(cache.get("aligned_words"))
        hits = []
        for word in cache.get("aligned_words", []):
            tokens = set(re.findall(r"[a-z]+", word["text"].lower()))
            index = math.ceil(float(word["available_at"])) - 1
            if (tokens.intersection(LEXICON) and float(word.get("confidence", 0)) >= .8
                    and index >= 3 and index + 8 < count):
                hits.append({"index": index, "word": word["text"], "role": word["role"],
                             "available_at": word["available_at"], "confidence": word.get("confidence")})
        hits.sort(key=lambda w: (w["available_at"], w["role"], w["word"]))
        inventory["history_eligible_pairs"] += count >= 20
        inventory["lexicon_eligible_pairs"] += bool(hits)
        candidates[name] = {"full_seconds": count, "event": hits[0] if hits else None}
    # Round-robin across original video sources, fixed hash ordering within each.
    sources = {}
    for name, row in candidates.items():
        if row["full_seconds"] >= 20:
            sources.setdefault(source_video_id(name), []).append(name)
    order = lambda x: fingerprint([seed, x])
    groups = [sorted(sources[s], key=order) for s in sorted(sources, key=order)]
    selected = []
    while len(selected) < pairs and any(groups):
        for group in groups:
            if group and len(selected) < pairs:
                selected.append(group.pop(0))
    if len(selected) < 2 or len({source_video_id(n) for n in selected}) < 2:
        raise ValueError("History/partner probes need at least two independent sources")
    donors = {name: next(other for other in selected
                        if source_video_id(other) != source_video_id(name)
                        and order([name, other]) == min(order([name, x]) for x in selected
                            if source_video_id(x) != source_video_id(name))) for name in selected}
    rows = []
    for name in sorted(set(selected) | {n for n, v in candidates.items() if v["event"]}):
        for role in ("speaker1", "speaker2"):
            rows.append({"name": f"{name}_{role}", "source": source_video_id(name),
                "history": name in selected,
                "donor": f"{donors[name]}_{role}" if name in selected else None,
                "event": candidates[name]["event"], "full_seconds": candidates[name]["full_seconds"]})
    result = {"version": 1, "created_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "split": split, "selection_seed": seed, "inventory": inventory, "rows": rows,
        "feature_digest": dataset_manifest["digest"], "lexicon": LEXICON,
        "protocol": {"checkpoint_steps": 1000, "chunk_frames": 25, "history_seconds": 3,
            "history_reset_before_index": 8, "history_score_indices": list(range(11, 19)),
            "event_relative_indices": list(range(9)), "repeat_relative_indices": [0, 2, 4],
            "event_min_alignment_confidence": .8, "bootstrap_seed": seed,
            "bootstrap_replicates": 2000, "uncertainty_unit": "original video source",
            "event_label": "fixed English lexicon candidate; not a human emotion label",
            "primary_quality": "one-second mean expression MSE against real target, full natural inputs",
            "history": "reset only persistent z/relation at 8s; identical AV and cached dialogue text; score after 3s flush",
            "partner": "shuffle partner SSM observations from a different source; retain original fast affect and all raw AV/text",
            "coupling_off": "same checkpoint and observations, disable dyadic influence and relation update",
            "pulse": "zero/single/three repetitions of one real text-event embedding; identical natural suffix observations; suffix is not assumed neutral",
            "no_counterfactual_ground_truth": True}}
    result["digest"] = fingerprint(result)
    dump(output, result)
    print(json.dumps({"manifest": str(output), "directed_clips": len(rows), "inventory": inventory}), flush=True)
    return result


def move_observation(value, device):
    return EventObservation(**{field.name: (getattr(value, field.name).to(device)
        if torch.is_tensor(getattr(value, field.name)) else getattr(value, field.name))
        for field in dataclasses.fields(value)})


def no_event(value):
    return dataclasses.replace(value, event_present=torch.zeros(len(value.aff), dtype=torch.bool, device=value.aff.device))


@torch.inference_mode()
def capture_trace(model, packets, device, max_seconds):
    """Capture an actual streaming forward. Reuse frozen pre-FiLM activations only."""
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError("Activation reuse requires a fully frozen evaluation model")
    captured = {}
    def interaction_hook(module, args, output):
        captured["hidden"] = output.detach().cpu()
    def observation_hook(module, args, output):
        captured.setdefault("observations", []).append(move_observation(output, "cpu"))
    handles = [model.generator.baseline.interaction_module.register_forward_hook(interaction_hook),
               model.observer.register_forward_hook(observation_hook)]
    trace, state = [], None
    try:
        for packet, truth, mask in packets:
            if float(packet["time"]) > max_seconds:
                break
            captured.clear()
            prediction, state, diagnostics = model(to_device(packet, device), state)
            if len(captured["observations"]) != 2:
                raise ValueError("Expected exactly two role observations per packet")
            trace.append({**captured, "prediction": prediction.cpu(), "truth": truth.cpu(),
                "mask": mask.cpu(), "time": float(packet["time"]), "count": prediction.shape[1],
                "context": diagnostics["context"].cpu(), "z": state.emotion.z.cpu(),
                "target_speaking": bool(packet.get("target_speech_active", False)),
                "partner_speaking": bool(packet.get("partner_speech_active", False))})
    finally:
        for handle in handles:
            handle.remove()
    return trace


def pack_context(model, first, second, state):
    slow = torch.cat([state.z[:, 0], state.z[:, 1], state.relation], -1)
    if model.variant in ("none", "affect"):
        slow = torch.zeros_like(slow)
    elif model.variant == "self":
        slow = torch.cat([state.z[:, 0], torch.zeros_like(state.z[:, 1]), torch.zeros_like(state.relation)], -1)
    return torch.cat([first.aff, second.aff, slow], -1)


@torch.inference_mode()
def replay_contexts(model, trace, branch="full", event=None, donor=None, role="speaker1"):
    device = next(model.parameters()).device
    ids = torch.full((1, 2), -1, dtype=torch.long, device=device)
    state = model.state_model.initialize(ids)
    contexts, states, now = [], [], 0.
    anchor = None if event is None else int(event["index"])
    event_role = None if event is None else int(event["role"] != role)
    pulse = None if anchor is None else move_observation(trace[anchor]["observations"][event_role], device)
    if pulse is not None and not bool(pulse.event_present):
        raise ValueError("Selected lexical event does not correspond to an available event embedding")
    for index, item in enumerate(trace):
        first, second = (move_observation(o, device) for o in item["observations"])
        observations = [first, second]
        if branch == "history_reset" and index == 8:
            state = model.state_model.initialize(ids)
        if branch == "partner_shuffled" and index >= 8:
            if donor is None or index >= len(donor):
                raise ValueError("Partner donor must cover the complete intervention window")
            observations[1] = dataclasses.replace(move_observation(donor[index]["observations"][1], device),
                                                 action_duration=second.action_duration)
        if anchor is not None and index >= anchor:
            if branch == "events_removed":
                # Remove only the anchor chunk's selected-role event; later real events remain.
                if index == anchor:
                    observations[event_role] = no_event(observations[event_role])
            elif branch.startswith("pulse_"):
                observations = [no_event(o) for o in observations]
                offsets = {"pulse_zero": [], "pulse_single": [0], "pulse_repeated": [0, 2, 4]}[branch]
                if index - anchor in offsets:
                    original = observations[event_role]
                    if not bool(original.modality_mask[:, 2].all()):
                        raise ValueError("Pulse injection requires the original available text modality")
                    observations[event_role] = dataclasses.replace(original, event=pulse.event,
                                                                   event_present=pulse.event_present)
        state = model.state_model.observe(state, observations, item["time"] - now,
            enable_partner=model.variant == "dyadic" and branch != "coupling_off")
        now = item["time"]
        # Fast affect always uses the original current observations, even in partner interventions.
        contexts.append(pack_context(model, first, second, state).cpu())
        states.append(state.z.cpu())
    return contexts, states


@torch.inference_mode()
def render_contexts(model, trace, contexts):
    """Frame-aligned 3s history, using only activations captured for this checkpoint/input."""
    if model.variant == "none":
        return [item["prediction"] for item in trace]
    device = next(model.parameters()).device
    history, result = None, []
    for item, value in zip(trace, contexts):
        current = value[:, None].expand(-1, item["count"], -1)
        joined = current if history is None else torch.cat([history, current], 1)
        if joined.shape[1] != item["hidden"].shape[1]:
            raise ValueError("Replay context and original causal activation window disagree")
        conditioned = model.generator.film(item["hidden"].to(device), joined.to(device))
        prediction = model.generator.baseline.synthesis_module(conditioned)[:, -item["count"]:]
        result.append(prediction.cpu())
        history = joined[:, -model.history_frames:] if model.history_frames else None
    return result


def mean_expression(value, mask):
    return (value[..., :50].double() * mask[..., None]).sum((0, 1)) / mask.sum().clamp_min(1)


def score(trace, predictions, indices):
    total, mean_sse, mean_count = ReconstructionTotals(), 0., 0
    for i in indices:
        item = trace[i]
        previous = None if i == 0 else (predictions[i-1][:, -1], trace[i-1]["truth"][:, -1], trace[i-1]["mask"][:, -1])
        total.update(predictions[i], item["truth"], item["mask"], previous)
        if item["mask"].any():
            error = mean_expression(predictions[i], item["mask"]) - mean_expression(item["truth"], item["mask"])
            mean_sse += float(error.square().sum())
            mean_count += len(error)
    return {"sse": total.square_error, "elements": total.elements, "mean_expression_sse": mean_sse,
            "mean_expression_elements": mean_count, **total.metrics(),
            "mean_expression_mse": mean_sse / max(mean_count, 1)}


def difference(trace, full, changed, indices):
    sse, elements = 0., 0
    curve = []
    for i in indices:
        mask = trace[i]["mask"]
        value = (full[i][..., :50].double() - changed[i][..., :50].double()).square()
        local = float(torch.where(mask[..., None], value, 0.).sum())
        count = int(mask.sum()) * 50
        curve.append(math.sqrt(local/count) if count else None)
        sse += local
        elements += count
    return {"expression_rms": math.sqrt(sse/max(elements, 1)), "per_second_rms": curve}


def event_trajectory(trace, predictions, anchor):
    """Shape response relative to each sequence's own 2s pre-event level; no emotion label."""
    indices = list(range(anchor, anchor+9))
    def curve(values):
        prior = torch.stack([mean_expression(values[i], trace[i]["mask"]) for i in (anchor-2, anchor-1)]).mean(0)
        delta = torch.stack([mean_expression(values[i], trace[i]["mask"]) for i in indices]) - prior
        return delta, delta.square().mean(-1).sqrt()
    observed, response = curve(predictions)
    target, target_response = curve([item["truth"] for item in trace])
    return {"centered_trajectory_mse": float((observed-target).square().mean()),
        "response_amplitude_mae": float((response-target_response).abs().mean()),
        "peak_lag_error_seconds": float(abs(int(response.argmax()) - int(target_response.argmax()))),
        "late_response_error": float(abs(response[5:].mean()-target_response[5:].mean())),
        "response_area_error": float(abs(response.sum()-target_response.sum())),
        "generated_response": response.tolist(), "target_response": target_response.tolist()}


def maximum_difference(a, b):
    return max(float((x-y).abs().max()) for x, y in zip(a, b))


@torch.inference_mode()
def run(checkpoint, manifest, output, device="cuda:0"):
    torch.set_num_threads(4)
    torch.manual_seed(6666)
    specification = json.loads(Path(manifest).read_text(encoding="utf-8"))
    if specification["digest"] != fingerprint({k: v for k, v in specification.items() if k != "digest"}):
        raise ValueError("Experiment manifest changed")
    started = time.time()
    model, cfg, payload = load_generation(checkpoint, device)
    model.eval().requires_grad_(False)
    if cfg.DUALTALK.FPS != 25 or cfg.DUALTALK.HISTORY_SECONDS != 3 or payload["global_step"] != specification["protocol"]["checkpoint_steps"]:
        raise ValueError("Checkpoint is outside the frozen comparison protocol")
    metadata = {"checkpoint": str(checkpoint), "global_step": payload["global_step"], "seed": cfg.SEED,
                "generation_protocol": payload["protocol"], "variant": model.variant}
    del payload
    dataset = make_dataset(cfg, specification["split"], model.construction_info["feature_source"])
    if dataset.feature_digest != specification["feature_digest"]:
        raise ValueError("Experiment features differ from the frozen selection")
    metadata.update(feature_digest=dataset.feature_digest, split_digest=dataset.manifest_digest,
                    feature_source=model.construction_info["feature_source"])
    names = {name: i for i, name in enumerate(dataset.names)}
    traces = {}
    for number, row in enumerate(specification["rows"]):
        # Donor observations must extend to the recipient's end; capture 30s at most.
        limit = max(20 if row["history"] else 0, 0 if row["event"] is None else row["event"]["index"]+9)
        traces[row["name"]] = capture_trace(model, dataset.packets(names[row["name"]]), device, limit)
        if (number+1) % 8 == 0:
            print(json.dumps({"stage": "natural_forward", "variant": model.variant,
                "clips": number+1, "total": len(specification["rows"]), "elapsed_seconds": round(time.time()-started)}), flush=True)
    records, parity = [], {"context_max_absolute_error": 0., "prediction_max_absolute_error": 0.}
    for number, row in enumerate(specification["rows"]):
        trace = traces[row["name"]]
        full = [item["prediction"] for item in trace]
        contexts, _ = replay_contexts(model, trace, event=row["event"], role=row["name"].rsplit("_", 1)[1])
        parity["context_max_absolute_error"] = max(parity["context_max_absolute_error"],
            maximum_difference(contexts, [item["context"] for item in trace]))
        # Check every clip's replay; no optimizer or backbone activation changes are allowed.
        reconstructed = render_contexts(model, trace, contexts)
        parity["prediction_max_absolute_error"] = max(parity["prediction_max_absolute_error"], maximum_difference(full, reconstructed))
        if parity["context_max_absolute_error"] > 1e-6 or parity["prediction_max_absolute_error"] > 2e-5:
            raise AssertionError(f"Replay changed normal streaming inference: {parity}")
        record = {"name": row["name"], "source": row["source"], "history": None, "event": None, "partner": None}
        if row["history"]:
            indices = specification["protocol"]["history_score_indices"]
            # History/partner probes end before 20s even when a later event needs a longer trace.
            short = trace[:20]
            alt = {}
            for branch in ("history_reset", "partner_shuffled", "coupling_off"):
                context, _ = replay_contexts(model, short, branch, donor=traces[row["donor"]])
                alt[branch] = render_contexts(model, short, context)
            natural = score(trace, full, indices)
            record["history"] = {"natural": natural, "reset": score(short, alt["history_reset"], indices),
                "effect": difference(short, full, alt["history_reset"], indices)}
            listening = [i for i in indices if not trace[i]["target_speaking"] and trace[i]["partner_speaking"]]
            speaking = [i for i in indices if trace[i]["target_speaking"]]
            record["partner"] = {"natural": natural, "donor": row["donor"],
                "shuffled": score(short, alt["partner_shuffled"], indices),
                "coupling_off": score(short, alt["coupling_off"], indices),
                "shuffle_effect": difference(short, full, alt["partner_shuffled"], indices),
                "coupling_effect": difference(short, full, alt["coupling_off"], indices),
                "listening": score(trace, full, listening), "speaking": score(trace, full, speaking)}
        if row["event"] is not None:
            anchor = row["event"]["index"]
            indices = list(range(anchor, anchor+9))
            alternatives, states = {}, {}
            role = row["name"].rsplit("_", 1)[1]
            for branch in ("events_removed", "pulse_zero", "pulse_single", "pulse_repeated"):
                context, state = replay_contexts(model, trace, branch, row["event"], role=role)
                alternatives[branch] = render_contexts(model, trace, context)
                states[branch] = state
            state_curve = lambda branch: [float((states[branch][i][:, 0]-states["pulse_zero"][i][:, 0]).square().mean().sqrt()) for i in indices]
            record["event"] = {"anchor": row["event"], "natural": score(trace, full, indices),
                "removed": score(trace, alternatives["events_removed"], indices),
                "removal_effect": difference(trace, full, alternatives["events_removed"], indices),
                "trajectory": event_trajectory(trace, full, anchor),
                "single_effect": difference(trace, alternatives["pulse_zero"], alternatives["pulse_single"], indices),
                "repeated_effect": difference(trace, alternatives["pulse_zero"], alternatives["pulse_repeated"], indices),
                "single_target_state_rms": state_curve("pulse_single"),
                "repeated_target_state_rms": state_curve("pulse_repeated")}
        records.append(record)
        if (number+1) % 8 == 0:
            print(json.dumps({"stage": "interventions", "variant": model.variant, "clips": number+1,
                "total": len(specification["rows"]), "elapsed_seconds": round(time.time()-started)}), flush=True)
    result = {"version": 1, "manifest_digest": specification["digest"], "metadata": metadata,
        "text_protocol": dataset.text_protocol, "parity": parity, "records": records,
        "elapsed_seconds": time.time()-started, "complete": True}
    dump(output, result)
    print(json.dumps({"finished": str(output), "variant": model.variant, "parity": parity,
        "elapsed_seconds": result["elapsed_seconds"]}), flush=True)
    return result


def weighted(rows, metric="mean_expression_mse"):
    if metric == "mean_expression_mse":
        total = sum(r["mean_expression_sse"] for r in rows)
        count = sum(r["mean_expression_elements"] for r in rows)
    else:
        name = metric.removesuffix("_mse")
        total = sum(r["sse"][name] for r in rows)
        count = sum(r["elements"][name] for r in rows)
    return total/count if count else None


def paired_summary(base, condition, key, metric, seed=6666, bootstrap=2000):
    names = sorted(base)
    if names != sorted(condition):
        raise ValueError("Comparison does not contain the exact same paired clips")
    groups = {}
    for name in names:
        a, b = key(base[name]), key(condition[name])
        if a["elements"] != b["elements"] or a["mean_expression_elements"] != b["mean_expression_elements"]:
            raise ValueError("Comparison has different valid target elements")
        groups.setdefault(base[name]["source"], []).append(name)
    a = weighted([key(base[n]) for n in names], metric)
    b = weighted([key(condition[n]) for n in names], metric)
    if a is None or b is None:
        return {"none": a, "condition": b, "delta": None, "relative_percent": None,
                "source_cluster_95ci": None, "source_clusters": len(groups)}
    clusters = list(groups.values())
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(bootstrap):
        sample = [n for i in rng.integers(len(clusters), size=len(clusters)) for n in clusters[i]]
        av = weighted([key(base[n]) for n in sample], metric)
        bv = weighted([key(condition[n]) for n in sample], metric)
        if av is not None and bv is not None:
            deltas.append(bv-av)
    return {"none": a, "condition": b, "delta": b-a, "relative_percent": 100*(b-a)/a if a else None,
        "source_cluster_95ci": np.quantile(deltas, [.025, .975]).tolist() if deltas else None,
        "source_clusters": len(clusters), "directed_clips": len(names)}


def summarize(inputs, output):
    values = [json.loads(Path(path).read_text(encoding="utf-8")) for path in inputs]
    by_variant = {v["metadata"]["variant"]: v for v in values}
    if len(by_variant) != len(values) or set(by_variant) != {"none", "affect", "self", "dyadic"}:
        raise ValueError("Provide all four independent controls exactly once")
    reference = by_variant["none"]
    for result in values:
        if not result["complete"] or result["manifest_digest"] != reference["manifest_digest"]:
            raise ValueError("Only identical completed experiments can be compared")
        for field in ("global_step", "seed", "generation_protocol", "feature_digest", "split_digest", "feature_source"):
            if result["metadata"][field] != reference["metadata"][field]:
                raise ValueError(f"Different comparison protocol: {field}")
    summary = {"manifest_digest": reference["manifest_digest"], "quality": {}, "mechanisms": {},
        "interpretation": "Negative error differences favor the condition. Sensitivity is not appropriateness. CIs cover video sampling, not training-seed variation.",
        "parity": {k: v["parity"] for k, v in by_variant.items()}}
    metrics = ("mean_expression_mse", "expression_mse", "jaw_mse", "neck_mse", "velocity_mse", "boundary_velocity_mse")
    for experiment in ("history", "event", "partner"):
        rows = {variant: {r["name"]: r for r in result["records"] if r[experiment] is not None}
                for variant, result in by_variant.items()}
        summary["quality"][experiment] = {variant: {metric: paired_summary(rows["none"], rows[variant],
            lambda r: r[experiment]["natural"], metric) for metric in metrics} for variant in ("affect", "self", "dyadic")}
        summary["mechanisms"][experiment] = {}
        for variant, items in rows.items():
            records = [r[experiment] for r in items.values()]
            if not records:
                summary["mechanisms"][experiment][variant] = {"count": 0}
                continue
            value = {"count": len(records)}
            if experiment == "history":
                value.update(mean_output_rms=float(np.mean([r["effect"]["expression_rms"] for r in records])),
                    mean_per_second_rms=np.mean([r["effect"]["per_second_rms"] for r in records], 0).tolist(),
                    natural_mean_expression_mse=weighted([r["natural"] for r in records]),
                    reset_mean_expression_mse=weighted([r["reset"] for r in records]))
            elif experiment == "partner":
                for mode in ("shuffled", "coupling_off", "listening", "speaking"):
                    value[mode+"_mean_expression_mse"] = weighted([r[mode] for r in records])
                    value[mode+"_valid_chunks"] = sum(r[mode]["evaluated_chunks"] for r in records)
                value.update(shuffle_output_rms=float(np.mean([r["shuffle_effect"]["expression_rms"] for r in records])),
                    coupling_output_rms=float(np.mean([r["coupling_effect"]["expression_rms"] for r in records])),
                    natural_mean_expression_mse=weighted([r["natural"] for r in records]))
            else:
                for field in ("centered_trajectory_mse", "response_amplitude_mae", "peak_lag_error_seconds",
                              "late_response_error", "response_area_error"):
                    value[field] = float(np.mean([r["trajectory"][field] for r in records]))
                for field in ("single_effect", "repeated_effect"):
                    value[field] = np.mean([r[field]["per_second_rms"] for r in records], 0).tolist()
                for field in ("single_target_state_rms", "repeated_target_state_rms"):
                    value[field] = np.mean([r[field] for r in records], 0).tolist()
                value.update(natural_mean_expression_mse=weighted([r["natural"] for r in records]),
                    removed_mean_expression_mse=weighted([r["removed"] for r in records]),
                    anchor_removal_output_rms=float(np.mean([r["removal_effect"]["expression_rms"] for r in records])))
            summary["mechanisms"][experiment][variant] = value
    dump(output, summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--feature-root", required=True)
    prep.add_argument("--output", required=True)
    prep.add_argument("--pairs", type=int, default=32)
    prep.add_argument("--seed", type=int, default=6666)
    prep.add_argument("--split", default="test", choices=["test", "ood", "val"])
    runner = sub.add_parser("run")
    runner.add_argument("--checkpoint", required=True)
    runner.add_argument("--manifest", required=True)
    runner.add_argument("--output", required=True)
    runner.add_argument("--device", default="cuda:0")
    report = sub.add_parser("summarize")
    report.add_argument("--inputs", nargs="+", required=True)
    report.add_argument("--output", required=True)
    arguments = vars(parser.parse_args())
    command = arguments.pop("command")
    {"prepare": prepare, "run": run, "summarize": summarize}[command](**arguments)


if __name__ == "__main__":
    main()
