"""Paired dialogue effects and seed dispersion for completed v2 evaluations."""
import argparse
import json
from pathlib import Path

import numpy as np
from emotion_ssm.data.protocol import source_video_id


def paired_clips(rows):
    """Combine both directions of a paired clip before estimating uncertainty."""
    result = {}
    for name, row in rows.items():
        clip = name.removesuffix("_speaker1").removesuffix("_speaker2")
        totals = result.setdefault(clip, {"sse": {}, "elements": {}})
        for field in ("sse", "elements"):
            for key, value in row[field].items():
                totals[field][key] = totals[field].get(key, 0) + value
    for row in result.values():
        row.update({f"{key}_mse": value/max(row["elements"][key], 1) for key, value in row["sse"].items()})
    return result


def compare(baselines, conditions, bootstrap=5000, seed=6666):
    if len(baselines) != len(conditions) or not baselines:
        raise ValueError("Provide one baseline and condition evaluation per matching seed")
    if bootstrap < 1:
        raise ValueError("bootstrap must be positive")
    if len({value["variant"] for value in conditions}) != 1 or conditions[0]["variant"] == "none":
        raise ValueError("Compare one conditioned variant across all seeds")
    seed_values, paired, common_names = [], [], None
    metrics = [f"{name}_mse" for name in ("expression", "jaw", "neck", "velocity", "boundary_velocity")]
    for base, conditioned in zip(baselines, conditions):
        for value in (base, conditioned):
            if value.get("format_version") != 2 or not value.get("complete_evaluation"):
                raise ValueError("Only complete v2 evaluations can enter a comparison")
        for field in ("seed", "split", "split_digest", "feature_digest", "text_protocol", "feature_source", "protocol"):
            if base[field] != conditioned[field]:
                raise ValueError(f"Comparison protocol differs: {field}")
            if field != "seed" and base[field] != baselines[0][field]:
                raise ValueError(f"Seeds must share the same protocol: {field}")
        if base["variant"] != "none" or base["ablation"] != "full" or conditioned["ablation"] != "full":
            raise ValueError("Use independently retrained controls, with full evaluation")
        b = {row["dialogue"]: row for row in base["dialogues"]}
        c = {row["dialogue"]: row for row in conditioned["dialogues"]}
        if set(b) != set(c) or len(b) != len(base["dialogues"]) or len(c) != len(conditioned["dialogues"]):
            raise ValueError("Missing or duplicated paired dialogues")
        for name in b:
            if b[name]["elements"] != c[name]["elements"]:
                raise ValueError(f"Different valid targets for {name}")
        b, c = paired_clips(b), paired_clips(c)
        names = sorted(b)
        if not names:
            raise ValueError("Evaluation has no paired clips")
        if common_names is not None and names != common_names:
            raise ValueError("Seeds must evaluate the same dialogue set")
        common_names = names
        for name in names:
            if b[name]["elements"] != c[name]["elements"]:
                raise ValueError(f"Different valid targets for {name}")
        seed_values.append([conditioned["metrics"][m]-base["metrics"][m] for m in metrics])
        paired.append([[c[name][m]-b[name][m] for m in metrics] for name in names])
    if len({value["seed"] for value in baselines}) != len(baselines):
        raise ValueError("Duplicate training seeds")
    seed_values, paired = np.asarray(seed_values), np.asarray(paired)
    effects = paired.mean(0)
    rng = np.random.default_rng(seed)
    groups = {}
    for index, name in enumerate(common_names):
        groups.setdefault(source_video_id(name), []).append(index)
    groups = list(groups.values())
    boot = np.stack([effects[np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))])].mean(0)
                     for _ in range(bootstrap)])
    return {"direction": "condition minus baseline; negative is lower error",
        "seeds": [value["seed"] for value in baselines], "paired_clips": len(effects), "source_clusters": len(groups),
        "split": baselines[0]["split"], "text_protocol": baselines[0]["text_protocol"],
        "metrics": {metric: {"seed_weighted_delta_mean": float(seed_values[:, i].mean()),
            "seed_weighted_delta_std": float(seed_values[:, i].std(ddof=1)) if len(seed_values)>1 else None,
            "paired_clip_delta_mean": float(effects[:, i].mean()),
            "source_cluster_95ci": np.quantile(boot[:, i], [.025, .975]).tolist(),
            "per_seed_weighted_delta": seed_values[:, i].tolist()}
            for i, metric in enumerate(metrics)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, nargs="+", required=True)
    parser.add_argument("--condition", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    read = lambda paths: [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    result = compare(read(args.baseline), read(args.condition))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
