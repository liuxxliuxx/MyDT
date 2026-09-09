"""Validate the actual exported A0 best checkpoint before downstream training."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emotion_ssm.train.observation_v3 import semantic_gate
from emotion_ssm.utils.checkpoint_v3 import read_checkpoint


def validate_export(root, *, allow_untrained=False):
    root = Path(root)
    path = root / "best.pt"
    payload = read_checkpoint(path)
    if payload["kind"] != "observation_v3":
        raise ValueError("A0 gate requires an observation_v3 checkpoint")
    selected = payload.get("construction", {}).get("exported_model")
    metrics = payload.get("metrics", {})
    if selected not in ("student", "ema") or metrics.get("selected_model") != selected:
        raise ValueError("Checkpoint lacks consistent explicit student/EMA export identity")
    reference = "observer" if selected == "student" else "ema"
    export_weights, selected_weights = payload["models"].get("teacher"), payload["models"].get(reference)
    if export_weights is None or selected_weights is None or export_weights.keys() != selected_weights.keys():
        raise ValueError("Checkpoint does not contain the selected exported teacher weights")
    if any(not torch.equal(value, selected_weights[name]) for name, value in export_weights.items()):
        raise ValueError("Exported teacher weights differ from the model whose metrics were selected")
    evaluated = metrics.get(selected, {})
    if evaluated.get("evaluation_model") != selected or evaluated.get("evaluation_split") != "val":
        raise ValueError("Selected weights lack independent validation metrics")
    if evaluated.get("training_statistics", {}).get("split") != "train":
        raise ValueError("Constant baselines must be calculated from the training split only")
    report = semantic_gate(evaluated, allow_untrained=allow_untrained)
    report.update(checkpoint=str(path.resolve()), checkpoint_step=int(payload["global_step"]),
                  checkpoint_protocol=payload["protocol"], selected_model=selected,
                  validation_source="metrics stored in actual best.pt; validation.json is not consulted",
                  bypass_authorization="explicit --allow-untrained (smoke only)" if allow_untrained else None,
                  exported_weights_verified=True)
    (root / "a0_semantic_gate.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--allow-untrained", action="store_true", help="Explicitly bypass semantic quality only for smoke runs")
    args = parser.parse_args()
    report = validate_export(args.root, allow_untrained=args.allow_untrained)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    if not report["allowed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
