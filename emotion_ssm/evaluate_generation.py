"""Evaluate a complete streaming checkpoint without upstream weight files."""
import argparse
import json
from pathlib import Path

from emotion_ssm.train.generation import evaluate, make_dataset
from emotion_ssm.utils.generation_checkpoint import load_generation


def evaluate_checkpoint(checkpoint, split, output, device="cpu", data_root=None,
                        feature_root=None, split_manifest=None, ablation="full", max_dialogues=0):
    model, cfg, metadata = load_generation(checkpoint, device)
    cfg = cfg.clone(); cfg.defrost()
    if data_root:
        cfg.DATA.DUALTALK_ROOT = str(data_root)
    if feature_root:
        cfg.DUALTALK.TIMED_FEATURE_ROOT = str(feature_root)
    if split_manifest:
        cfg.DUALTALK.SPLIT_MANIFEST = str(split_manifest)
    cfg.freeze()
    dataset = make_dataset(cfg, split, model.construction_info["feature_source"])
    metrics, records = evaluate(model, dataset, device, max_dialogues=max_dialogues, ablation=ablation)
    result = {"format_version": 2, "checkpoint": str(checkpoint), "global_step": metadata["global_step"],
              "variant": model.variant, "ablation": ablation, "split": split,
              "seed": cfg.SEED, "text_protocol": dataset.text_protocol,
              "split_digest": dataset.manifest_digest, "protocol": metadata["protocol"],
              "feature_digest": dataset.feature_digest,
              "evaluation_unit": "directed_clip; no verified cross-file dialogue mapping",
              "feature_source": model.construction_info["feature_source"],
              "complete_evaluation": not max_dialogues, "metrics": metrics, "dialogues": records}
    from emotion_ssm.train.common import readonly_roots
    from emotion_ssm.utils.paths import ensure_output_directory
    output = Path(output)
    ensure_output_directory(output.parent, readonly_roots(cfg))
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--split", choices=["val", "test", "ood"], default="test")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--feature-root", type=Path)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--ablation", choices=["full", "film_off", "self_only"], default="full")
    parser.add_argument("--max-dialogues", type=int, default=0)
    args = parser.parse_args()
    evaluate_checkpoint(**vars(args))


if __name__ == "__main__":
    main()
