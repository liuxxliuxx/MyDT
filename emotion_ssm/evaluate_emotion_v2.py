"""Independent emotion evaluation from one complete checkpoint."""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from emotion_ssm.data import UnifiedUtteranceDataset, build_speaker_vocabulary
from emotion_ssm.data.full_dialogues import FullDialogueDataset, collate_full_dialogues
from emotion_ssm.evaluate import evaluate_observation, state_curves
from emotion_ssm.train.common import build_feature_stores, readonly_roots
from emotion_ssm.train.dynamics_core import DynamicsTrainingBundle
from emotion_ssm.train.dynamics_trainer import validate
from emotion_ssm.utils.distributed import DistributedContext
from emotion_ssm.utils.emotion_checkpoint import load_emotion
from emotion_ssm.utils.paths import ensure_output_directory


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--emotiontalk-root", type=Path)
    p.add_argument("--iemocap-root", type=Path)
    p.add_argument("--batch-size", type=int, default=0)
    args = p.parse_args()
    device = torch.device(args.device)
    bundle, teacher, cfg, checkpoint = load_emotion(args.checkpoint, device)
    cfg = cfg.clone(); cfg.defrost()
    if args.emotiontalk_root:
        cfg.DATA.EMOTIONTALK_ROOT = str(args.emotiontalk_root)
    if args.iemocap_root:
        cfg.DATA.IEMOCAP_FEATURE_ROOT = str(args.iemocap_root)
    cfg.freeze()
    stores = build_feature_stores(cfg)
    vocabulary = build_speaker_vocabulary(stores)
    if isinstance(bundle, DynamicsTrainingBundle):
        dataset = FullDialogueDataset(stores, args.split, vocabulary)
        training = FullDialogueDataset(stores, "train", vocabulary)
        loader = DataLoader(dataset, batch_size=args.batch_size or cfg.TRAIN.SEQUENCE_BATCH_SIZE,
                            shuffle=False, collate_fn=collate_full_dialogues)
        bundle.rollout_mode = "joint"
        context = DistributedContext(0, 0, 1, device)
        metrics = validate(bundle, teacher, loader, training.class_weights().to(device), context,
                           cfg.DYNAMICS.ENABLE_PARTNER, cfg.DYNAMICS.ENABLE_PARTNER)
        metrics["state_curves"] = state_curves(bundle.state_model)
    else:
        dataset = UnifiedUtteranceDataset(stores, args.split, vocabulary)
        loader = DataLoader(dataset, batch_size=args.batch_size or cfg.TRAIN.BATCH_SIZE, shuffle=False)
        metrics, _ = evaluate_observation(bundle.encoder, bundle.heads, loader, device)
    result = {"format_version": 2, "clock": "observation_endpoint", "split": args.split,
              "checkpoint": str(args.checkpoint), "global_step": checkpoint["global_step"],
              "feature_sources": checkpoint.get("feature_sources", {}),
              "split_digests": checkpoint.get("split_digests", {}), "metrics": metrics}
    ensure_output_directory(args.output.parent, readonly_roots(cfg))
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
