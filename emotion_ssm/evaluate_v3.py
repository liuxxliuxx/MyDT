"""Independent test/OOD evaluation with the checkpoint's original 25-frame protocol."""
import argparse
import json
from pathlib import Path

import torch.distributed as dist

from emotion_ssm.utils.checkpoint_v3 import load_avatar, read_checkpoint
from emotion_ssm.utils.distributed import init_distributed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("val", "test", "ood"), default="test")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-dialogues", type=int, default=0)
    args = parser.parse_args()
    payload = read_checkpoint(args.checkpoint)
    config = payload["config"]
    context = init_distributed(config["train"]["device"], config["train"]["seed"], True)
    model, _, payload = load_avatar(payload, context.device)
    del payload
    from emotion_ssm.train.generation_v3 import GenerationTokenDataset, evaluate
    dataset = GenerationTokenDataset(config["data"]["dualtalk_tokens"], config["data"]["dualtalk_raw"], args.split)
    metrics, dialogues = evaluate(model, dataset, context.device, context.rank, context.world_size, args.max_dialogues)
    if context.is_main:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({"protocol": config["protocol"], "split": args.split,
                         "checkpoint": args.checkpoint, "variant": model.variant,
                         "metrics": metrics, "dialogues": dialogues}, indent=2), encoding="utf-8")
        print(json.dumps(metrics), flush=True)
    context.barrier()
    if context.enabled:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
