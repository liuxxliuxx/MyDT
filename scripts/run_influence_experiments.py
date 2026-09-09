"""Run the fixed three-experiment pilot on one explicitly allowed evaluation GPU."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    allowed = {"GPU-5b497823-4a84-bde7-5670-2172ee96245d", "GPU-9117039b-5194-5d46-d4a7-088ff06ce552"}
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible not in allowed:
        raise ValueError("Select exactly one of server1 physical GPU2/3 by its verified UUID")
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = args.output_root / "manifest.json"
    base = [sys.executable, "-B", "-u", "-m", "emotion_ssm.evaluate_influence"]
    if not manifest.exists():
        subprocess.run(base + ["prepare", "--feature-root", str(args.feature_root),
            "--output", str(manifest), "--pairs", "32", "--seed", "6666"], check=True)
    for variant in ("none", "affect", "self", "dyadic"):
        output = args.output_root / (variant + ".json")
        if output.exists():
            saved = json.loads(output.read_text(encoding="utf-8"))
            if saved.get("complete") and saved["manifest_digest"] == json.loads(manifest.read_text())["digest"]:
                continue
            raise ValueError(f"Existing output has a different protocol: {output}")
        command = base + ["run", "--checkpoint", str(args.checkpoint_root/variant/"best_generation.pt"),
            "--manifest", str(manifest), "--output", str(output), "--device", "cuda:0"]
        print(json.dumps({"variant": variant, "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                          "command": command}), flush=True)
        with (args.output_root / (variant + ".log")).open("w", encoding="utf-8") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    subprocess.run(base + ["summarize", "--inputs"] + [str(args.output_root/(v+".json"))
        for v in ("none", "affect", "self", "dyadic")] +
        ["--output", str(args.output_root/"summary.json")], check=True)
    print(json.dumps({"complete": True, "output": str(args.output_root)}), flush=True)


if __name__ == "__main__":
    main()
