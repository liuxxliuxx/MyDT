"""Stop before formal experiments if pilot training is numerically invalid."""
import argparse
import json
import math
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def records(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def check_run(root, allow_untrained=False):
    root = Path(root)
    failures, result = [], {"variant_metrics": {}, "allow_untrained": allow_untrained}
    steps, blocks = set(), set()
    for variant in ("none", "affect", "self", "dyadic"):
        folder = root / "pilot" / variant
        if not (folder / "best.pt").is_file():
            failures.append(f"{variant}: best checkpoint missing")
            continue
        status = read(folder / "training_status.json")
        if status["status"] != "complete":
            failures.append(f"{variant}: pilot incomplete")
        steps.add(status["step"])
        values = records(folder / "train_metrics.jsonl")
        for row in values:
            blocks.add(row["global_valid_blocks"])
            for key in ("loss", "grad_norm", "generation_total", "generator_grad_norm", "state_grad_norm", "observer_grad_norm"):
                if key in row and not math.isfinite(row[key]):
                    failures.append(f"{variant}: nonfinite {key}")
        if not values or not any(row.get("generator_grad_norm", row.get("grad_norm", 0)) > 0 for row in values):
            failures.append(f"{variant}: no generator gradient observed")
        if variant in ("self", "dyadic") and not any(row.get("state_grad_norm", 0) > 0 for row in values):
            failures.append(f"{variant}: no gradient reached the persistent state")
        validation = read(folder / "validation.json")["metrics"]
        if not validation.get("valid_frames"):
            failures.append(f"{variant}: no valid validation frames")
        result["variant_metrics"][variant] = validation
        if not allow_untrained:
            # This catches a numerical constant, not a claim of useful emotion.
            representations = validation.get("representation_diagnostics", {})
            if not representations:
                failures.append(f"{variant}: representation diagnostics missing")
            for name, item in representations.items():
                if isinstance(item, dict) and item.get("samples", 0) >= 16:
                    std = item.get("embedding_std", item.get("std"))
                    if std is not None and (not math.isfinite(std) or std < .001):
                        failures.append(f"{variant}/{name}: affect representation collapsed")
    if len(steps) != 1 or blocks != {32}:
        failures.append(f"Control budgets differ: steps={steps}, new_blocks={blocks}")
    result.update(passed=not failures, failures=failures,
                  interpretation="Code/numerical gate only. Better emotion or expression requires independent paired evaluation.")
    path = root / "pilot_gate.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if failures:
        raise RuntimeError("Pilot rejected: " + "; ".join(failures))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--allow-untrained", action="store_true", help="Two-step smoke test only; skip representation quality threshold")
    args = parser.parse_args()
    result = check_run(args.root, args.allow_untrained)
    print(json.dumps({"passed": result["passed"], "root": args.root}), flush=True)


if __name__ == "__main__":
    main()
