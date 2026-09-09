"""Bind extraction models to domains actually trained in a Phase-B checkpoint."""
import argparse
import json
from pathlib import Path

from emotion_ssm.data.protocol import require_compatible, fingerprint
from emotion_ssm.utils.generation_checkpoint import read_checkpoint, checkpoint_config


def create(checkpoint, output, roots=None):
    payload = read_checkpoint(checkpoint)
    cfg = checkpoint_config(payload)
    if payload.get("format_version") != 2 or payload.get("data_semantics") != "masked-vad-endpoint-v2":
        raise ValueError("Re-train Phase B with v2 semantics before exporting feature sources")
    roots = roots or {"emotiontalk": cfg.DATA.EMOTIONTALK_ROOT, "iemocap": cfg.DATA.IEMOCAP_FEATURE_ROOT}
    domains = {}
    for name in cfg.DATA.SOURCES:
        source = json.loads((Path(roots[name])/"metadata"/"feature_source.json").read_text(encoding="utf-8"))
        require_compatible(source, source)
        recorded = payload.get("feature_sources", {}).get(name)
        if recorded is None:
            raise ValueError(f"Checkpoint has no feature provenance for {name}; regenerate Phase B")
        require_compatible(recorded, source)
        domains[name] = {**source, "trained": True}
    result = {"version": 2, "domains": domains, "phase_b": str(checkpoint),
              "training_config_digest": fingerprint(payload["config"])}
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    create(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
