"""Reproducible upstream training, calibration and matched-budget controls."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

VARIANTS = ("none", "affect", "self", "dyadic")


def planned_commands(args):
    sources = ["emotiontalk", "iemocap"] if args.source == "all" else [args.source]
    seeds = args.seeds or ([6666] if args.mode == "pilot" else [6666, 6667, 6668])
    steps = 1000 if args.mode == "pilot" else 30000
    selected = {"upstream", "features", "calibration", "controls", "evaluate"} if args.stages == "all" else set(args.stages.split(","))
    unknown = selected - {"upstream", "features", "calibration", "controls", "evaluate"}
    if unknown:
        raise ValueError(f"Unknown stages: {unknown}")
    commands = []
    python = [sys.executable, "-B", "-m"]
    train_python = python if args.nproc == 1 else [sys.executable, "-m", "torch.distributed.run",
        "--standalone", "--nproc_per_node", str(args.nproc), "-m"]
    for seed in seeds:
        upstream = Path(args.output_root) / args.source / f"seed{seed}"
        controls = Path(args.output_root) / args.source / args.mode / f"seed{seed}"
        artifacts = Path(args.artifacts_root) / args.source
        source_manifest = artifacts / f"seed{seed}" / "feature_sources.json"
        timed = artifacts / "timed"
        a0 = upstream / "phase_a_observation" / "upstream_v2"
        phase_a = upstream / "phase_a_dynamics" / "upstream_v2" / "phase_a_best.pt"
        phase_b = upstream / "phase_b_coupling" / "upstream_v2" / "phase_b_best.pt"
        calibration = upstream / "dualtalk_calibration" / "upstream_v2" / "observer.pt"
        common = ["SEED", str(seed), "DEVICE", args.device, "DATA.SOURCES", repr(sources),
            "DATA.EMOTIONTALK_ROOT", str(args.emotiontalk_root), "DATA.IEMOCAP_FEATURE_ROOT", str(args.iemocap_root),
            "DATA.DUALTALK_ROOT", str(args.data_root), "TRAIN.OUTPUT_ROOT", str(upstream),
            "TRAIN.EXPERIMENT_NAME", "upstream_v2", "DATA.ALLOW_LEGACY_FEATURES", "False"]
        observer = ["TRAIN.OBSERVATION_CHECKPOINT", str(a0/"observation_encoder.pt"),
            "TRAIN.EMA_TEACHER_CHECKPOINT", str(a0/"ema_teacher.pt"),
            "TRAIN.EMOTION_HEADS_CHECKPOINT", str(a0/"emotion_heads.pt")]
        generation = [*common, "DUALTALK.SOURCE_MANIFEST", str(source_manifest),
            "DUALTALK.PHASE_B_CHECKPOINT", str(phase_b), "DUALTALK.BASELINE_CHECKPOINT", str(args.baseline),
            "DUALTALK.CALIBRATION_CHECKPOINT", str(calibration),
            "DUALTALK.TIMED_FEATURE_ROOT", str(timed), "DUALTALK.SPLIT_MANIFEST", str(timed/"splits.json"),
            "DUALTALK.SPLIT_SEED", "6666", "DUALTALK.ADAPTER_INIT_SOURCE", "-1"]
        def stage(module, config, options, distributed=True):
            commands.append([*(train_python if distributed else python), module, "--config", config, *options])
        if "upstream" in selected:
            stage("emotion_ssm.train.phase_a_observation", "configs/phase_a_observation.yaml", common)
            stage("emotion_ssm.train.phase_a_dynamics", "configs/phase_a_dynamics.yaml", [*common, *observer,
                "DYNAMICS.FULL_DIALOGUES", "True", "DYNAMICS.BPTT_EVENTS", "32", "DYNAMICS.ROLLOUT_MODE", "joint"])
            stage("emotion_ssm.train.phase_b", "configs/phase_b.yaml", [*common, *observer,
                "TRAIN.PHASE_A_CHECKPOINT", str(phase_a), "DYNAMICS.KEEP_AFFECT_FROZEN", "True",
                "DYNAMICS.FULL_DIALOGUES", "True", "DYNAMICS.BPTT_EVENTS", "32", "DYNAMICS.ROLLOUT_MODE", "joint"])
        if "features" in selected:
            commands.append([*python, "emotion_ssm.preprocess.source_manifest", "--checkpoint", str(phase_b), "--output", str(source_manifest)])
            commands.append([*python, "emotion_ssm.preprocess.dualtalk", "--config", "configs/dualtalk_conditioned.yaml",
                "--output", str(timed), "--device", args.device,
                *(["--words-root", str(args.words_root)] if args.words_root else []), *generation])
        if "calibration" in selected:
            stage("emotion_ssm.train.calibrate_dualtalk", "configs/dualtalk_conditioned.yaml", [*generation,
                "TRAIN.MAX_STEPS", str(args.calibration_steps), "TRAIN.VAL_EVERY_STEPS", "250"], distributed=True)
        if "controls" in selected:
            for variant in VARIANTS:
                stage("emotion_ssm.train.generation", f"configs/dualtalk_v2_{variant}.yaml", [*generation,
                    "TRAIN.OUTPUT_ROOT", str(controls), "TRAIN.EXPERIMENT_NAME", variant,
                    "TRAIN.MAX_STEPS", str(steps), "TRAIN.GLOBAL_CHUNKS_PER_STEP", "32", "TRAIN.LR", "0.0001",
                    "TRAIN.VAL_EVERY_STEPS", "1000", "LOSS.GENERATION_STATE", "0.0"])
        if "evaluate" in selected:
            for variant in VARIANTS:
                for split in ("test", "ood"):
                    checkpoint = controls / "dualtalk_conditioned" / variant / "best_generation.pt"
                    commands.append([*python, "emotion_ssm.evaluate_generation", "--checkpoint", str(checkpoint),
                        "--split", split, "--data-root", str(args.data_root), "--feature-root", str(timed),
                        "--split-manifest", str(timed/"splits.json"), "--device", args.device,
                        "--output", str(controls/"evaluation"/f"{variant}_{split}.json")])
    return commands


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", choices=["all", "emotiontalk", "iemocap"], default="all")
    p.add_argument("--mode", choices=["pilot", "full"], default="pilot")
    p.add_argument("--stages", default="controls", help="comma-separated upstream,features,calibration,controls,evaluate; or all")
    p.add_argument("--seeds", nargs="+", type=int)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--data-root", type=Path, default=Path("datasets/dualtalk"))
    p.add_argument("--emotiontalk-root", type=Path, default=Path("artifacts/features/emotiontalk_v2"))
    p.add_argument("--iemocap-root", type=Path, default=Path("artifacts/features/iemocap_v2"))
    p.add_argument("--output-root", type=Path, default=Path("runs/v2"))
    p.add_argument("--artifacts-root", type=Path, default=Path("artifacts/v2"))
    p.add_argument("--words-root", type=Path)
    p.add_argument("--calibration-steps", type=int, default=3000)
    p.add_argument("--nproc", type=int, default=1)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--print-only", action="store_true")
    return p


def main():
    args = parser().parse_args()
    if args.nproc < 1 or 32 % args.nproc:
        raise ValueError("nproc must be a positive divisor of 32")
    for command in planned_commands(args):
        print(json.dumps(command, ensure_ascii=False), flush=True)
        if not args.print_only:
            subprocess.run(command, cwd=Path(__file__).resolve().parents[1], check=True)


if __name__ == "__main__":
    main()
