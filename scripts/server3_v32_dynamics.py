"""One-off server3 flow experiment: verify imports, smoke, then fresh training.

Server1 is never modified. The old calibrated observer defines fixed affect
coordinates; the new masking policy is a separate A0 experiment, not claimed
by this dynamics-only run. Credentials are not accepted or stored here.
"""
from __future__ import annotations

import argparse
import copy
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def run(root, run_id, gpu):
    root = Path(root).resolve()
    folder = root / "runs" / run_id
    folder.mkdir(parents=True, exist_ok=True)
    os.chdir(root)
    sys.path.insert(0, str(root))
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "PYTHONUNBUFFERED": "1",
           "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
           "TOKENIZERS_PARALLELISM": "false", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
           "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    os.environ.update(env)

    def status(stage, **fields):
        record = {"stage": stage, "updated": datetime.datetime.now().astimezone().isoformat(),
                  "pid": os.getpid(), "gpu_uuid": gpu, **fields}
        write(folder / "status.json", record)
        print(json.dumps(record), flush=True)

    def execute(stage, command):
        log = folder / (stage + ".log")
        status(stage, command=command, log=str(log))
        with log.open("ab", buffering=0) as handle:
            subprocess.run(command, cwd=root, env=env, stdout=handle, stderr=handle, check=True)

    try:
        previous = None
        while True:
            dependencies = {}
            for name in ("transfer", "environment"):
                path = folder / (name + "_status.json")
                try:
                    dependencies[name] = json.loads(path.read_text()) if path.exists() else {"status": "pending"}
                except json.JSONDecodeError:
                    dependencies[name] = {"status": "publishing"}
            if any(value["status"] == "failed" for value in dependencies.values()):
                raise RuntimeError("Setup failed; inspect dependency logs")
            signature = [(name, value["status"], value.get("stage")) for name, value in dependencies.items()]
            if signature != previous:
                status("waiting_for_dependencies", dependencies=dependencies)
                previous = signature
            if all(value["status"] == "complete" for value in dependencies.values()):
                break
            time.sleep(15)

        import torch
        from emotion_ssm.config_v3 import default_config, validate_config, write_config, DYNAMICS_REVISION
        from emotion_ssm.utils.checkpoint_v3 import read_checkpoint, manifest_provenance
        assert torch.cuda.is_available() and torch.cuda.device_count() == 1
        gate = json.loads((folder / "setup/source_a0_gate.json").read_text())
        if not gate.get("passed") or gate.get("bypassed"):
            raise ValueError("The archived upstream A0 semantic gate did not pass")
        source_path = folder / "upstream/calibration_best.pt"
        upstream = read_checkpoint(source_path)
        if upstream["kind"] != "calibration_v3":
            raise ValueError("Expected the complete calibrated observer checkpoint")
        original = json.loads((folder / "upstream/source_dynamics_config.json").read_text())
        config = copy.deepcopy(original)
        defaults = default_config()
        config["state"] = copy.deepcopy(defaults["state"])
        config["observer"] = copy.deepcopy(upstream["construction"]["observer"])
        config["train"].update(device="cuda:0", max_steps=10000, dynamics_steps=10000,
            dynamics_revision=DYNAMICS_REVISION, generation_revision=defaults["train"]["generation_revision"],
            masking=None, learning_revision=upstream["config"]["train"].get("learning_revision"),
            global_chunks_per_step=32, checkpoint_every=250, validate_every=250, log_every=10,
            validation_max_dialogues=16, experiment_scope="flow_only_fixed_v313_observer",
            compile_adaptive_flow=True)
        imported = root / "artifacts/v3_2_import_20260909"
        config["data"]["token_roots"] = [str(imported / (name + "_tokens"))
                                            for name in ("emotiontalk", "iemocap", "dualtalk")]
        config["data"]["dualtalk_tokens"] = str(imported / "dualtalk_tokens")
        config["data"]["dualtalk_raw"] = str(root / "datasets/dualtalk")
        config["paths"].update(output=str(folder / "training/seed6666/dynamics"),
            observation_checkpoint=str(source_path), dynamics_checkpoint="", baseline="", resume="")
        validate_config(config)
        current = manifest_provenance(config)
        canonical = lambda values: sorted(json.dumps(value, sort_keys=True) for value in values)
        if canonical(current.values()) != canonical(upstream["provenance"].values()):
            raise ValueError("Imported tokens differ from the observer's feature/split provenance")
        files = 0
        for token_root in config["data"]["token_roots"]:
            directory = Path(token_root)
            manifest = json.loads((directory / "manifest.json").read_text())
            for item in manifest["dialogues"].values():
                if not (directory / item["path"]).is_file():
                    raise FileNotFoundError(directory / item["path"])
                files += 1
        write(folder / "experiment_protocol.json", {
            "scope": "flow_only_fixed_v313_observer", "state": config["state"],
            "dynamics_revision": DYNAMICS_REVISION, "upstream_learning_revision": config["train"]["learning_revision"],
            "new_context_masking_trained": False, "optimizer_resumed": False,
            "upstream_checkpoint_sha256": sha(source_path), "token_dialogues": files,
            "source_a0_gate_passed": True, "provenance": current,
            "gpu_uuid": gpu, "world_size": 1, "seed": config["train"]["seed"],
            "steps": 10000, "global_new_blocks_per_step": 32,
            "flow_execution": "compiled", "matmul_precision": "highest",
            "server1_training_modified": False, "best_selection": "raw_affect_mse",
            "semantic_metrics_reported_separately": True})
        write_config(folder / "configs/dynamics.json", config)
        receipt_path = folder / "preflight_receipt.json"
        reusable = False
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            reusable = (receipt.get("passed") and receipt.get("gpu_optimizer_steps") == 2
                and receipt.get("regression_passed") == 38 and receipt.get("state") == config["state"]
                and receipt.get("source_sha256") == sha(source_path)
                and all(receipt.get(key) == config["train"][key] for key in
                    ("global_chunks_per_step", "forecast_seconds", "tbptt_seconds", "compile_adaptive_flow"))
                and receipt.get("code_sha256") and all(sha(root/name) == digest
                    for name, digest in receipt["code_sha256"].items()))
        if not reusable:
            execute("regression", [sys.executable, "-m", "pytest", "-q", "--disable-warnings",
                "tests/test_v32_adaptive_flow.py", "tests/test_v32_context_masking.py",
                "tests/test_v32_generation.py", "tests/test_v32_protocol.py",
                "tests/test_v3_dynamics_training.py", "tests/test_v3_checkpoint.py",
                "--basetemp", str(folder / ("pytest_" + str(time.time_ns())))])
            smoke = copy.deepcopy(config)
            smoke["train"].update(max_steps=2, dynamics_steps=2, validate_every=2,
                                  checkpoint_every=2, log_every=1, validation_max_dialogues=3)
            smoke["paths"]["output"] = str(folder / "smoke")
            write_config(folder / "configs/smoke.json", smoke)
            execute("smoke", [sys.executable, "-B", "-m", "emotion_ssm.train.dynamics_v3", "--config",
                               str(folder / "configs/smoke.json")])
        else:
            status("preflight_reused", receipt=str(receipt_path), smoke_dataset=receipt["smoke_dataset"])
        execute("dynamics", [sys.executable, "-B", "-m", "emotion_ssm.train.dynamics_v3", "--config",
                              str(folder / "configs/dynamics.json")])
        status("complete", output=config["paths"]["output"])
    except Exception as error:
        status("failed", error=str(error))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    if Path(args.run_id).name != args.run_id or not args.run_id.startswith("v3_2_"):
        raise ValueError("Expected a distinct v3.2 run directory")
    if args.launch:
        folder = Path(args.root).resolve() / "runs" / args.run_id
        folder.mkdir(parents=True, exist_ok=True)
        with (folder / "pipeline.log").open("ab", buffering=0) as handle:
            child = subprocess.Popen([sys.executable, "-u", __file__, "--root", args.root,
                "--run-id", args.run_id, "--gpu", args.gpu], cwd=args.root,
                stdin=subprocess.DEVNULL, stdout=handle, stderr=handle, start_new_session=True)
        print(json.dumps({"pipeline_pid": child.pid, "log": str(folder / "pipeline.log")}))
    else:
        run(args.root, args.run_id, args.gpu)


if __name__ == "__main__":
    main()
