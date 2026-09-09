"""Server-1 V3.1 preprocessing, gated emotion learning and two-GPU controls."""
from __future__ import annotations

import argparse
import copy
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

GPU_UUIDS = ("GPU-5b497823-4a84-bde7-5670-2172ee96245d", "GPU-9117039b-5194-5d46-d4a7-088ff06ce552")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def merge_token_shards(shards, destination):
    """Reference shard files without copying large caches or dropping provenance."""
    from emotion_ssm.data.protocol import fingerprint
    from emotion_ssm.config_v3 import PROTOCOL
    from emotion_ssm.data.packets_v3 import SUPERVISION_REVISION, validate_supervision_manifest
    merged = {"protocol": PROTOCOL, "feature_sources": {},
              "splits": {s: [] for s in ("train", "val", "test", "ood")},
              "dialogues": {}, "data_gaps": [], "errors": [], "quality": {}}
    destination = Path(destination)
    for shard in shards:
        shard = Path(shard)
        manifest = json.loads((shard / "manifest.json").read_text())
        if manifest["protocol"] != merged["protocol"] or manifest.get("errors"):
            raise ValueError(f"Unusable token shard: {shard}")
        validate_supervision_manifest(manifest)
        for key, source in manifest["feature_sources"].items():
            if key in merged["feature_sources"] and merged["feature_sources"][key] != source:
                raise ValueError("Feature sources differ between shards")
            merged["feature_sources"][key] = source
        for name, record in manifest["dialogues"].items():
            if name in merged["dialogues"]:
                raise ValueError("Duplicate dialogue in disjoint token shards")
            merged["dialogues"][name] = {**record, "path": os.path.relpath(shard / record["path"], destination)}
        for split, names in manifest["splits"].items():
            merged["splits"][split].extend(names)
        merged["data_gaps"].extend(manifest.get("data_gaps", []))
        for domain, quality in manifest.get("quality", {}).items():
            if domain not in merged["quality"]:
                merged["quality"][domain] = copy.deepcopy(quality)
                continue
            target = merged["quality"][domain]
            target["role_packets"] += quality["role_packets"]
            target["text_missing_role_streams"] += quality["text_missing_role_streams"]
            target["text_missing_dialogues"].extend(quality["text_missing_dialogues"])
            for counter in ("endpoint_utterances", "utterances_longer_than_audio_history"):
                target[counter] = target.get(counter, 0) + quality.get(counter, 0)
            for modality, number in quality["modality_available_packets"].items():
                target["modality_available_packets"][modality] += number
            for modality, counts in quality["tokens"].items():
                for name, number in counts.items():
                    target["tokens"][modality][name] += number
            if target.get("inherited_alignment_report") != quality.get("inherited_alignment_report"):
                raise ValueError("Alignment provenance differs between token shards")
        if manifest.get("diagnostic_subset"):
            merged["diagnostic_subset"] = True
    for names in merged["splits"].values():
        names.sort()
    # IE sources reached here only after the interpretation revision was checked.
    # ET/DT legacy sources retain their compatible label semantics.
    merged["supervision_revision"] = SUPERVISION_REVISION
    merged["digest"] = fingerprint(merged)
    write_json(destination / "manifest.json", merged)
    return merged


def validate_reused_artifacts(roots, smoke=False):
    """A small diagnostic cache must never silently become a formal dataset."""
    from emotion_ssm.config_v3 import PROTOCOL
    from emotion_ssm.data.packets_v3 import validate_supervision_manifest
    for root in roots:
        path = Path(root) / "manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("protocol") != PROTOCOL or manifest.get("errors"):
            raise ValueError(f"Incompatible or incomplete V3.1 cache: {path}")
        validate_supervision_manifest(manifest)
        if manifest.get("diagnostic_subset") and not smoke:
            raise ValueError(f"Formal training cannot reuse diagnostic subset: {path}")


def resolve_reused_token_roots(specifications, sources):
    """Reuse explicitly named, version-checked sources without moving caches."""
    roots = {}
    for specification in specifications:
        name, separator, value = specification.partition("=")
        if not separator or not value or name not in sources or name in roots:
            raise ValueError("Each reused token root must be a unique selected dataset=path")
        roots[name] = Path(value).resolve()
    if set(roots) != set(sources):
        raise ValueError("Explicit token reuse must supply every selected dataset")
    return {name: roots[name] for name in sources}


def resolve_reused_observation_calibration(specifications, seeds):
    result = {}
    for specification in specifications:
        key, separator, value = specification.partition("=")
        if not separator or not key.isdecimal() or not value:
            raise ValueError("Upstream reuse requires a unique selected SEED=SEED_DIR")
        seed = int(key)
        if seed not in seeds or seed in result:
            raise ValueError("Upstream reuse requires a unique selected SEED=SEED_DIR")
        result[seed] = Path(value).resolve()
    return result


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reuse_observation_calibration(source, destination, config):
    """Copy completed same-seed upstream exports; never restore their optimizers.

    Check and gate private staged copies before publishing the new seed directory.
    Identical completed copies may be inspected again, but existing downstream
    work or different destination content is never overwritten.
    """
    import torch
    from emotion_ssm.config_v3 import validate_config
    from emotion_ssm.models.token_observer import TokenObserver
    from emotion_ssm.utils.checkpoint_v3 import manifest_provenance, read_checkpoint
    from scripts.validate_a0_v31 import validate_export

    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("Upstream source and new seed destination must be disjoint")
    if any((destination / name).exists() for name in ("dynamics", "pilot", "formal")):
        raise ValueError("Reused upstream requires fresh dynamics and generation output directories")
    if not config["train"].get("require_a0_gate", True) or config["paths"].get("resume"):
        raise ValueError("Formal upstream reuse requires the A0 gate and fresh downstream training")
    seed = int(config["train"]["seed"])
    canonical = lambda value: {str(Path(root).resolve()): entry for root, entry in value.items()}
    expected = canonical(manifest_provenance(config))
    if not expected:
        raise ValueError("Upstream reuse requires actual token manifest provenance")
    validate_reused_artifacts(expected, smoke=False)
    expected_observer = TokenObserver(config["observer"]).construction()
    records = {}
    for stage, kind, budget_key in (("observation", "observation_v3", "observation_steps"),
                                    ("calibration", "calibration_v3", "calibration_steps")):
        path = source / stage / "best.pt"
        digest = _file_sha256(path)
        payload = read_checkpoint(path)
        required = ("config", "construction", "models", "global_step", "metrics", "run_state",
                    "optimizer", "rng_state", "provenance")
        if payload.get("kind") != kind or any(name not in payload for name in required):
            raise ValueError(f"Incomplete or wrong-kind {stage} checkpoint: {path}")
        original = validate_config(payload["config"])
        if (original["train"].get("learning_revision") != config["train"].get("learning_revision")
                or original["train"].get("masking") != config["train"].get("masking")):
            raise ValueError(f"Source {stage} masking/learning protocol differs; retrain upstream for the new experiment")
        if int(original["train"]["seed"]) != seed:
            raise ValueError(f"Source {stage} train.seed does not match requested seed {seed}")
        actual = canonical(manifest_provenance(original))
        if canonical(payload["provenance"]) != actual or actual != expected:
            raise ValueError(f"Source {stage} token manifest provenance differs from actual/new config")
        if not isinstance(payload["optimizer"], dict) or not payload["rng_state"] or not payload["run_state"]:
            raise ValueError(f"Incomplete training state in {stage} checkpoint")
        if not {"state", "param_groups"}.issubset(payload["optimizer"]):
            raise ValueError(f"Incomplete optimizer archive in {stage} checkpoint")
        model_names = ("observer", "teacher", "ema") if stage == "observation" else (
            "observer", "teacher", "coordinate_teacher")
        if not set(model_names).issubset(payload["models"]) or "observer" not in payload["construction"]:
            raise ValueError(f"Incomplete self-contained {stage} models")
        observer = TokenObserver(payload["construction"]["observer"])
        if observer.construction() != expected_observer:
            raise ValueError(f"Source {stage} observer construction differs from the new config")
        for name in model_names:
            weights = payload["models"][name]
            if any(not torch.is_tensor(value) or not torch.isfinite(value).all() for value in weights.values()):
                raise ValueError(f"Non-finite/incomplete {stage} {name} weights")
            observer.load_state_dict(weights, strict=True)
        status_path = source / stage / "training_status.json"
        status = json.loads(status_path.read_text(encoding="utf-8"))
        budget = int(original["train"].get(budget_key, original["train"]["max_steps"]))
        if (status.get("status") != "complete" or status.get("stage") != stage or
                status.get("step") != budget or status.get("max_steps") != budget or
                budget < int(config["train"][budget_key]) or not 0 < int(payload["global_step"]) <= budget):
            raise ValueError(f"Source {stage} has not completed the required training budget")
        if stage == "observation" and (not original["train"].get("require_a0_gate", True) or
                status.get("semantic_gate_required") is not True or status.get("semantic_gate_passed") is not True):
            raise ValueError("Source A0 must have completed its required semantic gate without bypass")
        if stage == "calibration" and any(not torch.equal(value, payload["models"]["teacher"][name])
                                         for name, value in payload["models"]["observer"].items()):
            raise ValueError("Calibration teacher is not the exported calibrated observer")
        records[stage] = {"source": str(path), "destination": str(destination / stage / "best.pt"),
                          "sha256": digest, "kind": kind, "global_step": int(payload["global_step"]),
                          "source_completed_steps": budget, "optimizer_steps_in_new_run": 0,
                          "source_training_status_sha256": _file_sha256(status_path)}
        del observer, payload
    gate_path = source / "observation/a0_semantic_gate.json"
    original_gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if (original_gate.get("passed") is not True or original_gate.get("allowed") is not True or
            original_gate.get("bypassed") is not False or original_gate.get("bypass_authorization") or
            original_gate.get("checkpoint_step") != records["observation"]["global_step"]):
        raise ValueError("Source A0 semantic gate failed, was bypassed, or refers to another checkpoint")
    receipt = {"protocol": "same-seed-observation-calibration-reuse-v1", "seed": seed,
               "source_seed_directory": str(source), "destination_seed_directory": str(destination),
               "checkpoints": records, "token_manifests": {root: value["sha256"] for root, value in expected.items()},
               "source_a0_gate_sha256": _file_sha256(gate_path),
               "upstream_training_performed_in_new_run": False,
               "downstream_initialization": {"dynamics": "new state model and optimizer; copied calibration observer",
                                             "generation": "new optimizer; this run's dynamics and generator baseline"},
               "source_optimizer_policy": "archived in copied exports; never restored"}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".upstream-reuse-", dir=destination.parent) as temporary:
        staged = Path(temporary) / "seed"
        for stage, record in records.items():
            target = staged / stage / "best.pt"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(record["source"], target)
            if _file_sha256(target) != record["sha256"] or _file_sha256(record["source"]) != record["sha256"]:
                raise ValueError("Upstream checkpoint changed during validation/copy")
        gate = validate_export(staged / "observation", allow_untrained=False)
        if not gate["passed"] or not gate["allowed"] or gate["bypassed"] or gate.get("bypass_authorization"):
            raise ValueError("Copied A0 failed the original semantic gate; downstream training prohibited")
        gate["checkpoint"] = records["observation"]["destination"]
        write_json(staged / "observation/a0_semantic_gate.json", gate)
        write_json(staged / "reused_upstream.json", receipt)
        if destination.exists():
            wanted = {path.relative_to(staged) for path in staged.rglob("*") if path.is_file()}
            existing = {path.relative_to(destination) for path in destination.rglob("*") if path.is_file()}
            if wanted != existing or any(_file_sha256(staged / path) != _file_sha256(destination / path) for path in wanted):
                raise ValueError("Refusing to overwrite different existing upstream destination files")
        else:
            staged.rename(destination)
    return receipt


def source_configs(root, artifacts, smoke=False, selected="all"):
    old = root / "artifacts/v2_retrain_20260907_gpu23"
    sources = {
        "emotiontalk": {"dataset": "emotiontalk", "input_root": str(old / "emotiontalk_v2"),
                        "audio_model": "/home/s21_yhr/lzh/Emotiontalk_work/models/chinese-hubert-base",
                        "text_model": "/home/s21_yhr/lzh/Emotiontalk_work/models/chinese-macbert-base"},
        "iemocap": {"dataset": "iemocap", "input_root": str(old / "iemocap_v2"), "fold": 5,
                    "audio_model": "facebook/wav2vec2-base-960h", "text_model": "roberta-base"},
        "dualtalk": {"dataset": "dualtalk", "input_root": str(root / "datasets/dualtalk"),
                     "timed_root": str(old / "generation/all/timed"),
                     "audio_model": "facebook/wav2vec2-base-960h", "text_model": "roberta-base"},
    }
    if selected != "all":
        sources = {name: value for name, value in sources.items() if name in (selected, "dualtalk")}
        for modality in ("audio", "text"):
            sources["dualtalk"][modality + "_model"] = sources[selected][modality + "_model"]
    for name in [key for key in sources if key != "dualtalk"]:
        mapping = {}
        for labels in sorted((Path(sources[name]["input_root"]) / "dialogues").glob("*/labels.json")):
            dialogue = labels.parent.name
            for item in json.loads(labels.read_text())["utterances"]:
                uid = item["utterance_id"]
                wav = (Path("/home/s21_yhr/lzh/Emotiontalk_work/raw") / item["audio"] if name == "emotiontalk" else
                       root / "datasets/iemocap/raw" / ("Session" + str(int(dialogue[3:5]))) / "sentences/wav" / dialogue / (uid + ".wav"))
                if not wav.is_file():
                    raise FileNotFoundError(wav)
                mapping[dialogue + "/" + uid] = str(wav)
        path = artifacts / (name + "_audio_manifest.json")
        write_json(path, mapping)
        sources[name]["audio_manifest"] = str(path)
    return sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--smoke", action="store_true", help="Two source dialogues per split and two optimizer steps")
    parser.add_argument("--pilot-only", action="store_true")
    parser.add_argument("--source", choices=("all", "emotiontalk", "iemocap"), default="all")
    parser.add_argument("--skip-preprocess", action="store_true")
    parser.add_argument("--reuse-artifacts", help="Existing version-checked token artifacts for a smoke rerun")
    parser.add_argument("--reuse-token-root", action="append", default=[], metavar="DATASET=PATH",
                        help="Version-checked feature caches for every source; all training starts fresh")
    parser.add_argument("--reuse-upstream", help="Existing seed directory with observation/calibration/dynamics best checkpoints")
    parser.add_argument("--reuse-observation-calibration", action="append", default=[], metavar="SEED=SEED_DIR",
                        help="Reuse completed same-seed gated A0/calibration; train fresh dynamics and generation")
    args = parser.parse_args()
    if args.reuse_token_root and args.reuse_artifacts:
        parser.error("Use either --reuse-artifacts or --reuse-token-root")
    if args.reuse_observation_calibration and (args.smoke or args.reuse_upstream):
        parser.error("Formal A0/calibration reuse cannot be combined with smoke or --reuse-upstream")
    seeds = [6666] if args.smoke or args.pilot_only else [6666, 6667, 6668]
    reused_upstream = resolve_reused_observation_calibration(args.reuse_observation_calibration, seeds)
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    # Direct `python scripts/...py` starts sys.path at scripts/, not the project.
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    if os.environ.get("CUDA_VISIBLE_DEVICES") != ",".join(GPU_UUIDS):
        raise RuntimeError("Only approved physical GPU 2 and 3 UUIDs may be visible")
    if Path(args.run_id).name != args.run_id or not args.run_id.startswith("v3_"):
        raise ValueError("Use a distinct v3_ run identifier")
    stamp = root / "runs" / args.run_id
    artifacts = Path(args.reuse_artifacts).resolve() if args.reuse_artifacts else root / "artifacts" / args.run_id
    if args.reuse_upstream and not args.smoke:
        raise ValueError("Reusing upstream this way is for code smoke tests; formal seeds retrain independently")
    if reused_upstream:
        if any(source == stamp or source in stamp.parents or stamp in source.parents for source in reused_upstream.values()):
            raise ValueError("The new run must be disjoint from every reused source seed directory")
        if any((stamp / name).exists() for name in ("experiment_protocol.json", "status.json", "training")):
            raise ValueError("Formal upstream reuse requires a new run; existing experiment files will not be overwritten")
    stamp.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1",
           "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
           "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
           "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    complete = []

    def status(stage, **extra):
        value = {"stage": stage, "updated": datetime.datetime.now().isoformat(),
                 "physical_gpus": [2, 3], "completed_stages": list(complete), **extra}
        write_json(stamp / "status.json", value)
        print(json.dumps(value, ensure_ascii=False), flush=True)

    def run(command, name, extra_env=None):
        status(name, command=command, log=str(stamp / (name + ".log")))
        with (stamp / (name + ".log")).open("a") as log:
            subprocess.run(command, env={**env, **(extra_env or {})}, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
        complete.append(name)

    def parallel_preprocess(configs):
        processes = []
        status("preprocessing", configs=[name for name, _ in configs])
        for (name, config), gpu in zip(configs, GPU_UUIDS):
            path = artifacts / "configs" / (name + ".json")
            write_json(path, config)
            log = (stamp / (name + ".log")).open("a")
            proc = subprocess.Popen([sys.executable, "-B", "-m", "emotion_ssm.preprocess.tokens_v3", "--config", str(path)],
                                    env={**env, "CUDA_VISIBLE_DEVICES": gpu}, stdout=log, stderr=subprocess.STDOUT)
            log.close()
            processes.append((name, proc))
        errors = [(name, code) for name, proc in processes if (code := proc.wait()) != 0]
        if errors:
            raise RuntimeError(f"Token preparation failed: {errors}")
        complete.extend(name for name, _ in processes)

    def stage(module, config, name):
        path = stamp / "configs" / (name + ".json")
        write_json(path, config)
        run([sys.executable, "-B", "-m", "torch.distributed.run", "--standalone", "--nproc_per_node", "2",
             "-m", module, "--config", str(path)], name)

    try:
        from emotion_ssm.config_v3 import default_config, FORMAT_VERSION
        metadata_artifacts = root / "artifacts" / args.run_id if reused_upstream else artifacts
        sources = source_configs(root, metadata_artifacts, args.smoke, args.source)
        roots = {name: artifacts / (name + "_tokens") for name in sources}
        if args.reuse_token_root:
            roots = resolve_reused_token_roots(args.reuse_token_root, sources)
        if not args.skip_preprocess and not args.reuse_artifacts and not args.reuse_token_root:
            configs = []
            for name in [key for key in sources if key != "dualtalk"]:
                cfg = {"sources": [sources[name]], "output_root": str(roots[name]), "device": "cuda:0"}
                if args.smoke:
                    cfg["max_dialogues"] = 2
                configs.append(("preprocess_" + name, cfg))
            parallel_preprocess(configs)
            configs, shards = [], []
            for rank in (0, 1):
                output = artifacts / ("dualtalk_tokens_shard" + str(rank))
                cfg = {"sources": [sources["dualtalk"]], "output_root": str(output), "device": "cuda:0",
                       "shard_rank": rank, "shard_world": 2}
                if args.smoke:
                    cfg["max_dialogues"] = 4
                configs.append(("preprocess_dualtalk_" + str(rank), cfg))
                shards.append(output)
            parallel_preprocess(configs)
            merge_token_shards(shards, roots["dualtalk"])
        if args.reuse_artifacts or args.reuse_token_root:
            validate_reused_artifacts(roots.values(), smoke=args.smoke)
        config = default_config()
        config["data"].update(token_roots=[str(roots[name]) for name in sources],
                              source_names=list(sources),
                              dualtalk_tokens=str(roots["dualtalk"]), dualtalk_raw=str(root / "datasets/dualtalk"),
                              streaming_source={"audio_model": sources["dualtalk"]["audio_model"],
                                                "text_model": sources["dualtalk"]["text_model"], "audio_dim": 768, "text_dim": 768})
        config["paths"]["baseline"] = str(root / "model/dualtalk_baseline.pth")
        if args.smoke:
            config["train"].update(max_steps=2, observation_steps=2, dynamics_steps=2, calibration_steps=2,
                                   validate_every=2, validation_max_dialogues=2, log_every=1,
                                   require_a0_gate=False)
        from emotion_ssm.data.packets_v3 import SUPERVISION_REVISION
        write_json(stamp / "experiment_protocol.json", {"format_version": FORMAT_VERSION, "token_protocol": config["protocol"],
                   "supervision_revision": SUPERVISION_REVISION,
                   "dynamics_revision": config["train"]["dynamics_revision"],
                   "learning_revision": config["train"]["learning_revision"],
                   "generation_revision": config["train"]["generation_revision"],
                   "masking": config["train"].get("masking"), "state_flow": config["state"],
                   "future_label_protocol": config["train"]["future_label_protocol"],
                    "token_roots": {name: str(path) for name, path in roots.items()},
                    "reused_observation_calibration": {str(seed): str(path) for seed, path in reused_upstream.items()},
                    "upstream_reuse_budget_accounting": "source completion recorded separately; zero new A0/calibration steps for reused seeds",
                   "physical_gpus": [2, 3], "pilot_steps": 2 if args.smoke else 1000,
                   "formal_steps": 30000, "global_new_blocks_per_step": 32, "seeds": [6666, 6667, 6668],
                   "selection": "validation generation_total", "test_for_selection": False,
                   "a0_gate": "A/AT/AVT semantics, class distribution and deterministic normalized affect against train-only constants",
                   "formal_gate": "A0 semantic gate followed by finite losses/gradients and representation diagnostics; generation superiority measured separately"})
        base_config = copy.deepcopy(config)
        for seed in seeds:
            config = copy.deepcopy(base_config)
            config["train"]["seed"] = seed
            upstream = stamp / "training" / f"seed{seed}"
            if args.reuse_upstream:
                initialization = Path(args.reuse_upstream).resolve()
                config["paths"]["observation_checkpoint"] = str(initialization / "calibration/best.pt")
                config["paths"]["dynamics_checkpoint"] = str(initialization / "dynamics/best.pt")
            else:
                if seed in reused_upstream:
                    receipt = reuse_observation_calibration(reused_upstream[seed], upstream, config)
                    complete.extend((f"seed{seed}_observation_reused", f"seed{seed}_calibration_reused"))
                    status(f"seed{seed}_upstream_reused", provenance=str(upstream / "reused_upstream.json"),
                           source_seed_directory=receipt["source_seed_directory"])
                else:
                    if reused_upstream and upstream.exists():
                        raise ValueError("A new non-reused seed must not overwrite an existing training directory")
                    config["paths"]["output"] = str(upstream / "observation")
                    stage("emotion_ssm.train.observation_v3", config, f"seed{seed}_observation")
                run([sys.executable, "-B", "scripts/validate_a0_v31.py", "--root", str(upstream / "observation"),
                     *(["--allow-untrained"] if args.smoke else [])], f"seed{seed}_a0_gate")
                if seed not in reused_upstream:
                    config["paths"]["observation_checkpoint"] = str(upstream / "observation/best.pt")
                    config["paths"]["output"] = str(upstream / "calibration")
                    calibrated = copy.deepcopy(config)
                    calibrated["train"]["stage"] = "calibration"
                    stage("emotion_ssm.train.observation_v3", calibrated, f"seed{seed}_calibration")
                config["paths"]["observation_checkpoint"] = str(upstream / "calibration/best.pt")
                config["paths"]["output"] = str(upstream / "dynamics")
                stage("emotion_ssm.train.dynamics_v3", config, f"seed{seed}_dynamics")
                config["paths"]["dynamics_checkpoint"] = str(upstream / "dynamics/best.pt")
            modes = [("pilot", 2 if args.smoke else 1000)] if seed == 6666 else []
            if not args.smoke and not args.pilot_only:
                modes.append(("formal", 30000))
            for mode, steps in modes:
                for variant in ("none", "affect", "self", "dyadic"):
                    experiment = copy.deepcopy(config)
                    experiment["train"].update(max_steps=steps, validation_max_dialogues=2 if args.smoke else (16 if mode == "pilot" else 0),
                                                validate_every=2 if args.smoke else (250 if mode == "pilot" else 1000))
                    experiment["generation"]["variant"] = variant
                    experiment["paths"]["output"] = str(upstream / mode / variant)
                    stage("emotion_ssm.train.generation_v3", experiment, f"seed{seed}_{mode}_{variant}")
                    if mode == "formal":
                        for split in ("test", "ood"):
                            run([sys.executable, "-B", "-m", "torch.distributed.run", "--standalone", "--nproc_per_node", "2",
                                 "-m", "emotion_ssm.evaluate_v3", "--checkpoint", str(upstream / mode / variant / "best.pt"),
                                 "--split", split, "--output", str(upstream / mode / variant / (split + ".json"))],
                                f"seed{seed}_{mode}_{variant}_{split}")
                if mode == "pilot":
                    run([sys.executable, "-B", "scripts/validate_v3_run.py", "--root", str(upstream),
                         *(["--allow-untrained"] if args.smoke else [])], "pilot_gate")
        status("completed", note="Training completed; scientific effects require paired test/OOD reports")
    except BaseException as error:
        status("failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
