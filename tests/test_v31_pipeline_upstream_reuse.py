"""Formal upstream reuse validates provenance/gates and starts fresh downstream."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from emotion_ssm import config_v3
from emotion_ssm.data.packets_v3 import SUPERVISION_REVISION
from emotion_ssm.models.token_observer import TokenObserver
from emotion_ssm.utils.checkpoint_v3 import save_checkpoint
from scripts.validate_a0_v31 import validate_export


def _pipeline():
    path = Path(__file__).resolve().parents[1] / "scripts/server1_v3_pipeline.py"
    spec = importlib.util.spec_from_file_location("pipeline_upstream_reuse_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _metrics():
    result = {"missing": False, "emotion_samples": 8, "samples": 8,
              "supported_classes": 2, "supported_predicted_classes": 2,
              "macro_f1": 0.8, "uar": 0.8, "constant_class": {"macro_f1": 0.3, "uar": 0.5},
              "embedding_variance_trace": 0.4, "vad_elements": 3, "vad_mse": 0.1,
              "vad_constant_train_mean_mse": 0.5}
    return {"training_statistics": {"split": "train", "domains": {
                "1": {"class_counts": [4, 4, 0, 0, 0, 0, 0], "vad_count": [1, 1, 1]}}},
            "evaluation_model": "student", "evaluation_split": "val", "selection_loss": 0.1,
            **{f"domain1/{subset}": copy.deepcopy(result) for subset in ("A", "AT", "AVT")}}


@pytest.fixture
def upstream(tmp_path):
    torch.set_num_threads(1)
    config = config_v3.default_config()
    config["observer"].update(audio_dim=8, text_dim=8, model_dim=8, affect_dim=4,
                               num_heads=2, num_layers=1, dropout=0.)
    roots = []
    for name in ("emotiontalk", "iemocap", "dualtalk"):
        root = tmp_path / "tokens" / name
        root.mkdir(parents=True)
        manifest = {"protocol": config_v3.PROTOCOL, "supervision_revision": SUPERVISION_REVISION,
                    "feature_sources": {name: {"audio_model": "local", "text_model": "local"}},
                    "splits": {key: [name + ":" + key] for key in ("train", "val", "test", "ood")},
                    "dialogues": {}, "errors": []}
        (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        roots.append(str(root))
    config["data"].update(token_roots=roots, dualtalk_tokens=roots[-1])
    source = tmp_path / "old_run/training/seed6666"
    observer = TokenObserver(config["observer"])
    optimizer = torch.optim.AdamW(observer.parameters(), lr=1e-4)
    for stage, kind, steps in (("observation", "observation_v3", 10000),
                                ("calibration", "calibration_v3", 2500)):
        original = copy.deepcopy(config)
        original["train"]["max_steps"] = steps
        original["paths"]["output"] = str(source / stage)
        if stage == "calibration":
            original["train"]["stage"] = "calibration"
            original["paths"]["observation_checkpoint"] = str(source / "observation/best.pt")
        metrics = _metrics()
        models = {"observer": observer, "teacher": copy.deepcopy(observer),
                  "ema" if stage == "observation" else "coordinate_teacher": copy.deepcopy(observer)}
        save_checkpoint(source / stage / "best.pt", models, original,
                        {"observer": observer.construction(), "exported_model": "student"}, kind,
                        step=steps // 2, optimizer=optimizer,
                        metrics={**metrics, "student": metrics, "selected_model": "student"},
                        run_state={"world_size": 2, "best_gate_passed": stage == "observation"})
        (source / stage / "training_status.json").write_text(json.dumps({
            "stage": stage, "status": "complete", "step": steps, "max_steps": steps,
            "semantic_gate_required": stage == "observation", "semantic_gate_passed": stage == "observation",
        }), encoding="utf-8")
    validate_export(source / "observation")
    return source, config, tmp_path / "new_run/training/seed6666"


def _edit_checkpoint(source, stage, edit):
    path = source / stage / "best.pt"
    value = torch.load(path, map_location="cpu", weights_only=False)
    edit(value)
    torch.save(value, path)


def test_reuse_copies_only_upstream_without_mutating_source_and_is_idempotent(upstream):
    source, config, destination = upstream
    pipeline = _pipeline()
    before = {path.relative_to(source): pipeline._file_sha256(path) for path in source.rglob("*") if path.is_file()}
    receipt = pipeline.reuse_observation_calibration(source, destination, config)
    assert receipt == pipeline.reuse_observation_calibration(source, destination, config)
    assert receipt["upstream_training_performed_in_new_run"] is False
    for stage in ("observation", "calibration"):
        old, new = source / stage / "best.pt", destination / stage / "best.pt"
        assert not os.path.samefile(old, new)
        assert pipeline._file_sha256(old) == pipeline._file_sha256(new)
        assert receipt["checkpoints"][stage]["destination"] == str(new.resolve())
        assert receipt["checkpoints"][stage]["optimizer_steps_in_new_run"] == 0
    assert not (destination / "dynamics").exists()
    assert not (destination / "observation/training_status.json").exists()
    gate = json.loads((destination / "observation/a0_semantic_gate.json").read_text())
    assert gate["passed"] and not gate["bypassed"] and gate["bypass_authorization"] is None
    assert gate["checkpoint"] == str((destination / "observation/best.pt").resolve())
    assert before == {path.relative_to(source): pipeline._file_sha256(path) for path in source.rglob("*") if path.is_file()}


@pytest.mark.parametrize("stage", ["observation", "calibration"])
def test_source_seed_mismatch_is_rejected(upstream, stage):
    source, config, destination = upstream
    _edit_checkpoint(source, stage, lambda value: value["config"]["train"].update(seed=6667))
    with pytest.raises(ValueError, match="train.seed"):
        _pipeline().reuse_observation_calibration(source, destination, config)
    assert not destination.exists()


@pytest.mark.parametrize("change", ["manifest_changed", "new_root", "saved_provenance"])
def test_actual_and_new_manifest_source_mismatches_are_rejected(upstream, change):
    source, config, destination = upstream
    if change == "manifest_changed":
        path = Path(config["data"]["token_roots"][0]) / "manifest.json"
        path.write_text(path.read_text() + "\n", encoding="utf-8")
    elif change == "new_root":
        old = Path(config["data"]["token_roots"][0])
        new = old.parent / "other_root"
        new.mkdir()
        (new / "manifest.json").write_bytes((old / "manifest.json").read_bytes())
        config["data"]["token_roots"][0] = str(new)
    else:
        _edit_checkpoint(source, "calibration", lambda value: value["provenance"].clear())
    with pytest.raises(ValueError, match="manifest provenance"):
        _pipeline().reuse_observation_calibration(source, destination, config)
    assert not destination.exists()


@pytest.mark.parametrize("field,value", [("format_version", 3), ("protocol", "legacy"), ("kind", "dynamics_v3")])
def test_wrong_checkpoint_format_protocol_or_stage_is_rejected(upstream, field, value):
    source, config, destination = upstream
    _edit_checkpoint(source, "calibration", lambda payload: payload.update({field: value}))
    with pytest.raises(ValueError, match="format-4|wrong-kind"):
        _pipeline().reuse_observation_calibration(source, destination, config)


def test_copy_is_rejected_when_actual_best_fails_gate_despite_old_pass_report(upstream):
    source, config, destination = upstream
    _edit_checkpoint(source, "observation", lambda payload:
                     payload["metrics"]["student"]["domain1/A"].update(supported_predicted_classes=1))
    with pytest.raises(ValueError, match="Copied A0 failed"):
        _pipeline().reuse_observation_calibration(source, destination, config)
    assert not destination.exists()


@pytest.mark.parametrize("change", ["bypass", "unfinished", "missing_model"])
def test_bypassed_incomplete_or_non_self_contained_upstream_is_rejected(upstream, change):
    source, config, destination = upstream
    if change == "bypass":
        path = source / "observation/a0_semantic_gate.json"
        gate = json.loads(path.read_text())
        gate.update(bypassed=True, bypass_authorization="explicit smoke")
        path.write_text(json.dumps(gate), encoding="utf-8")
    elif change == "unfinished":
        path = source / "calibration/training_status.json"
        status = json.loads(path.read_text())
        status["status"] = "running"
        path.write_text(json.dumps(status), encoding="utf-8")
    else:
        _edit_checkpoint(source, "calibration", lambda payload: payload["models"].pop("coordinate_teacher"))
    with pytest.raises(ValueError, match="bypassed|required training budget|self-contained"):
        _pipeline().reuse_observation_calibration(source, destination, config)
    assert not destination.exists()


def test_source_overlap_existing_files_and_old_dynamics_are_not_overwritten(upstream):
    source, config, destination = upstream
    pipeline = _pipeline()
    for target in (source, source / "nested", source.parent):
        with pytest.raises(ValueError, match="disjoint"):
            pipeline.reuse_observation_calibration(source, target, config)
    pipeline.reuse_observation_calibration(source, destination, config)
    target = destination / "calibration/best.pt"
    target.write_bytes(b"keep different destination")
    with pytest.raises(ValueError, match="overwrite different"):
        pipeline.reuse_observation_calibration(source, destination, config)
    assert target.read_bytes() == b"keep different destination"
    (destination / "dynamics").mkdir()
    with pytest.raises(ValueError, match="fresh dynamics"):
        pipeline.reuse_observation_calibration(source, destination, config)


@pytest.mark.parametrize("specifications", [["6666"], ["6666="], ["bad=path"], ["6669=path"],
                                            ["6666=one", "6666=two"]])
def test_repeated_option_rejects_malformed_duplicate_and_unselected_seeds(specifications):
    with pytest.raises(ValueError, match="unique selected SEED"):
        _pipeline().resolve_reused_observation_calibration(specifications, [6666, 6667, 6668])


def test_repeated_option_accepts_explicit_seed_subset(tmp_path):
    pipeline = _pipeline()
    assert pipeline.resolve_reused_observation_calibration(
        [f"6666={tmp_path / 'first'}", f"6668={tmp_path / 'second'}"], [6666, 6667, 6668]) == {
            6666: (tmp_path / "first").resolve(), 6668: (tmp_path / "second").resolve()}


@pytest.mark.parametrize("existing", ["experiment_protocol.json", "status.json"])
def test_new_option_refuses_existing_run_before_modifying_source_or_status(tmp_path, monkeypatch, existing):
    pipeline = _pipeline()
    (tmp_path / "scripts").mkdir()
    monkeypatch.setattr(pipeline, "__file__", str(tmp_path / "scripts/server1_v3_pipeline.py"))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", ",".join(pipeline.GPU_UUIDS))
    stamp = tmp_path / "runs/v3_existing"
    stamp.mkdir(parents=True)
    record = stamp / existing
    record.write_bytes(b"existing evidence")
    monkeypatch.setattr(sys, "argv", ["pipeline", "--run-id", "v3_existing",
                        "--reuse-observation-calibration", f"6666={tmp_path / 'source'}"])
    previous = Path.cwd()
    try:
        with pytest.raises(ValueError, match="existing experiment files"):
            pipeline.main()
    finally:
        os.chdir(previous)
    assert record.read_bytes() == b"existing evidence"
    assert not (tmp_path / "artifacts").exists()


def test_legacy_smoke_reuse_upstream_still_skips_all_upstream_training(tmp_path, monkeypatch):
    pipeline = _pipeline()
    (tmp_path / "scripts").mkdir()
    monkeypatch.setattr(pipeline, "__file__", str(tmp_path / "scripts/server1_v3_pipeline.py"))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", ",".join(pipeline.GPU_UUIDS))
    monkeypatch.setattr(pipeline, "source_configs", lambda *args: {
        name: {"audio_model": "local", "text_model": "local"}
        for name in ("emotiontalk", "iemocap", "dualtalk")})
    stages = []

    def fake_run(command, **kwargs):
        if "--config" in command:
            stages.append((command[command.index("--config") - 1], json.loads(Path(command[-1]).read_text())))
        assert "scripts/validate_a0_v31.py" not in command
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(pipeline.subprocess, "run", fake_run)
    source = tmp_path / "smoke_source"
    monkeypatch.setattr(sys, "argv", ["pipeline", "--run-id", "v3_smoke", "--smoke", "--skip-preprocess",
                                    "--reuse-upstream", str(source)])
    previous = Path.cwd()
    try:
        pipeline.main()
    finally:
        os.chdir(previous)
    assert len(stages) == 4
    assert [config["generation"]["variant"] for _, config in stages] == ["none", "affect", "self", "dyadic"]
    for module, config in stages:
        assert module == "emotion_ssm.train.generation_v3"
        assert config["train"]["max_steps"] == 2
        assert config["paths"]["observation_checkpoint"] == str(source / "calibration/best.pt")
        assert config["paths"]["dynamics_checkpoint"] == str(source / "dynamics/best.pt")


def test_one_reused_seed_keeps_other_seeds_and_all_four_generation_budgets(upstream, monkeypatch):
    source, config, destination = upstream
    pipeline = _pipeline()
    root = destination.parents[2]
    (root / "scripts").mkdir()
    monkeypatch.setattr(pipeline, "__file__", str(root / "scripts/server1_v3_pipeline.py"))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", ",".join(pipeline.GPU_UUIDS))
    monkeypatch.setattr(config_v3, "default_config", lambda: copy.deepcopy(config))
    names = ("emotiontalk", "iemocap", "dualtalk")
    monkeypatch.setattr(pipeline, "source_configs", lambda *args:
                        {name: {"audio_model": "local", "text_model": "local"} for name in names})
    commands, stages = [], []

    def fake_run(command, **kwargs):
        commands.append(command)
        if "--config" in command:
            stages.append((command[command.index("--config") - 1], json.loads(Path(command[-1]).read_text())))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(pipeline.subprocess, "run", fake_run)
    arguments = ["pipeline", "--run-id", "v3_new", "--reuse-observation-calibration", f"6666={source}"]
    for name, token_root in zip(names, config["data"]["token_roots"]):
        arguments.extend(["--reuse-token-root", f"{name}={token_root}"])
    monkeypatch.setattr(sys, "argv", arguments)
    previous = Path.cwd()
    try:
        pipeline.main()
    finally:
        os.chdir(previous)
    target = root / "runs/v3_new/training/seed6666"
    first_module, first = stages[0]
    assert first_module == "emotion_ssm.train.dynamics_v3"
    assert first["paths"]["observation_checkpoint"] == str(target / "calibration/best.pt")
    assert first["paths"]["output"] == str(target / "dynamics")
    assert first["paths"].get("resume", "") == ""
    assert first["paths"].get("dynamics_checkpoint", "") == ""
    assert len([cfg for module, cfg in stages if module == "emotion_ssm.train.observation_v3"]) == 4
    for seed in (6666, 6667, 6668):
        dynamics = [cfg for module, cfg in stages if module == "emotion_ssm.train.dynamics_v3" and cfg["train"]["seed"] == seed]
        assert len(dynamics) == 1 and dynamics[0]["train"]["dynamics_steps"] == 10000
        generated = [cfg for module, cfg in stages if module == "emotion_ssm.train.generation_v3" and cfg["train"]["seed"] == seed]
        assert len(generated) == (8 if seed == 6666 else 4)
        assert [cfg["generation"]["variant"] for cfg in generated[-4:]] == ["none", "affect", "self", "dyadic"]
        assert all(cfg["train"]["max_steps"] == 30000 for cfg in generated[-4:])
        assert all(cfg["train"]["global_chunks_per_step"] == 32 and not cfg["paths"].get("resume") for cfg in generated)
    assert all(cfg["train"]["max_steps"] == 1000 for _, cfg in stages[1:5])
    gate = next(command for command in commands if "scripts/validate_a0_v31.py" in command)
    assert "--allow-untrained" not in gate
    assert str(target / "observation") in gate
    protocol = json.loads((root / "runs/v3_new/experiment_protocol.json").read_text())
    assert protocol["physical_gpus"] == [2, 3]
    assert protocol["reused_observation_calibration"] == {"6666": str(source.resolve())}
