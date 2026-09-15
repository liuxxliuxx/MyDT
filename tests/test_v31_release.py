"""Revision and release boundaries for the repaired learning protocol."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from emotion_ssm.config_v3 import FORMAT_VERSION, PROTOCOL, default_config
from emotion_ssm.data.packets_v3 import SUPERVISION_REVISION
from emotion_ssm.utils.checkpoint_v3 import read_checkpoint


def pipeline_module():
    path = Path(__file__).resolve().parents[1] / "scripts/server1_v3_pipeline.py"
    spec = importlib.util.spec_from_file_location("v31_release_pipeline", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v30_optimizer_is_not_resumable_under_repaired_semantics():
    assert FORMAT_VERSION == 4
    assert PROTOCOL == "emotion-token-packets-v3.1"
    with pytest.raises(ValueError, match="legacy"):
        read_checkpoint({"format_version": 3, "protocol": "emotion-token-packets-v3",
                         "optimizer": {"state": {}}, "models": {}})


def test_formal_run_rejects_a_diagnostic_cache_even_with_matching_dimensions(tmp_path):
    pipeline = pipeline_module()
    root = tmp_path / "tokens"
    root.mkdir()
    path = root / "manifest.json"
    path.write_text(json.dumps({"protocol": PROTOCOL, "diagnostic_subset": True}), encoding="utf-8")
    pipeline.validate_reused_artifacts([root], smoke=True)
    with pytest.raises(ValueError, match="diagnostic subset"):
        pipeline.validate_reused_artifacts([root], smoke=False)
    path.write_text(json.dumps({"protocol": "emotion-token-packets-v3"}), encoding="utf-8")
    with pytest.raises(ValueError, match="Incompatible"):
        pipeline.validate_reused_artifacts([root], smoke=True)


def test_failed_a0_semantic_gate_prevents_calibration_and_generation(tmp_path, monkeypatch):
    pipeline = pipeline_module()
    (tmp_path / "scripts").mkdir()
    monkeypatch.setattr(pipeline, "__file__", str(tmp_path / "scripts/server1_v3_pipeline.py"))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", ",".join(pipeline.GPU_UUIDS))
    monkeypatch.setattr(pipeline, "source_configs", lambda *args: {
        name: {"audio_model": "local-audio", "text_model": "local-text"}
        for name in ("emotiontalk", "iemocap", "dualtalk")})
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        if "scripts/validate_a0_v31.py" in command:
            assert "--allow-untrained" not in command
            raise subprocess.CalledProcessError(2, command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(pipeline.subprocess, "run", run)
    monkeypatch.setattr(sys, "argv", ["pipeline", "--run-id", "v3_1_gate_test", "--skip-preprocess", "--pilot-only"])
    previous = Path.cwd()
    try:
        with pytest.raises(subprocess.CalledProcessError):
            pipeline.main()
    finally:
        os.chdir(previous)
    assert len(commands) == 2
    status = json.loads((tmp_path / "runs/v3_1_gate_test/status.json").read_text())
    assert status["stage"] == "failed"
    assert status["completed_stages"] == ["seed6666_observation"]


def test_repair_defaults_require_semantic_gate():
    config = default_config()
    assert config["train"]["require_a0_gate"]
    assert config["observer"]["token_protocol"] == PROTOCOL


def reused_manifests(tmp_path, iemocap_revision=SUPERVISION_REVISION):
    roots = {}
    for name in ("emotiontalk", "iemocap", "dualtalk"):
        root = tmp_path / (name+" cached tokens")
        root.mkdir()
        manifest = {"protocol": PROTOCOL, "feature_sources": {name: {"audio_model": "local-audio", "text_model": "local-text"}},
                    "splits": {split: [] for split in ("train", "val", "test", "ood")}, "dialogues": {}}
        if name == "iemocap" and iemocap_revision is not None:
            manifest["supervision_revision"] = iemocap_revision
        (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        roots[name] = root.resolve()
    return roots


def stub_pipeline_environment(pipeline, tmp_path, monkeypatch, commands):
    (tmp_path / "scripts").mkdir()
    monkeypatch.setattr(pipeline, "__file__", str(tmp_path / "scripts/server1_v3_pipeline.py"))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", ",".join(pipeline.GPU_UUIDS))
    def sources(root, artifacts, smoke, selected):
        names = ("emotiontalk", "iemocap", "dualtalk") if selected == "all" else (selected, "dualtalk")
        return {name: {"audio_model": "local-audio", "text_model": "local-text"} for name in names}
    monkeypatch.setattr(pipeline, "source_configs", sources)
    def run(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0)
    monkeypatch.setattr(pipeline.subprocess, "run", run)
    monkeypatch.setattr(pipeline.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("Explicit token reuse reran preprocessing"))


def test_explicit_token_roots_flow_through_fresh_full_training_without_changing_budget(tmp_path, monkeypatch):
    pipeline = pipeline_module()
    roots = reused_manifests(tmp_path)
    before = {name: (path / "manifest.json").read_bytes() for name, path in roots.items()}
    commands = []
    stub_pipeline_environment(pipeline, tmp_path, monkeypatch, commands)
    arguments = ["pipeline", "--run-id", "v3_explicit_cache_test"]
    for name in reversed(roots):
        arguments += ["--reuse-token-root", name+"="+str(roots[name])]
    monkeypatch.setattr(sys, "argv", arguments)
    previous = Path.cwd()
    try:
        pipeline.main()
    finally:
        os.chdir(previous)
    configs = [json.loads(Path(command[command.index("--config")+1]).read_text())
               for command in commands if "--config" in command]
    assert len(configs) == 25  # Three upstream stages per seed, 4 pilot, 12 formal.
    stage_roots = [str(path) for path in roots.values()]
    for cfg in configs:
        assert cfg["data"]["token_roots"] == stage_roots
        assert cfg["data"]["dualtalk_tokens"] == str(roots["dualtalk"])
        assert cfg["data"]["source_names"] == list(roots)
        assert cfg["train"]["global_chunks_per_step"] == 32
        assert cfg["paths"]["resume"] == ""
        seed_root = tmp_path / "runs/v3_explicit_cache_test/training" / ("seed"+str(cfg["train"]["seed"]))
        stage = Path(cfg["paths"]["output"]).name
        if stage == "observation":
            assert cfg["train"]["observation_steps"] == 10000
            assert cfg["paths"]["observation_checkpoint"] == ""
            assert cfg["paths"]["dynamics_checkpoint"] == ""
        elif stage == "calibration":
            assert cfg["train"]["calibration_steps"] == 2500
            assert cfg["paths"]["observation_checkpoint"] == str(seed_root / "observation/best.pt")
            assert cfg["paths"]["dynamics_checkpoint"] == ""
        elif stage == "dynamics":
            assert cfg["train"]["dynamics_steps"] == 10000
            assert cfg["paths"]["observation_checkpoint"] == str(seed_root / "calibration/best.pt")
            assert cfg["paths"]["dynamics_checkpoint"] == ""
        else:
            assert stage in ("none", "affect", "self", "dyadic")
            assert cfg["paths"]["dynamics_checkpoint"] == str(seed_root / "dynamics/best.pt")
            assert cfg["train"]["max_steps"] == (1000 if Path(cfg["paths"]["output"]).parent.name == "pilot" else 30000)
    for seed in (6666, 6667, 6668):
        assert sum(cfg["train"]["seed"] == seed and Path(cfg["paths"]["output"]).name == "observation" for cfg in configs) == 1
    for name, path in roots.items():
        assert (path / "manifest.json").read_bytes() == before[name]
    protocol = json.loads((tmp_path / "runs/v3_explicit_cache_test/experiment_protocol.json").read_text())
    assert protocol["token_roots"] == {name: str(path) for name, path in roots.items()}
    assert protocol["supervision_revision"] == SUPERVISION_REVISION
    assert protocol["formal_steps"] == 30000 and protocol["seeds"] == [6666, 6667, 6668]
    assert sum("scripts/validate_a0_v31.py" in command for command in commands) == 3
    assert all("--allow-untrained" not in command for command in commands)


@pytest.mark.parametrize("revision", [None, "endpoint-intensity-mask-old"])
def test_bad_iemocap_reuse_is_rejected_before_launching_any_training(tmp_path, monkeypatch, revision):
    pipeline = pipeline_module()
    roots = reused_manifests(tmp_path, iemocap_revision=revision)
    commands = []
    stub_pipeline_environment(pipeline, tmp_path, monkeypatch, commands)
    arguments = ["pipeline", "--run-id", "v3_bad_ie_cache_test", "--pilot-only"]
    for name, path in roots.items():
        arguments += ["--reuse-token-root", name+"="+str(path)]
    monkeypatch.setattr(sys, "argv", arguments)
    previous = Path.cwd()
    try:
        with pytest.raises(ValueError, match="IEMOCAP.*supervision_revision"):
            pipeline.main()
    finally:
        os.chdir(previous)
    assert not commands
    status = json.loads((tmp_path / "runs/v3_bad_ie_cache_test/status.json").read_text())
    assert status["stage"] == "failed" and status["completed_stages"] == []


def test_reuse_modes_are_mutually_exclusive_before_any_execution(monkeypatch):
    pipeline = pipeline_module()
    monkeypatch.setattr(sys, "argv", ["pipeline", "--run-id", "v3_bad_modes", "--reuse-artifacts", "somewhere",
                                     "--reuse-token-root", "dualtalk=elsewhere"])
    with pytest.raises(SystemExit) as caught:
        pipeline.main()
    assert caught.value.code == 2
