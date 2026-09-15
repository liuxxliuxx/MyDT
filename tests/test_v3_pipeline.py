"""Pipeline boundaries: stage budgets, source shards and embedded raw features."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import BertConfig, Wav2Vec2Config

from emotion_ssm.config_v3 import PROTOCOL, default_config
from emotion_ssm.data.packets_v3 import SUPERVISION_REVISION, TokenPacketDataset
from emotion_ssm.models.state_core import UnifiedEmotionStateCore
from emotion_ssm.models.token_observer import TokenObserver
from emotion_ssm.utils.checkpoint_v3 import build_avatar, load_avatar, save_checkpoint


def pipeline_module():
    path = Path(__file__).resolve().parents[1] / "scripts/server1_v3_pipeline.py"
    spec = importlib.util.spec_from_file_location("server1_v3_pipeline_contract", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def shard(root, name, split="train", source=None, supervision_revision=None):
    root.mkdir()
    payload = root / "dialogue.pt"
    torch.save({"protocol": PROTOCOL, "cache_id": name, "packets": []}, payload)
    value = {"protocol": PROTOCOL, "feature_sources": source or {"dualtalk": {"audio_model": "same"}},
             "splits": {key: [name] if key == split else [] for key in ("train", "val", "test", "ood")},
             "dialogues": {name: {"path": payload.name, "cache_id": name, "packets": 0}},
             "data_gaps": [{"dialogue": name, "text_missing": True}]}
    if supervision_revision is not None:
        value["supervision_revision"] = supervision_revision
    (root / "manifest.json").write_text(json.dumps(value), encoding="utf-8")
    return root


def test_merge_shards_resolves_payloads_without_copying_or_split_overlap(tmp_path):
    pipeline = pipeline_module()
    first = shard(tmp_path / "first", "dualtalk:a")
    second = shard(tmp_path / "second", "dualtalk:b", "val")
    destination = tmp_path / "merged"
    result = pipeline.merge_token_shards([first, second], destination)
    assert result["splits"]["train"] == ["dualtalk:a"]
    assert result["splits"]["val"] == ["dualtalk:b"]
    assert len(result["data_gaps"]) == 2
    assert not (destination / "dialogue.pt").exists()
    assert TokenPacketDataset(destination, "train")[0]["cache_id"] == "dualtalk:a"
    assert TokenPacketDataset(destination, "val")[0]["cache_id"] == "dualtalk:b"


def test_merge_rejects_duplicate_dialogues_and_incompatible_extractors(tmp_path):
    pipeline = pipeline_module()
    first = shard(tmp_path / "first", "same")
    repeated = shard(tmp_path / "repeat", "same", "val")
    different = shard(tmp_path / "different", "other", source={"dualtalk": {"audio_model": "changed"}})
    with pytest.raises(ValueError, match="Duplicate"):
        pipeline.merge_token_shards([first, repeated], tmp_path / "out1")
    with pytest.raises(ValueError, match="Feature sources"):
        pipeline.merge_token_shards([first, different], tmp_path / "out2")


@pytest.mark.parametrize("revision", [None, "endpoint-intensity-mask-old"])
def test_merge_rejects_iemocap_shard_without_repaired_supervision(tmp_path, revision):
    pipeline = pipeline_module()
    source = {"iemocap": {"audio_model": "same"}}
    cache = shard(tmp_path / "iemocap", "iemocap:a", source=source, supervision_revision=revision)
    destination = tmp_path / "merged"
    with pytest.raises(ValueError, match="IEMOCAP.*supervision_revision"):
        pipeline.merge_token_shards([cache], destination)
    assert not (destination / "manifest.json").exists()


def test_merge_propagates_repaired_iemocap_supervision_into_loadable_manifest(tmp_path):
    pipeline = pipeline_module()
    source = {"iemocap": {"audio_model": "same"}}
    first = shard(tmp_path / "first", "iemocap:a", source=source, supervision_revision=SUPERVISION_REVISION)
    second = shard(tmp_path / "second", "iemocap:b", "val", source, SUPERVISION_REVISION)
    destination = tmp_path / "merged"
    manifest = pipeline.merge_token_shards([first, second], destination)
    assert manifest["supervision_revision"] == SUPERVISION_REVISION
    assert json.loads((destination / "manifest.json").read_text())["supervision_revision"] == SUPERVISION_REVISION
    assert TokenPacketDataset(destination, "train")[0]["cache_id"] == "iemocap:a"
    assert TokenPacketDataset(destination, "val")[0]["cache_id"] == "iemocap:b"


@pytest.mark.parametrize("source", ["emotiontalk", "dualtalk"])
def test_merge_keeps_legacy_unaffected_sources_loadable(tmp_path, source):
    pipeline = pipeline_module()
    cache = shard(tmp_path / source, source+":a", source={source: {"audio_model": "same"}})
    destination = tmp_path / "merged"
    pipeline.merge_token_shards([cache], destination)
    assert TokenPacketDataset(destination, "train")[0]["cache_id"] == source+":a"


def test_explicit_reuse_roots_keep_selected_source_order_and_literal_path(tmp_path):
    pipeline = pipeline_module()
    sources = {name: {} for name in ("emotiontalk", "iemocap", "dualtalk")}
    expected = {name: (tmp_path / (name+"=cached tokens")).resolve() for name in sources}
    result = pipeline.resolve_reused_token_roots([name+"="+str(expected[name]) for name in reversed(sources)], sources)
    assert list(result) == list(sources)
    assert result == expected


@pytest.mark.parametrize("specifications", [
    [], ["emotiontalk=et", "iemocap=ie"],
    ["emotiontalk=et", "iemocap=ie", "dualtalk=dt", "dualtalk=again"],
    ["emotiontalk=et", "iemocap=ie", "dualtalk=dt", "unknown=other"],
    ["emotiontalk=et", "iemocap=ie", "dualtalk="],
    ["emotiontalk=et", "iemocap=ie", "dualtalk"],
])
def test_explicit_reuse_rejects_missing_repeated_unknown_and_malformed_sources(specifications):
    pipeline = pipeline_module()
    with pytest.raises(ValueError, match="every selected dataset|unique selected dataset=path"):
        pipeline.resolve_reused_token_roots(specifications, {name: {} for name in ("emotiontalk", "iemocap", "dualtalk")})


def test_explicit_reuse_supports_a_single_labelled_source_with_dualtalk(tmp_path):
    pipeline = pipeline_module()
    selected = {name: {} for name in ("iemocap", "dualtalk")}
    result = pipeline.resolve_reused_token_roots(["dualtalk="+str(tmp_path/"dt"), "iemocap="+str(tmp_path/"ie")], selected)
    assert list(result) == ["iemocap", "dualtalk"]
    with pytest.raises(ValueError, match="unique selected"):
        pipeline.resolve_reused_token_roots(["emotiontalk="+str(tmp_path/"et"), "dualtalk="+str(tmp_path/"dt")], selected)


def test_pipeline_stage_paths_and_pilot_budgets(tmp_path, monkeypatch):
    pipeline = pipeline_module()
    (tmp_path / "scripts").mkdir()
    monkeypatch.setattr(pipeline, "__file__", str(tmp_path / "scripts/server1_v3_pipeline.py"))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", ",".join(pipeline.GPU_UUIDS))
    monkeypatch.setattr(pipeline, "source_configs", lambda *args: {
        key: {"audio_model": "local-audio", "text_model": "local-text"}
        for key in ("emotiontalk", "iemocap", "dualtalk")})
    captured = []
    def fake_run(command, **kwargs):
        if "--config" in command:
            config = json.loads(Path(command[-1]).read_text(encoding="utf-8"))
            captured.append((command[command.index("--config")-1], config))
        return subprocess.CompletedProcess(command, 0)
    monkeypatch.setattr(pipeline.subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "argv", ["pipeline", "--run-id", "v3_test", "--skip-preprocess", "--pilot-only"])
    previous = Path.cwd()
    try:
        pipeline.main()
    finally:
        os.chdir(previous)
    assert len(captured) == 7
    observation, calibration, dynamics, *generators = [item[1] for item in captured]
    assert observation["train"]["observation_steps"] == 10000
    assert calibration["train"]["calibration_steps"] == 2500
    assert dynamics["train"]["dynamics_steps"] == 10000
    assert calibration["paths"]["observation_checkpoint"] == str(Path(observation["paths"]["output"]) / "best.pt")
    assert dynamics["paths"]["observation_checkpoint"] == str(Path(calibration["paths"]["output"]) / "best.pt")
    assert [cfg["generation"]["variant"] for cfg in generators] == ["none", "affect", "self", "dyadic"]
    for cfg in generators:
        assert cfg["train"]["max_steps"] == 1000
        assert cfg["train"]["global_chunks_per_step"] == 32
        assert cfg["paths"]["dynamics_checkpoint"] == str(Path(dynamics["paths"]["output"]) / "best.pt")


def test_complete_checkpoint_embeds_raw_extractor_not_only_observer(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    config = default_config()
    config["observer"].update(audio_dim=8, text_dim=8, model_dim=8, affect_dim=4,
                              num_heads=2, num_layers=1, dropout=0.)
    config["state"].update(observation_dim=4, relation_dim=2, hidden_dim=8)
    config["generation"]["feature_dim"] = 8
    audio = Wav2Vec2Config(hidden_size=8, num_hidden_layers=1, num_attention_heads=2,
                          intermediate_size=16, conv_dim=(8,8,8), conv_kernel=(10,3,3),
                          conv_stride=(5,2,2), num_conv_pos_embedding_groups=2,
                          num_conv_pos_embeddings=8, mask_time_prob=0., mask_feature_prob=0.,
                          hidden_dropout=0., attention_dropout=0., feat_proj_dropout=0.)
    text = BertConfig(hidden_size=8, num_hidden_layers=1, num_attention_heads=2,
                      intermediate_size=16, vocab_size=3)
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "[PAD]": 1, "hello": 2}, unk_token="[UNK]"))
    feature_source = {"audio_model": "deliberately-missing-audio", "text_model": "deliberately-missing-text",
                      "audio_dim": 8, "text_dim": 8}
    construction = {"observer": TokenObserver(config["observer"]).construction(),
                    "state": UnifiedEmotionStateCore(**config["state"]).get_config(),
                    "generator_audio": audio.to_dict(),
                    "features": {"source": feature_source, "backbone": {
                        "audio": audio.to_dict(), "text": text.to_dict(),
                        "tokenizer": tokenizer.to_str(), "special_tokens": {"unk_token": "[UNK]", "pad_token": "[PAD]"}}}}
    from transformers import AutoModel, AutoTokenizer
    def forbidden(*args, **kwargs):
        pytest.fail("Complete checkpoint accessed an external pretrained file")
    monkeypatch.setattr(AutoModel, "from_pretrained", forbidden)
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", forbidden)
    model = build_avatar(config, construction=construction, initialize=False).eval()
    path = tmp_path / "full.pt"
    save_checkpoint(path, {"system": model}, config, model.construction_info, "streaming_avatar_v3")
    restored, _, _ = load_avatar(path)
    waveform = torch.randn(1, 16000)
    before = model.features(waveform, [], 1., 0., "A")
    after = restored.features(waveform, [], 1., 0., "A")
    for key in before:
        if torch.is_tensor(before[key]):
            torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)
