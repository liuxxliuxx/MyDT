"""A new flow/masking experiment cannot silently inherit an old optimizer."""
import copy
import json

import pytest
import torch

from emotion_ssm.config_v3 import default_config, read_config, LEARNING_REVISION
from emotion_ssm.models.state_core import UnifiedEmotionStateCore
from emotion_ssm.models.token_observer import TokenObserver
from emotion_ssm.utils.checkpoint_v3 import load_observer, save_checkpoint, restore_training
from test_v31_pipeline_upstream_reuse import upstream, _pipeline, _edit_checkpoint


def test_old_saved_config_keeps_legacy_flow_and_mask_semantics(tmp_path):
    config = default_config()
    config["state"] = {"relation_dim": 3, "hidden_dim": 8}
    config["train"].update(dynamics_revision="v3.1.3-real-endpoint-forecast")
    for key in ("learning_revision", "masking", "generation_revision"):
        config["train"].pop(key)
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    loaded = read_config(path)
    assert loaded["state"] == config["state"]
    assert loaded["train"]["masking"] is None
    assert loaded["train"]["learning_revision"] != LEARNING_REVISION
    assert UnifiedEmotionStateCore(observation_dim=4, **loaded["state"]).adaptive_flow is None


def checkpoint(tmp_path):
    config = default_config()
    config["observer"].update(audio_dim=4, text_dim=4, affect_dim=4, model_dim=8,
                               num_layers=1, num_heads=2, dropout=0.)
    model = TokenObserver(config["observer"])
    optimizer = torch.optim.AdamW(model.parameters())
    sum(p.square().sum() for p in model.parameters()).backward()
    optimizer.step()
    path = tmp_path / "a0.pt"
    payload = save_checkpoint(path, {"observer": model}, config,
        {"observer": model.construction()}, "observation_v3", step=1, optimizer=optimizer)
    return config, model, optimizer, payload


def test_old_a0_weights_still_load_but_optimizer_cannot_resume(tmp_path):
    config, model, optimizer, payload = checkpoint(tmp_path)
    payload["config"]["train"].pop("learning_revision")
    payload["config"]["train"].pop("masking")
    restored, _ = load_observer(payload)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value, atol=0, rtol=0)
    with pytest.raises(ValueError, match="optimizer cannot resume"):
        restore_training(payload, {"observer": model}, optimizer, config=config)


@pytest.mark.parametrize("changed", ["masking", "state"])
def test_new_resume_is_exact_and_rejects_changed_training_semantics(tmp_path, changed):
    config, model, optimizer, payload = checkpoint(tmp_path)
    restored = TokenObserver(config["observer"])
    opt = torch.optim.AdamW(restored.parameters())
    restore_training(payload, {"observer": restored}, opt, config=config)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value, atol=0, rtol=0)
    assert opt.state_dict()["param_groups"] == optimizer.state_dict()["param_groups"]
    other = copy.deepcopy(config)
    if changed == "masking":
        other["train"]["masking"]["whole_modality_probability"] = .3
    else:
        other["state"]["flow_rank"] += 1
    with pytest.raises(ValueError, match="optimizer cannot resume"):
        restore_training(payload, {"observer": restored}, opt, config=other)


def test_new_mask_experiment_rejects_completed_old_upstream(upstream):
    source, config, destination = upstream
    _edit_checkpoint(source, "observation", lambda value: value["config"]["train"].update(masking=None))
    with pytest.raises(ValueError, match="retrain upstream"):
        _pipeline().reuse_observation_calibration(source, destination, config)
    assert not destination.exists()
