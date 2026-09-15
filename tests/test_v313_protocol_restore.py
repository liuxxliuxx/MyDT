"""Old exports retain their evaluation protocol; new losses need a fresh optimizer."""
import json

import pytest

from emotion_ssm.config_v3 import (
    DYNAMICS_REVISION, FUTURE_LABEL_PROTOCOL, LEGACY_FUTURE_LABEL_PROTOCOL,
    UNIT_LABEL_DYNAMICS_REVISIONS, default_config, read_config, validate_config,
)
from emotion_ssm.utils.checkpoint_v3 import require_training_revision


def test_read_saved_v312_config_does_not_assign_new_future_loss(tmp_path):
    config = default_config()
    config["train"]["dynamics_revision"] = "v3.1.2-state-memory-forecast"
    config["train"].pop("future_label_protocol")
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    loaded = read_config(path)
    assert loaded["train"]["future_label_protocol"] == LEGACY_FUTURE_LABEL_PROTOCOL
    assert loaded["train"]["dynamics_revision"] in UNIT_LABEL_DYNAMICS_REVISIONS
    assert json.loads(path.read_text()) == config


def test_new_revision_requires_real_endpoint_protocol():
    config = default_config()
    assert config["train"]["future_label_protocol"] == FUTURE_LABEL_PROTOCOL
    config["train"]["future_label_protocol"] = LEGACY_FUTURE_LABEL_PROTOCOL
    with pytest.raises(ValueError, match="real-endpoint"):
        validate_config(config)


def test_v312_optimizer_cannot_resume_v313_training():
    config = default_config()
    config["train"]["dynamics_revision"] = "v3.1.2-state-memory-forecast"
    with pytest.raises(ValueError, match="optimizer cannot resume"):
        require_training_revision({"kind": "dynamics_v3", "config": config})
    config["train"]["dynamics_revision"] = DYNAMICS_REVISION
    require_training_revision({"kind": "dynamics_v3", "config": config})
