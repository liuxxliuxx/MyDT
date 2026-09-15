"""Recovery uses saved architecture/weights, with no external init files."""
import copy

import pytest
import torch
from transformers import Wav2Vec2Config

from emotion_ssm.config_v3 import default_config
from emotion_ssm.models.state_core import UnifiedEmotionStateCore
from emotion_ssm.utils.checkpoint_v3 import build_avatar, load_avatar, read_checkpoint, save_checkpoint


def tiny_config():
    config = default_config()
    config["observer"].update(audio_dim=8, text_dim=8, model_dim=8, affect_dim=4,
                               num_layers=1, num_heads=2, dropout=0.)
    config["generation"]["feature_dim"] = 8
    config["state"].update(observation_dim=4, relation_dim=2, hidden_dim=8)
    config["paths"].update(baseline="not-present-baseline.pt", dynamics_checkpoint="not-present-upstream.pt")
    audio = Wav2Vec2Config(hidden_size=8, num_hidden_layers=1, num_attention_heads=2,
                          intermediate_size=16, conv_dim=(8, 8, 8), conv_kernel=(10, 3, 3),
                          conv_stride=(5, 2, 2), num_conv_pos_embedding_groups=2,
                          num_conv_pos_embeddings=8, mask_time_prob=0., mask_feature_prob=0.,
                          hidden_dropout=0., attention_dropout=0., feat_proj_dropout=0.)
    construction = {"observer": config["observer"],
                    "state": UnifiedEmotionStateCore(**config["state"]).get_config(),
                    "generator_audio": audio.to_dict(), "features": None}
    return config, construction


def test_complete_factory_checkpoint_independent_of_external_files(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    config, construction = tiny_config()
    model = build_avatar(config, construction=construction, initialize=False).eval()
    path = tmp_path / "full.pt"
    save_checkpoint(path, {"system": model}, config, model.construction_info, "streaming_avatar_v3", step=9)
    from transformers import AutoModel
    monkeypatch.setattr(AutoModel, "from_pretrained", lambda *a, **kw: pytest.fail("External pretrained file used on restore"))
    recovered, restored_config, payload = load_avatar(path)
    recovered.eval()
    assert restored_config == config
    assert payload["global_step"] == 9
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, recovered.state_dict()[name], rtol=0, atol=0)
    with torch.no_grad():
        audio_a, audio_b = torch.randn(1, 16000), torch.randn(1, 16000)
        visual = torch.randn(1, 25, 56)
        context = torch.randn(1, 25, model.state_model.context_dim)
        expected = model.generator(audio_a, audio_b, visual, context)
        actual = recovered.generator(audio_a, audio_b, visual, context)
    torch.testing.assert_close(expected, actual, rtol=0, atol=0)


def test_legacy_optimizer_cannot_resume_v3():
    with pytest.raises(ValueError, match="legacy"):
        read_checkpoint({"format_version": 2, "models": {}})


def test_checkpoint_rejects_incompatible_protocol(tmp_path):
    config, construction = tiny_config()
    config["generation"]["chunk_frames"] = 200
    with pytest.raises(ValueError, match="25"):
        build_avatar(config, construction=construction, initialize=False)


def test_checkpoint_state_roundtrip_does_not_reset_teacher(tmp_path):
    config, construction = tiny_config()
    model = build_avatar(config, construction=construction, initialize=False)
    with torch.no_grad():
        next(model.observer.parameters()).fill_(.3)
        next(model.teacher.parameters()).fill_(.7)
    path = tmp_path / "full.pt"
    save_checkpoint(path, {"system": model}, config, model.construction_info, "streaming_avatar_v3")
    restored, _, _ = load_avatar(path)
    assert torch.all(next(restored.observer.parameters()) == .3)
    assert torch.all(next(restored.teacher.parameters()) == .7)
