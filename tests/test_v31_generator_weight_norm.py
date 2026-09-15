"""Old Wav2Vec2 positional weight-norm keys must load without random leftovers."""
import pytest
import torch

from emotion_ssm.utils.checkpoint_v3 import build_avatar, load_avatar, save_checkpoint
from test_v3_checkpoint import tiny_config


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def legacy_positional_keys(state):
    """Represent an actual old weight-normalized checkpoint, keeping its tensors."""
    migrated, renamed = {}, 0
    for name, value in state.items():
        old = name.replace(".parametrizations.weight.original0", ".weight_g")
        old = old.replace(".parametrizations.weight.original1", ".weight_v")
        renamed += name != old
        migrated[old] = value.clone()
    assert renamed == 4, "Expected g/v in both independent speaker Wav2Vec2 encoders"
    return migrated


def assert_same_state(first, second):
    assert first.keys() == second.keys()
    for name, value in first.items():
        torch.testing.assert_close(value, second[name], rtol=0, atol=0, msg=name)


def test_baseline_strict_loader_migrates_old_positional_weight_g_v_and_preserves_output():
    config, construction = tiny_config()
    original = build_avatar(config, construction=construction, initialize=False).eval()
    restored = build_avatar(config, construction=construction, initialize=False).eval()
    legacy = legacy_positional_keys(original.generator.baseline.state_dict())
    wrapped = {"model_state_dict": {"module.baseline."+key: value for key, value in legacy.items()}}
    result = restored.generator.load_baseline_state_dict(wrapped)
    assert not result.missing_keys and not result.unexpected_keys
    assert_same_state(original.generator.baseline.state_dict(), restored.generator.baseline.state_dict())
    # Compare the reconstructed effective convolution, not only factor tensors.
    for name in ("audio_encoder1", "audio_encoder2"):
        first = getattr(original.generator.baseline.joint_encoder, name).encoder.pos_conv_embed.conv
        second = getattr(restored.generator.baseline.joint_encoder, name).encoder.pos_conv_embed.conv
        torch.testing.assert_close(first.weight, second.weight, rtol=0, atol=0)
    with torch.no_grad():
        audio_a, audio_b, flame = torch.randn(1,16000), torch.randn(1,16000), torch.randn(1,25,56)
        expected = original.generator(audio_a, audio_b, flame, enable_film=False)
        actual = restored.generator(audio_a, audio_b, flame, enable_film=False)
    torch.testing.assert_close(expected, actual, rtol=0, atol=0)


def test_complete_checkpoint_strict_restore_migrates_nested_legacy_weight_norm(tmp_path):
    config, construction = tiny_config()
    original = build_avatar(config, construction=construction, initialize=False).eval()
    path = tmp_path/"full.pt"
    payload = save_checkpoint(path, {"system": original}, config, original.construction_info, "streaming_avatar_v3")
    payload["models"]["system"] = legacy_positional_keys(payload["models"]["system"])
    torch.save(payload, path)
    restored, _, _ = load_avatar(path)
    assert_same_state(original.state_dict(), restored.state_dict())


def test_missing_positional_factor_fails_strict_loading_instead_of_leaving_initial_weights():
    config, construction = tiny_config()
    model = build_avatar(config, construction=construction, initialize=False)
    state = legacy_positional_keys(model.generator.baseline.state_dict())
    state.pop(next(key for key in state if key.endswith(".weight_v")))
    with pytest.raises((RuntimeError, KeyError)):
        model.generator.load_baseline_state_dict(state)
