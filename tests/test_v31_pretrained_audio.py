import copy

import pytest
import torch
from torch import nn

from tests._pretrained_audio_audit import load_pretrained_audio, verify_weight_norm_loading


class NormModel(nn.Module):
    base_model_prefix="hubert"
    def __init__(self):
        super().__init__()
        self.conv=nn.utils.parametrizations.weight_norm(nn.Conv1d(4,4,3),dim=2)
        self.other=nn.Linear(4,4)


def legacy(model,prefix=""):
    return {prefix+key.replace("parametrizations.weight.original0","weight_g").replace(
            "parametrizations.weight.original1","weight_v"):value.clone() for key,value in model.state_dict().items()}


def info():
    return {"missing_keys":["conv.parametrizations.weight.original0","conv.parametrizations.weight.original1"],
            "unexpected_keys":["conv.weight_g","conv.weight_v"],"mismatched_keys":[],"error_msgs":[]}


def test_legacy_hook_is_verified_without_mutating_a_single_weight():
    source=NormModel();checkpoint=legacy(source)
    loaded=NormModel();loaded.load_state_dict(checkpoint,strict=True)
    before=copy.deepcopy(loaded.state_dict())
    audit=verify_weight_norm_loading(loaded,info(),checkpoint)
    assert len(audit["verified_legacy_aliases"])==2
    assert not audit["model_parameters_modified"]
    for key,value in loaded.state_dict().items():
        torch.testing.assert_close(value,before[key],atol=0,rtol=0)
    torch.testing.assert_close(loaded.conv.weight,source.conv.weight,atol=0,rtol=0)


def test_missing_unrelated_weights_wrong_shapes_and_true_random_weights_fail_closed():
    model=NormModel();checkpoint=legacy(model);before=copy.deepcopy(model.state_dict())
    bad_info=info();bad_info["missing_keys"].append("other.weight")
    with pytest.raises(ValueError,match="unsupported missing"):
        verify_weight_norm_loading(model,bad_info,checkpoint)
    wrong_shape=dict(checkpoint);wrong_shape["conv.weight_g"]=torch.zeros(1)
    with pytest.raises(ValueError,match="shape"):
        verify_weight_norm_loading(model,info(),wrong_shape)
    random_source=dict(checkpoint);random_source["conv.weight_v"]=torch.randn_like(checkpoint["conv.weight_v"])
    with pytest.raises(ValueError,match="exact source"):
        verify_weight_norm_loading(model,info(),random_source)
    for key,value in model.state_dict().items():
        torch.testing.assert_close(value,before[key],atol=0,rtol=0)


def test_only_exact_source_or_declared_base_prefix_is_accepted():
    model=NormModel()
    assert len(verify_weight_norm_loading(model,info(),legacy(model,"hubert."))["verified_legacy_aliases"])==2
    with pytest.raises(ValueError,match="Missing or ambiguous"):
        verify_weight_norm_loading(model,info(),legacy(model,"arbitrary."))
    with pytest.raises(ValueError,match="Missing or ambiguous"):
        verify_weight_norm_loading(model,info(),{**legacy(model),**legacy(model,"hubert.")})


def test_real_hubert_legacy_checkpoint_load_matches_features_across_seeds(tmp_path):
    from transformers import HubertConfig,HubertModel
    old=torch.get_num_threads();torch.set_num_threads(1)
    try:
        config=HubertConfig(hidden_size=8,num_hidden_layers=1,num_attention_heads=2,intermediate_size=16,
                            conv_dim=(8,8),conv_kernel=(10,8),conv_stride=(5,8),
                            num_conv_pos_embedding_groups=2,num_conv_pos_embeddings=8)
        source=HubertModel(config).eval()
        source.config.save_pretrained(tmp_path)
        torch.save(legacy(source),tmp_path/"pytorch_model.bin")
        wave=torch.randn(1,8000)
        with torch.no_grad():
            expected=source(wave).last_hidden_state
        for seed in (111,999):
            torch.manual_seed(seed)
            loaded=load_pretrained_audio(str(tmp_path),local_files_only=True).eval()
            with torch.no_grad():
                observed=loaded(wave).last_hidden_state
            torch.testing.assert_close(observed,expected,atol=0,rtol=0)
            assert not loaded.pretrained_loading_audit["model_parameters_modified"]
    finally:
        torch.set_num_threads(old)
