"""Mixture probabilities, real missing modalities, and dependency leakage."""
import copy

import pytest
import torch

from emotion_ssm.config_v3 import default_config
from emotion_ssm.models.context_masking import GROUPS, sample_context_mask, sample_training_subset
from emotion_ssm.models.token_observer import TokenObserver
from emotion_ssm.train.observation_v3 import observation_objective


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def fixture(batch=8, tokens=16):
    torch.manual_seed(345)
    model = TokenObserver(dict(audio_dim=4, prosody_dim=4, text_dim=4, au_dim=4,
        flame_dim=4, model_dim=8, affect_dim=4, summary_dim=2, num_layers=1, num_heads=2, dropout=0.)).eval()
    features = {"now": torch.full((batch,), float(tokens)), "dt": torch.ones(batch),
        "domain_id": torch.arange(batch)%3, "text_roles": torch.zeros(batch,tokens,dtype=torch.long),
        "text_fresh_mask": torch.zeros(batch,tokens,dtype=torch.bool),
        "fresh_observation": torch.ones(batch,3,dtype=torch.bool)}
    for modes in GROUPS.values():
        for mode in modes:
            features[mode+"_tokens"] = torch.randn(batch,tokens,4)
            features[mode+"_mask"] = torch.ones(batch,tokens,dtype=torch.bool)
            features[mode+"_times"] = torch.arange(1.,tokens+1)[None].expand(batch,-1)
    return model, features


def policy(whole=.1):
    return {**default_config()["train"]["masking"], "whole_modality_probability": whole}


def test_span_masks_have_visible_context_and_mask_dependency_copies_together():
    model, features = fixture()
    before = copy.deepcopy(features)
    student, masks, diagnostic = sample_context_mask(model, features, policy(0.), torch.Generator().manual_seed(1))
    assert diagnostic["whole_modality_fraction"] == 0
    assert diagnostic["visible_rescue_fraction"] == 0
    for mode in masks:
        assert masks[mode].any()
        assert (student[mode+"_mask"] & ~masks[mode]).any(-1).all()
        torch.testing.assert_close(features[mode+"_tokens"], before[mode+"_tokens"], rtol=0, atol=0)
        torch.testing.assert_close(features[mode+"_mask"], before[mode+"_mask"], rtol=0, atol=0)
    assert torch.equal(masks["audio"], masks["prosody"])
    assert torch.equal(masks["au"], masks["flame"])
    for mode in ("audio", "au", "flame", "text"):
        assert ((masks[mode].sum(-1) >= 6) & (masks[mode].sum(-1) <= 11)).all()


def test_hidden_values_cannot_change_encoded_affect():
    model, features = fixture()
    for whole in (0.,1.):
        student, masks, _ = sample_context_mask(model, features, policy(whole), torch.Generator().manual_seed(55))
        other = copy.deepcopy(student)
        for mode, hidden in masks.items():
            other[mode+"_tokens"][hidden | ~student[mode+"_mask"]] = 900*torch.randn_like(other[mode+"_tokens"])[hidden | ~student[mode+"_mask"]]
        a = model.encode(student, corruption=masks)["observation"].aff
        b = model.encode(other, corruption=masks)["observation"].aff
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_visual_masks_follow_timestamps_with_different_stream_lengths():
    model, features = fixture(tokens=16)
    for key in ("au_tokens", "au_mask", "au_times"):
        features[key] = features[key][:, :8]
    _, masks, _ = sample_context_mask(model, features, policy(0.), torch.Generator().manual_seed(3))
    assert torch.equal(masks["au"], masks["flame"][:, :8])
    assert (masks["flame"][:, 8:].sum(-1) >= 3).all()
    assert (masks["flame"][:, 8:].sum(-1) <= 6).all()


def test_unknown_visual_identity_uses_explicit_missing_masks():
    model, features = fixture()
    for mode in ("au", "flame"):
        features[mode+"_mask"].zero_()
        features[mode+"_tokens"].fill_(float("nan"))
    student, masks, _ = sample_context_mask(model, features, policy(), torch.Generator().manual_seed(4))
    assert not student["au_mask"].any() and not student["flame_mask"].any()
    assert not masks["au"].any() and not masks["flame"].any()
    assert torch.isfinite(model.masked_loss(features, masking=policy())["total"])


def test_whole_modality_mask_is_actual_absence_and_never_removes_everything():
    model, features = fixture(batch=128)
    student, masks, diagnostic = sample_context_mask(model, features, policy(1.), torch.Generator().manual_seed(11))
    assert diagnostic["whole_modality_fraction"] == 1
    observed = torch.stack([torch.stack([student[m+"_mask"].any(-1) for m in members]).any(0)
                            for members in GROUPS.values()], -1)
    assert (observed.sum(-1)>=1).all() and (observed.sum(-1)<=2).all()
    assert {tuple(row.tolist()) for row in observed} == {(1,0,0),(0,1,0),(0,0,1),(1,1,0),(1,0,1),(0,1,1)}
    assert all(not mask.any() for mask in masks.values())
    assert torch.equal(student["audio_mask"], student["prosody_mask"])


def test_requested_mixture_and_all_seven_training_subsets_are_reproducible():
    model, features = fixture(batch=400, tokens=4)
    a, am, ad = sample_context_mask(model, features, policy(), torch.Generator().manual_seed(34))
    b, bm, bd = sample_context_mask(model, features, policy(), torch.Generator().manual_seed(34))
    assert .065 < float(ad["whole_modality_fraction"]) < .135
    for mode in am:
        assert torch.equal(am[mode], bm[mode]) and torch.equal(a[mode+"_mask"], b[mode+"_mask"])
    rng = torch.Generator().manual_seed(25)
    subsets = [sample_training_subset(policy(), rng) for _ in range(1200)]
    assert set(subsets)=={"A","V","T","AV","AT","VT","AVT"}
    assert .86 < subsets.count("AVT")/len(subsets) < .94


def test_short_or_missing_observations_never_become_mask_only_examples():
    model, features = fixture(batch=2,tokens=1)
    for mode in ("text","au","flame"):
        features[mode+"_mask"].zero_()
    for mode in ("audio","prosody"):
        features[mode+"_mask"][1].zero_()
    student, masks, diagnostic = sample_context_mask(model, features, policy(), torch.Generator().manual_seed(1))
    assert not masks["audio"].any() and not masks["prosody"].any()
    assert student["audio_mask"][0].all() and not student["audio_mask"][1].any()
    result = model.masked_loss(features, masking=policy())
    assert all(torch.isfinite(x) for x in result.values())
    result["total"].backward()
    assert model.affect_head[1].weight.grad is not None


def test_mixed_mask_keeps_clean_teacher_regularization_and_affect_bottleneck_gradients():
    model, features = fixture()
    teacher = copy.deepcopy(model).requires_grad_(False)
    clean = model.representation_loss(features)
    result = model.masked_loss(features, teacher=teacher, masking=policy())
    torch.testing.assert_close(result["representation_total"], clean["total"])
    result["total"].backward()
    assert model.affect_head[1].weight.grad.abs().sum() > 0
    assert model.fusion.layers[0].self_attn.in_proj_weight.grad.abs().sum() > 0
    assert model.affect_summary_heads["audio"][0].weight.grad.abs().sum() > 0
    assert all(x.grad is None for x in teacher.parameters())
    with pytest.raises(ValueError, match="do not stack"):
        model.masked_loss(features, subset="A", masking=policy())


def test_a0_objective_uses_mixed_mask_without_stacking_supervised_subset_dropout():
    model, features = fixture()
    result = observation_objective(model, {"features":features}, subset="A",
        teacher=copy.deepcopy(model).requires_grad_(False), weights={"masking":policy()})
    assert "masked_masking_whole_modality_fraction" in result
    assert torch.isfinite(result["total"])
    result["total"].backward()
