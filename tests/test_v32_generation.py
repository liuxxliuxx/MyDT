"""The new flow and masking work together in all four generation controls."""
import copy

import pytest
import torch

from emotion_ssm.config_v3 import default_config
from emotion_ssm.models.state_core import UnifiedEmotionStateCore
from emotion_ssm.train.generation_v3 import JointAuxiliary, configure_generation_stage, train_segment
from test_v3_generation import small_model, packet


@pytest.mark.parametrize("variant", ["none", "affect", "self", "dyadic"])
def test_adaptive_generation_optimizer_step_and_mixed_auxiliary(variant):
    torch.set_num_threads(1)
    torch.manual_seed(16)
    model = small_model(variant)
    model.state_model = UnifiedEmotionStateCore(observation_dim=8, relation_dim=4, hidden_dim=16,
        **default_config()["state"])
    model.teacher = copy.deepcopy(model.observer).requires_grad_(False).eval()
    parameters = configure_generation_stage(model, frozen_teacher=model.teacher)
    settings = default_config()["train"]
    settings["forecast_seconds"] = [1., 2.]
    auxiliary = JointAuxiliary(model, settings)
    examples = [(packet(t, event=t == 1), torch.zeros(1,25,56), torch.ones(1,25,dtype=torch.bool))
                for t in (1,2,3)]
    auxiliary.prepare(examples, "cpu", input_count=2)
    optimizer = torch.optim.AdamW(parameters, lr=1e-3, foreach=False)
    _, result = train_segment(model, examples[:2], optimizer, auxiliary_loss=auxiliary, tbptt_steps=4)
    assert result["valid_frames"] == 50 and result["global_valid_blocks"] == 2
    assert result["generator_grad_norm"] > 0
    assert (result["observer_grad_norm"] > 0) == (variant != "none")
    assert (result["state_grad_norm"] > 0) == (variant in ("self", "dyadic"))
    assert all(p.grad is None for p in model.teacher.parameters())
    if variant == "dyadic":
        for name in ("left", "conditioner.0.weight", "message.weight"):
            gradient = dict(model.state_model.adaptive_flow.named_parameters())[name].grad
            assert gradient is not None and gradient.abs().sum() > 0, name
