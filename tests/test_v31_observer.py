"""Mechanism regressions for clean affect learning, masking and word order."""
from __future__ import annotations

import copy
from datetime import timedelta
import inspect

import pytest
import torch

from emotion_ssm.models.token_observer import MODES, TokenObserver, TokenObserverConfig


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def observer(dropout=0., affect_dim=8):
    torch.manual_seed(3108)
    return TokenObserver(TokenObserverConfig(audio_dim=8, text_dim=8, au_dim=8, flame_dim=8,
                         model_dim=16, affect_dim=affect_dim, num_layers=1, num_heads=2,
                         summary_dim=8, dropout=dropout))


def features(batch=4, length=5, identical=False):
    generator = torch.Generator().manual_seed(74)
    result = {}
    for mode in MODES:
        value = torch.randn(1 if identical else batch, length, 8, generator=generator)
        result[mode + "_tokens"] = value.expand(batch, -1, -1).clone()
        result[mode + "_mask"] = torch.ones(batch, length, dtype=torch.bool)
        result[mode + "_times"] = torch.arange(length).float()[None].expand(batch, -1) * .5
    result.update(domain_id=torch.zeros(batch, dtype=torch.long), now=torch.full((batch,), length*.5),
                  text_roles=torch.zeros(batch, length, dtype=torch.long),
                  text_positions=torch.arange(length)[None].expand(batch, -1),
                  text_fresh_mask=torch.ones(batch, length, dtype=torch.bool),
                  fresh_observation=torch.ones(batch, 3, dtype=torch.bool),
                  action_present=torch.ones(batch, dtype=torch.bool), dt=torch.ones(batch))
    return result


def test_dropout_cannot_satisfy_clean_unit_affect_variance():
    model = observer(dropout=.6).train()
    batch = features(16, identical=True)
    noisy = model.encode(batch)["observation"].aff
    first = model.encode_clean(batch)["observation"].aff
    second = model.encode_clean(batch)["observation"].aff
    assert noisy.std(0, unbiased=False).mean() > .01
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert first.std(0, unbiased=False).mean() < 1e-7
    assert model.training and all(module.training for module in model.fusion.modules())
    loss = model.representation_loss(batch)
    assert loss["variance"] > .99
    assert loss["mean"] > .99 and loss["uniformity"] > .99
    assert loss["total"] > 1.1
    loss["total"].backward()
    assert model.affect_head[-1].weight.grad is not None


def test_raw_norm_inflation_does_not_reduce_representation_penalty():
    model = observer()
    batch = features(8)
    first = model.representation_loss(batch)
    with torch.no_grad():
        model.affect_head[-1].weight.mul_(100.)
        model.affect_head[-1].bias.mul_(100.)
    second = model.representation_loss(batch)
    for key in ("variance", "mean", "covariance", "uniformity", "total"):
        torch.testing.assert_close(first[key], second[key], rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("dimension", [8, 128])
def test_relative_variance_scale_penalizes_collapse_and_rewards_spread(dimension):
    model = observer(affect_dim=dimension)
    constant = torch.ones(2*dimension, dimension)
    # Positive and negative orthogonal axes give an exactly centred unit cloud.
    spread = torch.cat([torch.eye(dimension), -torch.eye(dimension)])
    bad = model.representation_from_affect(constant)
    good = model.representation_from_affect(spread)
    assert .99 < bad["variance"] <= 1.
    assert good["variance"] < 1e-5 and good["mean"] < 1e-6
    assert good["total"] < .03
    assert bad["total"] > 1.1


def test_domain_identity_alone_does_not_satisfy_within_domain_diversity():
    model = observer()
    values = torch.cat([torch.eye(8)[:1].expand(4, -1), -torch.eye(8)[:1].expand(4, -1)])
    result = model.representation_from_affect(values, domain_id=torch.tensor([0]*4 + [1]*4))
    assert result["domain0_variance"] > .99 and result["domain1_variance"] > .99
    assert result["domain0_samples"] == 4 and result["domain1_samples"] == 4


def test_affect_bottleneck_masked_summary_has_nonzero_affect_head_gradient():
    model = observer()
    batch = features(4)
    loss = model.masked_loss(batch)
    # Test the explicit summary task alone, without label or variance gradients.
    loss["affect_summary"].backward()
    gradient = model.affect_head[-1].weight.grad
    assert gradient is not None and gradient.abs().sum() > 1e-5
    assert model.adapters["audio"][0][0].weight.grad.abs().sum() > 1e-6
    assert model.reconstruct["audio"].weight.grad is None


def test_teacher_target_is_clean_stop_gradient_and_restores_training_mode():
    student = observer()
    teacher = observer(dropout=.7).train()
    batch = features(4)
    corruption = student.sample_corruption(batch, generator=torch.Generator().manual_seed(3))
    first = student.masked_loss(batch, corruption=corruption, teacher=teacher)
    teacher.eval()
    second = student.masked_loss(batch, corruption=corruption, teacher=teacher)
    torch.testing.assert_close(first["affect_distillation"], second["affect_distillation"], rtol=0, atol=0)
    teacher.train()
    student.masked_loss(batch, corruption=corruption, teacher=teacher)["affect_distillation"].backward()
    assert teacher.training and teacher.fusion.training
    assert all(parameter.grad is None for parameter in teacher.parameters())
    assert student.affect_head[-1].weight.grad.abs().sum() > 1e-6


@pytest.mark.parametrize("requested_mode", ["audio", "prosody"])
def test_masked_acoustic_atom_has_no_parallel_prosody_answer(requested_mode):
    model = observer().eval()
    batch = features(3)
    corruption = {requested_mode: torch.zeros_like(batch[requested_mode + "_mask"])}
    corruption[requested_mode][:, 1:4] = True
    changed = copy.deepcopy(batch)
    changed["audio_tokens"][:, 1:4] += 100
    changed["prosody_tokens"][:, 1:4] -= 100
    first = model.encode(batch, corruption=corruption)
    second = model.encode(changed, corruption=corruption)
    assert first["masked"]["audio"][:, 1:4].all()
    assert first["masked"]["prosody"][:, 1:4].all()
    torch.testing.assert_close(first["observation"].aff, second["observation"].aff, rtol=0, atol=0)
    torch.testing.assert_close(first["reconstruction"]["audio"], second["reconstruction"]["audio"], rtol=0, atol=0)


def test_audio_prosody_mask_ratio_is_sampled_once_per_atom():
    model = observer()
    batch = features(12, length=10)
    chosen = model.sample_corruption(batch, .4, torch.Generator().manual_seed(5))
    assert chosen["audio"].sum(1).eq(4).all()
    torch.testing.assert_close(chosen["audio"], chosen["prosody"])
    batch["prosody_times"] = batch["prosody_times"] + .1
    with pytest.raises(ValueError, match="timestamps"):
        model.encode(batch)


def test_prosody_retains_energy_information_and_respects_true_missing_mask():
    model = observer().eval()
    batch = features(2, identical=True)
    batch["prosody_tokens"][1, :, 0] += 3.
    result = model.encode_clean(batch, "A")["observation"].aff
    assert (result[0] - result[1]).abs().max() > 1e-4
    batch["prosody_mask"].zero_()
    result = model.encode_clean(batch, "A")["observation"].aff
    torch.testing.assert_close(result[0], result[1], rtol=0, atol=0)


def test_text_order_matters_when_availability_and_roles_are_identical():
    model = observer().eval()
    batch = features(2, length=7, identical=True)
    batch["text_times"].zero_()
    batch["text_tokens"][1] = batch["text_tokens"][1].flip(0)
    value = model.encode_clean(batch, "T")["observation"].aff
    assert (value[0] - value[1]).abs().max() > 1e-4
    # Batch storage order is unrelated to word order and must remain equivariant.
    reverse = {name: tensor.flip(0) for name, tensor in batch.items()}
    torch.testing.assert_close(model.encode_clean(reverse, "T")["observation"].aff, value.flip(0))


def test_missing_summary_targets_and_invalid_rows_do_not_fabricate_supervision():
    model = observer()
    batch = features(3)
    for mode in ("au", "flame", "text"):
        batch[mode + "_mask"].zero_()
    result = model.masked_loss(batch)
    for mode in ("au", "flame", "text"):
        assert result["summary_" + mode].item() == 0.
    for mode in MODES:
        batch[mode + "_mask"].zero_()
    result = model.masked_loss(batch)
    assert result["samples"].item() == 0 and result["total"].item() == 0
    result["total"].backward()
    values = torch.randn(3, 8, requires_grad=True)
    with torch.no_grad():
        values[1].fill_(float("nan"))
    result = model.representation_from_affect(values, torch.tensor([True, False, True]))
    assert result["samples"].item() == 2 and torch.isfinite(result["total"])
    result["total"].backward()
    assert torch.isfinite(values.grad).all()
    assert values.grad[1].count_nonzero() == 0


def test_small_per_packet_loss_has_no_implicit_distributed_collective(monkeypatch):
    model = observer()
    assert inspect.signature(model.masked_loss).parameters["distributed"].default is False
    assert inspect.signature(model.representation_from_affect).parameters["distributed"].default is False
    def forbidden(*args, **kwargs):
        raise AssertionError("Per-packet losses must never gather implicitly")
    import torch.distributed as dist
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "all_gather", forbidden)
    model.masked_loss(features(1))["total"].backward()


def test_clean_affect_keeps_float32_under_cpu_bfloat16():
    model = observer()
    # Some local CPUs cannot execute oneDNN BF16 linear backward. The native
    # fallback still exercises autocast and the representation precision path.
    with torch.backends.mkldnn.flags(enabled=False):
        with torch.autocast("cpu", dtype=torch.bfloat16):
            result = model.encode_clean(features(4))
            loss = model.representation_loss(features(4))["total"]
        loss.backward()
    assert result["raw_aff"].dtype == torch.bfloat16
    assert result["observation"].aff.dtype == torch.float32
    assert loss.dtype == torch.float32 and torch.isfinite(loss)
    assert torch.isfinite(model.affect_head[-1].weight.grad).all()


def test_diagnostics_use_clean_views_even_during_training():
    model = observer(dropout=.8).train()
    first, second = model.diagnostics(features(8, identical=True)), model.diagnostics(features(8, identical=True))
    assert first == second
    assert all(item["std"] < 1e-7 for item in first.values())
    assert model.training and model.fusion.training


def test_dropped_current_modality_cannot_invent_an_action_from_query_bias():
    model = observer().eval()
    batch = features(2)
    for mode in MODES:
        batch[mode + "_mask"].zero_()
    result = model.encode_clean(batch)["observation"]
    assert not result.action_present.any()
    assert result.action.count_nonzero() == 0
    assert result.action_duration.count_nonzero() == 0


def test_protocol_and_new_complete_state_are_explicit():
    with pytest.raises(ValueError, match="protocol"):
        TokenObserver({"token_protocol": "emotion-token-packets-v3"})
    first = observer().eval()
    second = TokenObserver(first.construction()).eval()
    second.load_state_dict(first.state_dict(), strict=True)
    batch = features(3)
    torch.testing.assert_close(first.encode_clean(batch)["observation"].aff,
                               second.encode_clean(batch)["observation"].aff, rtol=0, atol=0)
    assert "summary_projection_prosody" in first.state_dict()
    old = dict(first.state_dict())
    old["mode_embedding.weight"] = old["mode_embedding.weight"][:4]
    with pytest.raises(RuntimeError, match="size mismatch"):
        second.load_state_dict(old, strict=True)


def _distributed_representation_worker(rank, rendezvous, output):
    """Every case reaches backward even when one/all ranks have no valid rows."""
    import torch.distributed as dist
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=30))
    model = observer()
    rows = [torch.tensor([[1., 2., 0., 0., 0., 0., 0., 0.]]),
            torch.tensor([[0., 1., 2., 0., 0., 0., 0., 0.],
                          [0., 0., 1., 2., 0., 0., 0., 0.],
                          [0., 0., 0., 1., 2., 0., 0., 0.]])]
    validity = [([False], [True, True, True]),
                ([False], [False, False, False]),
                ([True], [False, False, False]),
                ([True], [True, True, True])]
    results = []
    for first, second in validity:
        values = rows[rank].clone().requires_grad_()
        result = model.representation_from_affect(values, torch.tensor((first, second)[rank]), distributed=True)
        # Distributed gather backward sums all ranks' equivalent objectives.
        # Dividing by world size here gives the single-process reference gradient.
        (result["total"] / 2).backward()
        reference = torch.cat(rows).requires_grad_()
        expected = model.representation_from_affect(reference, torch.tensor(first + second))
        expected["total"].backward()
        start = 0 if rank == 0 else len(rows[0])
        torch.testing.assert_close(result["total"], expected["total"], rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(values.grad, reference.grad[start:start+len(values)], rtol=1e-5, atol=1e-6)
        results.append({"samples": int(result["samples"]), "loss": float(result["total"].detach())})
    torch.save(results, output + str(rank) + ".pt")
    dist.destroy_process_group()


@pytest.mark.skipif(not torch.distributed.is_available() or not torch.distributed.is_gloo_available(), reason="Gloo unavailable")
def test_distributed_clean_statistics_masking_gradients_and_empty_rank(tmp_path):
    import torch.multiprocessing as mp
    rendezvous = (tmp_path / "representation_gloo").as_uri()
    output = str(tmp_path / "rank")
    mp.spawn(_distributed_representation_worker, args=(rendezvous, output), nprocs=2, join=True)
    first = torch.load(output + "0.pt", weights_only=False)
    second = torch.load(output + "1.pt", weights_only=False)
    assert first == second
    assert [row["samples"] for row in first] == [3, 0, 1, 4]
