from pathlib import Path

import torch

from emotion_ssm.config import get_cfg_defaults
from emotion_ssm.losses import observation_losses
from emotion_ssm.models import (
    SUBSET_MASKS,
    AffectDecoder,
    DyadicEmotionSSM,
    ObservationEncoder,
    ObservationSupervisionHeads,
    build_ema_teacher,
)
from emotion_ssm.train.dynamics_core import DynamicsTrainingBundle, make_teacher_aff
from emotion_ssm.train.dualtalk import ConditionedTrainingSystem
from emotion_ssm.models.conditioned_dualtalk import BlendshapeAffectProjector
from emotion_ssm.schema import DyadicState
from emotion_ssm.utils.checkpoint import load_training_checkpoint, save_training_checkpoint


def make_small_config():
    cfg = get_cfg_defaults()
    cfg.defrost()
    cfg.MODEL.AUDIO_DIM = 768
    cfg.MODEL.FACE_DIM = 35
    cfg.MODEL.TEXT_DIM = 768
    cfg.MODEL.MODEL_DIM = 16
    cfg.MODEL.OBSERVATION_DIM = 8
    cfg.MODEL.STATE_DIM = 16
    cfg.MODEL.RELATION_DIM = 6
    cfg.MODEL.INFLUENCE_DIM = 8
    cfg.MODEL.INFLUENCE_CHANNELS = 4
    cfg.MODEL.NUM_LAYERS = 1
    cfg.MODEL.NUM_HEADS = 4
    cfg.MODEL.NUM_TIMESCALES = 4
    cfg.MODEL.TAU_MAX = 10.0
    cfg.DYNAMICS.HORIZONS = [1, 2, 4]
    cfg.COUNTERFACTUAL.TOP_K = 2
    cfg.freeze()
    return cfg


def flatten_first_turn(batch):
    return {
        name: batch[name][:, 0]
        for name in ("audio", "face", "text", "modality_mask", "reliability", "dataset_id")
    }


def test_a0_and_both_dynamics_stages_backward(sequence_batch):
    torch.manual_seed(9)
    cfg = make_small_config()
    encoder = ObservationEncoder.from_config(cfg)
    teacher = build_ema_teacher(encoder)
    heads = ObservationSupervisionHeads(8, 4, 2)
    utterance = flatten_first_turn(sequence_batch)
    output = encoder(utterance, SUBSET_MASKS)
    with torch.no_grad():
        target = teacher(utterance, SUBSET_MASKS[-1:])
    predictions = heads(output.aff, 0.5)
    batch_labels = {
        name: sequence_batch[name][:, 0]
        for name in ("emotion", "intensity", "vad", "vad_mask")
    }
    batch_labels.update(
        {
            "speaker": torch.tensor([0, 2, -1]),
            "dataset_id": utterance["dataset_id"],
            "modality_mask": utterance["modality_mask"],
        }
    )
    a0 = observation_losses(
        output, target, predictions, batch_labels, torch.ones(7), cfg
    )
    assert torch.isfinite(a0["total"])
    a0["total"].backward()

    dynamics_encoder = ObservationEncoder.from_config(cfg)
    dynamics_encoder.requires_grad_(False)
    state_model = DyadicEmotionSSM.from_config(cfg, num_speakers=4)
    bundle = DynamicsTrainingBundle(
        dynamics_encoder, state_model, AffectDecoder(8), cfg
    )
    teacher_aff = make_teacher_aff(teacher, sequence_batch)
    for enable_partner, counterfactual in ((False, False), (True, True)):
        bundle.zero_grad(set_to_none=True)
        losses = bundle(
            sequence_batch,
            teacher_aff,
            torch.ones(7),
            enable_partner,
            counterfactual,
        )
        assert all(torch.isfinite(value).all() for value in losses.values())
        losses["total"].backward()


def test_checkpoint_roundtrip(tmp_path: Path):
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    path = tmp_path / "checkpoint.pt"
    save_training_checkpoint(path, 2, 7, {"model": model}, optimizer=optimizer)
    expected = {name: value.clone() for name, value in model.state_dict().items()}
    with torch.no_grad():
        model.weight.zero_()
    checkpoint = load_training_checkpoint(path, {"model": model}, optimizer=optimizer)
    assert checkpoint["epoch"] == 2 and checkpoint["global_step"] == 7
    assert all(torch.equal(model.state_dict()[name], value) for name, value in expected.items())


def test_conditioned_dualtalk_stage_backward():
    class DummyConditioner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.affect = torch.nn.Linear(1, 8)

        def forward(self, target, partner, dt):
            pooled = target.mean(-1, keepdim=True)
            affect = self.affect(pooled)
            context = torch.cat([affect, affect, affect[:, :2]], dim=-1)
            state = DyadicState(
                z=torch.stack([affect, affect], dim=1),
                relation=affect[:, :2],
                speaker_ids=torch.full((len(target), 2), -1, dtype=torch.long),
            )
            return context, state, {"target_aff": affect, "partner_aff": affect}

    class DummyGenerator(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.output = torch.nn.Linear(3, 56)

        def forward(self, target, partner, partner_blendshape, context, enable_film=True):
            return self.output(partner_blendshape[:, :, :3])

    system = ConditionedTrainingSystem(
        DummyGenerator(), DummyConditioner(), BlendshapeAffectProjector(56, 8), 0.1
    )
    system.set_state_frozen(False)
    batch = {
        "target_audio": torch.randn(2, 32),
        "partner_audio": torch.randn(2, 32),
        "target_blendshape": torch.randn(2, 6, 56),
        "partner_blendshape": torch.randn(2, 6, 56),
        "dt": torch.ones(2),
    }
    losses = system(batch)
    assert all(torch.isfinite(value).all() for value in losses.values())
    losses["total"].backward()
