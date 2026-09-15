"""Offline CPU contracts for the complete v2 pipeline (no downloads)."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from transformers import BertConfig, Wav2Vec2Config, Wav2Vec2Model
from tokenizers import Tokenizer, models, pre_tokenizers

from emotion_ssm.config import get_cfg_defaults
from emotion_ssm.data.protocol import FeatureSource, make_split_manifest, require_compatible, select_adapter_source
from emotion_ssm.losses import observation_losses
from emotion_ssm.models import ObservationEncoder, ObservationSupervisionHeads, AffectDecoder, SUBSET_MASKS
from emotion_ssm.models.dynamics import DyadicEmotionSSM
from emotion_ssm.models.streaming import StreamSessions, normalize_audio
from emotion_ssm.preprocess.ctc_align import ctc_viterbi
from emotion_ssm.preprocess.emotion_features import causal_texts
from emotion_ssm.preprocess.iemocap import slice_face_frames
from emotion_ssm.schema import EventObservation
from emotion_ssm.train.calibrate_dualtalk import calibration_pair, calibration_loss
from emotion_ssm.train.dynamics_core import DynamicsTrainingBundle, make_teacher_aff
from emotion_ssm.utils.audio import pooled_audio
from emotion_ssm.utils.generation_checkpoint import build_streaming, save_generation, load_generation
from emotion_ssm.utils.reconstruction import ReconstructionTotals, reconstruction_loss


def tiny_cfg():
    cfg = get_cfg_defaults()
    cfg.DEVICE = "cpu"
    cfg.MODEL.AUDIO_DIM = cfg.MODEL.TEXT_DIM = cfg.MODEL.MODEL_DIM = 16
    cfg.MODEL.OBSERVATION_DIM = cfg.MODEL.STATE_DIM = cfg.MODEL.INFLUENCE_DIM = 8
    cfg.MODEL.RELATION_DIM = 4
    cfg.MODEL.INFLUENCE_CHANNELS = cfg.MODEL.NUM_TIMESCALES = 2
    cfg.MODEL.NUM_LAYERS = cfg.MODEL.AU_NUM_LAYERS = 1
    cfg.MODEL.DROPOUT = 0.
    cfg.DUALTALK.FEATURE_DIM = 8
    cfg.DYNAMICS.HORIZONS = [1, 2]
    cfg.DYNAMICS.BPTT_EVENTS = 2
    cfg.TRAIN.FINETUNE_OBSERVATION = True
    cfg.TRAIN.AMP = False
    cfg.TRAIN.MAX_STEPS = 2
    cfg.TRAIN.GLOBAL_CHUNKS_PER_STEP = 2
    cfg.TRAIN.VAL_EVERY_STEPS = 1
    return cfg


def acoustic_config(norm="group"):
    return Wav2Vec2Config(hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
        intermediate_size=32, conv_dim=[8, 8, 8], conv_kernel=[10, 3, 3], conv_stride=[5, 4, 4],
        num_conv_pos_embeddings=8, num_conv_pos_embedding_groups=2, feat_extract_norm=norm,
        mask_time_prob=0., mask_feature_prob=0., hidden_dropout=0., attention_dropout=0.,
        feat_proj_dropout=0., final_dropout=0., layerdrop=0.)


def tiny_construction():
    tokenizer = Tokenizer(models.WordLevel({"[PAD]": 0, "[UNK]": 1, "hello": 2, "sad": 3,
        "avatar": 4, "user": 5, "[": 6, "]": 7}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    audio = acoustic_config().to_dict()
    text = BertConfig(vocab_size=8, hidden_size=16, num_hidden_layers=1,
        num_attention_heads=2, intermediate_size=32, hidden_dropout_prob=0., attention_probs_dropout_prob=0.).to_dict()
    source = FeatureSource("missing/audio", "missing/text", audio_dim=16, text_dim=16).to_dict()
    return {"adapter_source": 1, "feature_source": source, "generator_audio": audio,
        "features": {"audio": audio, "text": text, "tokenizer": tokenizer.to_str(),
                     "special_tokens": {"pad_token": "[PAD]", "unk_token": "[UNK]"}}}


@pytest.fixture
def system():
    torch.manual_seed(17)
    model = build_streaming(tiny_cfg(), construction=tiny_construction(), initialize=False)
    with torch.no_grad():
        model.generator.film.affine.weight.normal_(std=.1)
    return model.eval()


def packet(time=1., frames=25, session="one"):
    return {"session_id": session, "roles": ("avatar", "user"), "time": time,
            "target_audio": torch.randn(1, frames*640), "partner_audio": torch.randn(1, frames*640),
            "partner_blendshape": torch.randn(1, frames, 56),
            "partner_visual_mask": torch.ones(1, frames, dtype=torch.bool)}


def test_stream_prefix_future_target_and_late_text(system):
    first, future = packet(), packet(2.)
    future["words"] = [{"id": "w", "role": "user", "text": "sad", "start": .1, "end": .3, "available_at": 1.6}]
    with torch.no_grad():
        before, state, diagnostics = system(first)
        snapshot = before.clone()
        second, _, _ = system(future, state)
        # Replaying the prefix never reads future audio, visual, or text.
        changed = copy.deepcopy(first)
        changed["words"] = future["words"]
        changed["target_flame"] = torch.randn(1, 25, 56) * 1000
        replay, replay_state, observed = system(changed)
        assert torch.equal(before, replay)
        assert torch.equal(diagnostics["context"], observed["context"])
        assert torch.equal(snapshot, before)
        assert not replay_state.words
        late = copy.deepcopy(future)
        late["words"][0]["available_at"] = .4  # actual delivery is now, not .4
        _, late_state, late_diag = system(late, state)
        assert late_state.words["w"]["available_at"] == 2.
        assert not torch.equal(diagnostics["partner_aff"], late_diag["partner_aff"])
        assert second.shape == (1, 25, 56)


def test_missing_audio_visual_payload_is_ignored(system):
    value = packet()
    value.update(target_audio_length=0, partner_audio_length=0,
                 partner_visual_mask=torch.zeros(1, 25, dtype=torch.bool))
    with torch.no_grad():
        expected, _, diagnostics = system(value)
        value["target_audio"].fill_(float("nan"))
        value["partner_audio"].fill_(10000.)
        value["partner_blendshape"].fill_(float("nan"))
        actual, _, observed = system(value)
    assert torch.equal(expected, actual)
    assert not diagnostics["target_modalities"].any()
    assert not observed["partner_modalities"].any()


def test_resume_preserves_saved_paths_until_explicitly_relocated(tmp_path, monkeypatch):
    import sys
    from emotion_ssm.config import parse_config_args
    from emotion_ssm.utils.emotion_checkpoint import deployment_paths
    saved = tiny_cfg()
    saved.DATA.DUALTALK_ROOT = "original/raw"
    saved.DUALTALK.TIMED_FEATURE_ROOT = "original/timed"
    saved.TRAIN.OUTPUT_ROOT = "original/runs"
    requested = tiny_cfg()
    requested.TRAIN.RESUME = "model.pt"
    resumed = deployment_paths(saved, requested)
    assert resumed.DATA.DUALTALK_ROOT == saved.DATA.DUALTALK_ROOT
    assert resumed.DUALTALK.TIMED_FEATURE_ROOT == saved.DUALTALK.TIMED_FEATURE_ROOT
    assert resumed.TRAIN.OUTPUT_ROOT == saved.TRAIN.OUTPUT_ROOT
    config = tmp_path/"config.yaml"
    config.write_text("DEVICE: cpu\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["train", "--config", str(config), "--resume", "model.pt",
                                   "DATA.DUALTALK_ROOT", "relocated/raw", "SEED", "123"])
    requested, _ = parse_config_args("test")
    resumed = deployment_paths(saved, requested)
    assert resumed.DATA.DUALTALK_ROOT == "relocated/raw"
    assert resumed.SEED == saved.SEED
    assert resumed.DUALTALK.TIMED_FEATURE_ROOT == saved.DUALTALK.TIMED_FEATURE_ROOT


def test_openface_rebuild_runs_extractor_even_when_csv_exists(tmp_path, monkeypatch):
    from emotion_ssm.preprocess import iemocap
    csv = tmp_path/"video.csv"
    csv.write_text("old output", encoding="utf-8")
    calls = []
    monkeypatch.setattr(iemocap.subprocess, "run", lambda *a, **k: calls.append((a, k)))
    assert iemocap.run_openface("FeatureExtraction", Path("video.avi"), tmp_path) == csv
    assert not calls
    assert iemocap.run_openface("FeatureExtraction", Path("video.avi"), tmp_path, rebuild=True) == csv
    assert len(calls) == 1


def test_local_model_revision_and_resume_data_reject_changed_content(tmp_path):
    from types import SimpleNamespace
    from emotion_ssm.data.protocol import model_revision
    from emotion_ssm.utils.checkpoint import data_provenance, validate_resume_data
    model = tmp_path/"model"
    model.mkdir()
    weights = model/"model.safetensors"
    weights.write_bytes(b"initial weights")
    first = model_revision(model, SimpleNamespace())
    weights.write_bytes(b"changed weights")
    assert model_revision(model, SimpleNamespace()) != first
    cfg = tiny_cfg()
    cfg.DATA.SOURCES = ["emotiontalk"]
    cfg.DATA.EMOTIONTALK_ROOT = str(tmp_path/"features")
    dialogue = Path(cfg.DATA.EMOTIONTALK_ROOT)/"dialogues"/"one"
    dialogue.mkdir(parents=True)
    provenance = dialogue/"provenance.json"
    provenance.write_text('{"cache_id":"one"}', encoding="utf-8")
    payload = data_provenance(cfg)
    validate_resume_data(payload, cfg)
    provenance.write_text('{"cache_id":"rebuilt"}', encoding="utf-8")
    with pytest.raises(ValueError, match="data_digests"):
        validate_resume_data(payload, cfg)


def test_a0_complete_checkpoint_restores_teacher(tmp_path, monkeypatch):
    from emotion_ssm.train.common import ObservationTrainingBundle
    from emotion_ssm.utils.checkpoint import save_training_checkpoint
    from emotion_ssm.utils.emotion_checkpoint import load_emotion
    cfg = tiny_cfg()
    bundle = ObservationTrainingBundle(ObservationEncoder.from_config(cfg), ObservationSupervisionHeads(8, 3, 3))
    teacher = copy.deepcopy(bundle.encoder).eval().requires_grad_(False)
    path = tmp_path/"a0.pt"
    save_training_checkpoint(path, 1, 2, {"bundle": bundle, "teacher": teacher}, config=cfg.dump())
    def forbidden(*a, **k):
        raise AssertionError("external pretrained weights accessed")
    monkeypatch.setattr("transformers.AutoModel.from_pretrained", forbidden)
    restored, restored_teacher, _, _ = load_emotion(path)
    assert restored.heads.speaker[2].out_features == 3
    for key, value in bundle.state_dict().items():
        assert torch.equal(restored.state_dict()[key], value)
    for key, value in teacher.state_dict().items():
        assert torch.equal(restored_teacher.state_dict()[key], value)


def test_paired_report_groups_roles_and_checks_experiment_protocol():
    from emotion_ssm.paired_report import compare
    keys = ("expression", "jaw", "neck", "velocity", "boundary_velocity")
    def result(seed, variant, error):
        rows = [{"dialogue": f"video{video}_sub_video_1_speaker{role}",
                 "sse": {k: error*role for k in keys}, "elements": {k: role for k in keys}}
                for video in range(2) for role in (1, 2)]
        return {"format_version": 2, "complete_evaluation": True, "seed": seed, "split": "test",
            "split_digest": "split", "feature_digest": "features", "text_protocol": "offline",
            "feature_source": {}, "protocol": {"max_steps": 30000}, "variant": variant,
            "ablation": "full", "dialogues": rows, "metrics": {k+"_mse": error for k in keys}}
    base = [result(seed, "none", 2.) for seed in (6666, 6667, 6668)]
    condition = [result(seed, "dyadic", 1.) for seed in (6666, 6667, 6668)]
    report = compare(base, condition, bootstrap=20)
    assert report["paired_clips"] == 2 and report["source_clusters"] == 2
    assert report["metrics"]["jaw_mse"]["source_cluster_95ci"] == [-1., -1.]
    condition[1]["variant"] = "affect"
    with pytest.raises(ValueError, match="one conditioned variant"):
        compare(base, condition, bootstrap=20)
    condition[1]["variant"] = "dyadic"
    condition[1]["text_protocol"] = "asr"
    with pytest.raises(ValueError, match="protocol"):
        compare(base, condition, bootstrap=20)


def test_stream_segments_sessions_roles_and_tail(system):
    packets = [packet(1.), packet(2.), packet(2.28, 7)]
    manager = StreamSessions(system)
    expected, state = [], None
    with torch.no_grad():
        for p in packets:
            out, state, _ = system(p, state)
            expected.append(out)
        actual = [manager(p)[0] for p in packets]
        assert torch.equal(torch.cat(expected, 1), torch.cat(actual, 1))
        assert manager.states["one"].history[0]["context"].shape[1] <= 75
        other = copy.deepcopy(packets[0]); other["session_id"] = "two"
        assert torch.equal(manager(other)[0], expected[0])
        wrong = copy.deepcopy(packets[0]); wrong["roles"] = ("user", "avatar")
        with pytest.raises(ValueError, match="reset"):
            manager(wrong)
        manager.reset("one")
        swapped, _, _ = manager(wrong)
        fresh, _, _ = system(wrong)
        assert torch.equal(swapped, fresh)


def test_complete_checkpoint_has_no_external_dependency(system, tmp_path, monkeypatch):
    cfg = tiny_cfg()
    cfg.DUALTALK.BASELINE_CHECKPOINT = str(tmp_path / "deleted_baseline.pt")
    cfg.DUALTALK.PHASE_B_CHECKPOINT = str(tmp_path / "deleted_phase_b.pt")
    cfg.DUALTALK.CALIBRATION_CHECKPOINT = str(tmp_path / "deleted_observer.pt")
    checkpoint = tmp_path / "complete.pt"
    p = packet()
    expected = system(p)[0]
    optimizer = torch.optim.AdamW([p for p in system.parameters() if p.requires_grad])
    save_generation(checkpoint, system, cfg, 3, optimizer)
    from transformers import AutoModel, AutoTokenizer
    def forbidden(*a, **k):
        raise AssertionError("External pretrained loading was attempted")
    monkeypatch.setattr(AutoModel, "from_pretrained", forbidden)
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", forbidden)
    restored, loaded, payload = load_generation(checkpoint)
    restored.eval()
    assert torch.equal(expected, restored(p)[0])
    assert payload["global_step"] == 3
    assert loaded.DUALTALK.FEATURE_DIM == 8


def observation(mask, event=True, duration=1.):
    return EventObservation(torch.randn(1, 8), torch.randn(1, 8), torch.randn(1, 8),
        torch.ones(1, 3), torch.tensor([mask], dtype=torch.bool), torch.tensor([event]), torch.tensor([duration]))


def test_missing_event_silence_and_irregular_prediction(system):
    ssm = system.state_model
    state = ssm.initialize(torch.full((1, 2), -1))
    state.z = torch.randn_like(state.z)
    empty = observation([0, 0, 0], False)
    assert torch.equal(ssm.event_stimulus(empty), torch.zeros(1, 8))
    torch.testing.assert_close(ssm.observe(state, (empty, empty), .7).z, ssm.decay_only(state, .7).z)
    torch.testing.assert_close(ssm.decay_only(ssm.decay_only(state, .7), 2.3).z, ssm.decay_only(state, 3.).z)
    visual = observation([0, 1, 0], False)
    assert not torch.equal(ssm.observe(state, (empty, visual), 1.).z, ssm.decay_only(state, 1.).z)
    predictions = ssm.predict_at(state, 10., [10.3, 12.9, 15.])
    for t, predicted in zip([.3, 2.9, 5.], predictions):
        torch.testing.assert_close(predicted.z, ssm.decay_only(state, t).z)
    inputs = [(11., (empty, visual))]
    conditional = ssm.predict_at(state, 10., [12.9], inputs)[0]
    visual.aff *= 1000
    assert torch.equal(conditional.z, ssm.predict_at(state, 10., [12.9], inputs)[0].z)


@pytest.mark.parametrize("variant", ["none", "affect", "self", "dyadic"])
def test_generation_backward_and_identical_backbone_permissions(system, variant):
    system.variant = variant
    system.configure_trainable()
    permissions = {n: p.requires_grad for n, p in system.generator.baseline.named_parameters()}
    assert all(v == ("feature_extractor" not in k) for k, v in permissions.items())
    p = packet()
    generated, _, _ = system(p)
    truth = torch.randn_like(generated)
    loss = reconstruction_loss(generated, truth)["total"]
    loss.backward()
    assert system.generator.baseline.joint_encoder.audio_projection.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in system.observer.parameters())
    assert all(p.grad is None for p in system.state_model.parameters())
    if variant == "none":
        direct = system.generator.baseline(normalize_audio(p["target_audio"]), normalize_audio(p["partner_audio"]), p["partner_blendshape"])
        torch.testing.assert_close(generated, direct, rtol=0, atol=0)


@pytest.mark.parametrize("norm", ["group", "layer"])
def test_short_audio_and_padding_pooling(norm):
    acoustic = Wav2Vec2Model(acoustic_config(norm)).eval()
    wave = torch.randn(1, 640)
    with torch.no_grad():
        expected = pooled_audio(acoustic, wave)
        batch = torch.cat([F.pad(wave, (0, 640)), torch.randn(1, 1280)], 0)
        observed = pooled_audio(acoustic, batch, torch.tensor([640, 1280]))
        torch.testing.assert_close(observed[:1], expected, rtol=1e-4, atol=1e-5)
        assert torch.equal(pooled_audio(acoustic, torch.zeros(1, 2)), torch.zeros(1, 16))


def test_source_split_text_alignment_and_identity():
    assert select_adapter_source(["iemocap"]) == 1
    assert select_adapter_source(["emotiontalk"]) == 0
    with pytest.raises(ValueError):
        select_adapter_source(["iemocap"], 0)
    first = FeatureSource("hubert", "bert").to_dict()
    with pytest.raises(ValueError, match="audio_model"):
        require_compatible(first, {**first, "audio_model": "wav2vec"})
    names = [f"video{i}_sub_video_1_speaker{s}" for i in range(20) for s in (1, 2)]
    manifest = make_split_manifest(names)
    assert not {s.split("_sub_video_")[0] for s in manifest["train"]} & {s.split("_sub_video_")[0] for s in manifest["val"]}
    items = [{"speaker_id": "a", "start_time": 0., "end_time": 1., "text": "hello"},
             {"speaker_id": "b", "start_time": 1., "end_time": 2., "text": "sad"}]
    contexts, events, _ = causal_texts(items, "dialogue")
    assert "sad" not in contexts[0] and "[A]" in contexts[1] and "[B]" in events[1]
    scores = torch.full((5, 3), -10.)
    scores[torch.arange(5), torch.tensor([0, 1, 0, 1, 0])] = 0.
    assert [(a, b) for a, b, _ in ctc_viterbi(scores, [1, 1])] == [(1, 2), (3, 4)]
    with pytest.raises(ValueError):
        ctc_viterbi(scores[:1], [1, 1])
    with pytest.raises(ValueError):
        ctc_viterbi(torch.empty(0, 3), [1])
    _, _, valid = slice_face_frames([{ "timestamp": .5, "face_id": "0", "confidence": 1., "success": 1, "au": [1.]*35}],
                                  0., 1., .8, require_identity=True)
    assert not valid.any()


def test_weighted_metrics_ignore_padding_and_batch_partition():
    p, t = torch.randn(5, 25, 56), torch.randn(5, 25, 56)
    mask = torch.arange(25)[None] < torch.tensor([25, 5, 10, 7, 25])[:, None]
    full = ReconstructionTotals(); full.update(p, t, mask)
    merged = ReconstructionTotals()
    for begin, end in [(0, 2), (2, 3), (3, 5)]:
        part = ReconstructionTotals(); part.update(p[begin:end], t[begin:end], mask[begin:end]); merged.merge(part)
    assert full.metrics() == pytest.approx(merged.metrics())
    previous = (p[:, -1], t[:, -1], torch.zeros(5, dtype=torch.bool))
    no_boundary = ReconstructionTotals(); no_boundary.update(p, t, mask, previous)
    assert no_boundary.elements["boundary_velocity"] == 0


def test_tiny_a0_phase_b_calibration_and_checkpoint(system, tmp_path):
    cfg = tiny_cfg()
    encoder = ObservationEncoder.from_config(cfg)
    teacher = copy.deepcopy(encoder).eval().requires_grad_(False)
    heads = ObservationSupervisionHeads(8, 2, 3)
    n = 8
    batch = {"audio": torch.randn(n, 16), "text": torch.randn(n, 16), "event_text": torch.randn(n, 16),
        "face": torch.randn(n, 3, 35), "face_frame_mask": torch.ones(n, 3, dtype=torch.bool),
        "face_confidence": torch.ones(n, 3), "modality_mask": torch.ones(n, 3, dtype=torch.bool),
        "event_present": torch.ones(n, dtype=torch.bool), "dataset_id": torch.arange(n)%2,
        "speaker": torch.arange(n)%2, "emotion": torch.arange(n)%7, "intensity": torch.rand(n),
        "vad": torch.rand(n, 3), "vad_mask": torch.ones(n, 3, dtype=torch.bool), "reliability": torch.ones(n, 3)}
    out = encoder(batch)
    loss = observation_losses(out, teacher(batch, torch.ones(1, 3, dtype=torch.bool)), heads(out.aff, 0.), batch, torch.ones(7), cfg)["total"]
    loss.backward()
    assert encoder.audio_adapter.adapters[0].layers[0].weight.grad.abs().sum() > 0
    sequence = {k: v.reshape(2, 4, *v.shape[1:]) for k, v in batch.items() if k != "speaker"}
    sequence.update(valid_mask=torch.ones(2, 4, dtype=torch.bool), active_role=torch.tensor([[0, 1, 0, 1]]).repeat(2, 1),
        speaker_ids=torch.tensor([[0, 1], [0, 1]]), dt_to_next=torch.tensor([[.5, 2., 1., 0.]]).repeat(2, 1),
        start_time=torch.tensor([[0., .5, 2.5, 3.5]]).repeat(2, 1),
        end_time=torch.tensor([[.5, 1., 3., 4.]]).repeat(2, 1),
        turn_position=torch.linspace(0, 1, 4).repeat(2, 1), dialogue_index=torch.tensor([0, 1]))
    bundle = DynamicsTrainingBundle(encoder, DyadicEmotionSSM.from_config(cfg, 2), AffectDecoder(8), cfg)
    bundle.set_phase_b_affect_frozen(True)
    bundle.zero_grad(set_to_none=True)
    values = bundle(sequence, make_teacher_aff(teacher, sequence), torch.ones(7), True, True)
    values["total"].backward()
    assert encoder.action_head[0].weight.grad is not None
    assert all(p.grad is None for p in encoder.shared_affect_projector.parameters())
    from emotion_ssm.utils.checkpoint import save_training_checkpoint, load_training_checkpoint
    save_training_checkpoint(tmp_path/"b.pt", 1, 1, {"bundle": bundle, "teacher": teacher}, config=cfg.dump())
    load_training_checkpoint(tmp_path/"b.pt", {"bundle": bundle})
    from emotion_ssm.utils.emotion_checkpoint import load_emotion
    restored, restored_teacher, restored_cfg, _ = load_emotion(tmp_path/"b.pt")
    assert restored_cfg.MODEL.OBSERVATION_DIM == 8
    for key, value in bundle.state_dict().items():
        assert torch.equal(restored.state_dict()[key], value)
    for key, value in teacher.state_dict().items():
        assert torch.equal(restored_teacher.state_dict()[key], value)
    observer = copy.deepcopy(system.observer)
    observer.flame.requires_grad_(True)
    from emotion_ssm.models.conditioned_dualtalk import BlendshapeAffectProjector
    projector = BlendshapeAffectProjector(56, 8)
    outputs = []
    for _ in range(3):
        p = packet()
        p["target_features"] = system.features(p["target_audio"], [], 1., 0., "avatar")
        truth, mask = torch.randn(1, 25, 56), torch.ones(1, 25, dtype=torch.bool)
        full, visual, target, valid = calibration_pair(observer, teacher, p, truth, mask, 1)
        outputs.append((full, visual, projector(truth), target, valid))
    calibration_loss(*(torch.cat(v) for v in zip(*outputs))).backward()
    assert observer.flame.input[0].weight.grad is not None


def _ddp_metrics_worker(rank, rendezvous, output):
    import torch.distributed as dist
    from emotion_ssm.train.common import ExactEvaluationSampler
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=Path(rendezvous).as_uri(), rank=rank, world_size=2)
    torch.manual_seed(91)
    prediction, target = torch.randn(5, 25, 56), torch.randn(5, 25, 56)
    mask = torch.arange(25)[None] < torch.tensor([25, 7, 19, 3, 11])[:, None]
    totals = ReconstructionTotals()
    indices = list(ExactEvaluationSampler(list(range(5)), rank, 2))
    for index in indices:
        totals.update(prediction[index:index+1], target[index:index+1], mask[index:index+1])
    totals.distributed_sum("cpu")
    Path(output + str(rank)).write_text(json.dumps({"metrics": totals.metrics(), "indices": indices}), encoding="utf-8")
    dist.destroy_process_group()


def test_real_ddp_metrics_have_no_duplicate_samples(tmp_path):
    import torch.multiprocessing as mp
    torch.manual_seed(91)
    p, t = torch.randn(5, 25, 56), torch.randn(5, 25, 56)
    mask = torch.arange(25)[None] < torch.tensor([25, 7, 19, 3, 11])[:, None]
    full = ReconstructionTotals(); full.update(p, t, mask)
    mp.spawn(_ddp_metrics_worker, args=(str(tmp_path/"rendezvous"), str(tmp_path/"rank")), nprocs=2, join=True)
    results = [json.loads((tmp_path/f"rank{rank}").read_text()) for rank in (0, 1)]
    assert sorted(results[0]["indices"] + results[1]["indices"]) == list(range(5))
    for result in results:
        assert result["metrics"] == pytest.approx(full.metrics(), abs=1e-12)


@pytest.mark.parametrize("variant", ["none", "affect", "self", "dyadic"])
def test_unified_optimizer_budget_and_resume(system, tmp_path, monkeypatch, variant):
    from emotion_ssm.train import generation
    cfg = tiny_cfg()
    cfg.TRAIN.OUTPUT_ROOT = str(tmp_path)
    cfg.TRAIN.EXPERIMENT_NAME = variant
    cfg.DUALTALK.VARIANT = variant
    system.variant = variant
    system.configure_trainable()
    class Dataset:
        names, manifest_digest = ["dialogue0", "dialogue1"], "same-split"
        def __init__(self):
            self.items = []
            for index in range(2):
                sequence = []
                for time, frames in ((1., 25), (1.2, 5)):
                    p = packet(time, frames, f"dialogue{index}")
                    for prefix in ("target", "partner"):
                        p[prefix+"_features"] = system.features(p[prefix+"_audio"], [], time, time-frames/25, "avatar")
                    sequence.append((p, torch.randn(1, frames, 56), torch.ones(1, frames, dtype=torch.bool)))
                self.items.append(sequence)
        def __len__(self):
            return 2
        def packets(self, index):
            yield from copy.deepcopy(self.items[index])
    data = Dataset()
    monkeypatch.setattr(generation, "build_streaming", lambda *a, **k: system)
    monkeypatch.setattr(generation, "make_dataset", lambda *a, **k: data)
    generation.run_generation(cfg)
    checkpoint = tmp_path / "dualtalk_conditioned" / variant / "last.pt"
    payload = torch.load(checkpoint, weights_only=False)
    assert payload["global_step"] == 2
    assert payload["run_state"]["ranks"][0]["packets_seen"] == 4
    assert payload["metrics"]["valid_frames"] == 60
    cfg.TRAIN.RESUME = str(checkpoint)
    generation.run_generation(cfg)  # already completed budget, no extra update
    assert torch.load(checkpoint, weights_only=False)["global_step"] == 2


def test_preprocessing_is_prefix_only_and_rejects_stale_inputs(system, tmp_path, monkeypatch):
    from scipy.io import wavfile
    from types import SimpleNamespace
    from emotion_ssm.preprocess import dualtalk as preparation
    from emotion_ssm.data.timed_dualtalk import TimedDualTalk
    from emotion_ssm.data.dualtalk import partner_stem
    raw, output = tmp_path/"raw", tmp_path/"features"
    (raw/"train").mkdir(parents=True)
    for video in ("video0", "video1"):
        for role in ("speaker1", "speaker2"):
            stem = f"{video}_sub_video_1_{role}"
            wavfile.write(raw/"train"/f"{stem}.wav", 16000, np.random.default_rng(2).normal(0, .1, 32*640).astype(np.float32))
            np.savez(raw/"train"/f"{stem}.npz", exp=np.zeros((32, 50), np.float32), pose=np.zeros((32, 6), np.float32))
            (raw/"train"/f"{stem}.txt").write_text("hello sad")
    def align(wave, text, role):
        return [{"id": f"{role}:{i}", "role": role, "text": word, "start": i+.1,
                 "end": i+.2, "available_at": i+.2} for i, word in enumerate(text.split())]
    monkeypatch.setattr(preparation, "FrozenFeatures", lambda *a, **k: system.features)
    monkeypatch.setattr(preparation, "TranscriptAligner", lambda **k: SimpleNamespace(align=align, model_name="test-trellis"))
    source = tiny_construction()["feature_source"]
    preparation.prepare(raw, output, source)
    dataset = TimedDualTalk(raw, output, output/"splits.json", expected_source=source)
    original_packets = list(dataset.packets(0))
    assert [p[1].shape[1] for p in original_packets] == [25, 7]
    expected = system(original_packets[0][0])[0]
    name = dataset.names[0]
    partner = partner_stem(name)
    for stem in (name, partner):
        rate, wave = wavfile.read(raw/"train"/f"{stem}.wav")
        wave[25*640:] *= -2
        wavfile.write(raw/"train"/f"{stem}.wav", rate, wave)
        (raw/"train"/f"{stem}.txt").write_text("hello hello")
    flame = np.zeros((32, 50), np.float32); flame[25:] = 100.
    np.savez(raw/"train"/f"{partner}.npz", exp=flame, pose=np.zeros((32, 6), np.float32))
    with pytest.raises(ValueError, match="Raw input changed"):
        list(dataset.packets(0))
    preparation.prepare(raw, output, source, rebuild=True)
    changed = TimedDualTalk(raw, output, output/"splits.json", expected_source=source)
    assert dataset.feature_digest != changed.feature_digest
    assert torch.equal(expected, system(next(changed.packets(0))[0])[0])


def test_injected_target_visual_is_removed_from_observation(system):
    p = packet()
    p["target_features"] = system.features(p["target_audio"], [], 1., 0., "avatar")
    expected, _, first = system(p)
    p["target_features"]["modality_mask"][:, 1] = True
    p["target_features"].update(flame=torch.randn(1, 25, 56)*100., visual_token=torch.randn(1, 16)*100.)
    actual, _, second = system(p)
    assert torch.equal(expected, actual)
    assert torch.equal(first["context"], second["context"])
    with pytest.raises(ValueError, match="caching"):
        system.generator.baseline.joint_encoder.extract_audio_features(p["target_audio"], p["partner_audio"])


def test_emotiontalk_rejects_synthetic_arousal_and_dominance(feature_root):
    from emotion_ssm.data.feature_store import FeatureDialogueStore
    from emotion_ssm.data.common import SpeakerVocabulary
    path = feature_root/"dialogues"/"dialogue_001"/"labels.json"
    labels = json.loads(path.read_text())
    for item in labels["utterances"]:
        item["vad"] = [1., 1., 1.]
        item["vad_mask"] = [True, True, True]
    path.write_text(json.dumps(labels))
    store = FeatureDialogueStore(feature_root, "emotiontalk", 0)
    record = store.load_dialogue("dialogue_001", SpeakerVocabulary(store.collect_speakers("train")))
    assert record.vad[:, 0].tolist() == [-1., -.5, 0., .5]
    assert not record.vad_mask[:, 1:].any()


def test_prediction_statistics_are_batch_invariant_with_missing_labels():
    from emotion_ssm.utils.prediction_statistics import PredictionStatistics
    n = 9
    predictions = {"aff": torch.randn(n, 8), "emotion": torch.randn(n, 7),
                   "intensity": torch.randn(n), "vad": torch.randn(n, 3)}
    target = {"aff": torch.randn(n, 8), "emotion": torch.arange(n)%7, "intensity": torch.randn(n),
        "intensity_mask": torch.arange(n)%2 == 0, "vad": torch.randn(n, 3),
        "vad_mask": torch.rand(n, 3) > .5}
    all_stats, split_stats = PredictionStatistics(), PredictionStatistics()
    all_stats.update("h1", predictions, target, torch.ones(n, dtype=torch.bool), torch.arange(1., 8.))
    for start, stop in ((0, 2), (2, 5), (5, 9)):
        split_stats.update("h1", {k: v[start:stop] for k, v in predictions.items()},
            {k: v[start:stop] for k, v in target.items()}, torch.ones(stop-start, dtype=torch.bool), torch.arange(1., 8.))
    assert all_stats.metrics() == pytest.approx(split_stats.metrics(), abs=1e-12)


def test_control_scripts_share_budget_and_source_bindings():
    from scripts.run_v2_experiments import parser, planned_commands
    for source in ("all", "iemocap", "emotiontalk"):
        args = parser().parse_args(["--baseline", "unused.pt", "--mode", "full", "--source", source])
        commands = planned_commands(args)
        assert len(commands) == 12
        for command in commands:
            assert command[command.index("TRAIN.MAX_STEPS")+1] == "30000"
            assert command[command.index("TRAIN.GLOBAL_CHUNKS_PER_STEP")+1] == "32"
            assert command[command.index("DUALTALK.ADAPTER_INIT_SOURCE")+1] == "-1"
            assert source in command[command.index("DUALTALK.SOURCE_MANIFEST")+1]


def test_counterfactual_validation_pools_are_independent_of_batch_and_rank():
    from types import SimpleNamespace
    from torch.utils.data import DataLoader
    from emotion_ssm.train.dynamics_trainer import validation_batches
    dataset = list(range(11))
    single = SimpleNamespace(rank=0, world_size=1)
    expected = [value.tolist() for value in validation_batches(DataLoader(dataset, batch_size=1), single, 4)]
    assert expected == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10]]
    assert [value.tolist() for value in validation_batches(DataLoader(dataset, batch_size=7), single, 4)] == expected
    ranks = [[value.tolist() for value in validation_batches(DataLoader(dataset, batch_size=2),
              SimpleNamespace(rank=rank, world_size=2), 4)] for rank in range(2)]
    assert sorted(ranks[0] + ranks[1]) == expected


def test_full_dynamics_validation_metrics_ignore_loader_batch_size():
    from torch.utils.data import DataLoader
    from emotion_ssm.data.full_dialogues import collate_full_dialogues
    from emotion_ssm.train.dynamics_trainer import validate
    from emotion_ssm.utils.distributed import DistributedContext
    torch.manual_seed(9)
    cfg = tiny_cfg()
    cfg.COUNTERFACTUAL.MAX_ACTION_COSINE = 1.1  # Ensure this fixture has candidates.
    encoder = ObservationEncoder.from_config(cfg)
    teacher = copy.deepcopy(encoder).eval().requires_grad_(False)
    bundle = DynamicsTrainingBundle(encoder, DyadicEmotionSSM.from_config(cfg, 2), AffectDecoder(8), cfg)
    data = []
    for i in range(3):
        n = 4
        role = torch.tensor([0, 1, 0, 1])
        data.append({"audio": torch.randn(n, 16), "text": torch.randn(n, 16), "event_text": torch.randn(n, 16),
            "face": torch.randn(n, 3, 35), "face_frame_mask": torch.ones(n, 3, dtype=torch.bool),
            "face_confidence": torch.ones(n, 3), "modality_mask": torch.ones(n, 3, dtype=torch.bool),
            "event_present": torch.ones(n, dtype=torch.bool), "dataset_id": torch.zeros(n, dtype=torch.long),
            "emotion": role.clone(), "intensity": torch.ones(n)*.5, "intensity_mask": torch.ones(n, dtype=torch.bool),
            "vad": torch.rand(n, 3), "vad_mask": torch.ones(n, 3, dtype=torch.bool), "reliability": torch.ones(n, 3),
            "valid_mask": torch.ones(n, dtype=torch.bool), "active_role": role, "speaker_ids": torch.tensor([0, 1]),
            "dt_to_next": torch.tensor([1., 2., .5, 0.]), "start_time": torch.tensor([0., 1., 3., 3.5]),
            "end_time": torch.tensor([.5, 1.5, 3.5, 4.]), "turn_position": torch.linspace(0, 1, n),
            "dialogue_index": torch.tensor(i)})
    context = DistributedContext(0, 0, 1, torch.device("cpu"))
    metrics = [validate(bundle, teacher, DataLoader(data, batch_size=size, collate_fn=collate_full_dialogues),
                        torch.ones(7), context, True, True) for size in (1, 3)]
    assert metrics[0] == pytest.approx(metrics[1], abs=1e-12)
    assert metrics[0]["val_cf_candidate_coverage"] > 0
