from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from emotion_ssm.config import load_config
from emotion_ssm.data import DialogueWindowDataset, UnifiedUtteranceDataset, build_speaker_vocabulary
from emotion_ssm.metrics import ccc_value, classification_metrics, confusion_matrix
from emotion_ssm.models import (
    SUBSET_MASKS,
    SUBSET_NAMES,
    ObservationEncoder,
    ObservationSupervisionHeads,
)
from emotion_ssm.train.common import build_feature_stores, move_to_device
from emotion_ssm.train.dynamics_core import (
    initialize_dynamics_bundle,
    load_component_state,
    make_teacher_aff,
)
from emotion_ssm.utils.paths import ensure_output_directory


@torch.no_grad()
def evaluate_observation(encoder, heads, loader, device) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
    encoder.eval()
    heads.eval()
    matrices = [torch.zeros(7, 7, dtype=torch.long) for _ in SUBSET_NAMES]
    intensity_error = torch.zeros(len(SUBSET_NAMES))
    counts = torch.zeros(len(SUBSET_NAMES))
    vad_prediction: List[torch.Tensor] = []
    vad_target: List[torch.Tensor] = []
    vad_mask: List[torch.Tensor] = []
    full_aff: List[torch.Tensor] = []
    speakers: List[torch.Tensor] = []
    domains: List[torch.Tensor] = []
    for raw_batch in loader:
        batch = move_to_device(raw_batch, device)
        output = encoder(batch, SUBSET_MASKS.to(device))
        predictions = heads(output.aff, 0.0)
        emotion_prediction = predictions["emotion"].argmax(-1)
        for subset in range(len(SUBSET_NAMES)):
            valid = output.valid_subsets[:, subset]
            matrices[subset] += confusion_matrix(
                batch["emotion"], emotion_prediction[:, subset], mask=valid
            )
            intensity_error[subset] += (
                predictions["intensity"][:, subset] - batch["intensity"]
            ).abs()[valid].sum().cpu()
            counts[subset] += valid.sum().cpu()
        vad_prediction.append(predictions["vad"][:, -1].cpu())
        vad_target.append(batch["vad"].cpu())
        vad_mask.append((batch["vad_mask"] & output.valid_subsets[:, -1, None]).cpu())
        full_aff.append(output.aff[:, -1].cpu())
        speakers.append(batch["speaker"].cpu())
        domains.append(batch["dataset_id"].cpu())
    metrics: Dict[str, float] = {}
    for subset, name in enumerate(SUBSET_NAMES):
        values = classification_metrics(matrices[subset])
        metrics[f"{name}_macro_f1"] = values["macro_f1"]
        metrics[f"{name}_uar"] = values["uar"]
        metrics[f"{name}_intensity_mae"] = float(
            intensity_error[subset] / counts[subset].clamp_min(1)
        )
    metrics["AVT_vad_ccc"] = ccc_value(
        torch.cat(vad_prediction), torch.cat(vad_target), torch.cat(vad_mask)
    )
    evidence = {
        "aff": torch.cat(full_aff),
        "speaker": torch.cat(speakers),
        "domain": torch.cat(domains),
    }
    return metrics, evidence


def linear_leakage_probe(features: torch.Tensor, target: torch.Tensor, seed: int) -> float:
    valid = target >= 0
    features = features[valid].float()
    target = target[valid].long()
    if len(features) < 10 or target.unique().numel() < 2:
        return 0.0
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(len(features), generator=generator)
    split = max(int(len(features) * 0.8), 1)
    train_index, test_index = permutation[:split], permutation[split:]
    if len(test_index) == 0:
        return 0.0
    mean = features[train_index].mean(0)
    std = features[train_index].std(0).clamp_min(1e-6)
    x_train = (features[train_index] - mean) / std
    x_test = (features[test_index] - mean) / std
    classifier = nn.Linear(features.shape[-1], int(target.max()) + 1)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.03)
    for _ in range(100):
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(classifier(x_train), target[train_index])
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        return float((classifier(x_test).argmax(-1) == target[test_index]).float().mean())


@torch.no_grad()
def evaluate_dynamics(
    bundle, teacher, loader, class_weights, device, enable_partner: bool
) -> Dict[str, float]:
    bundle.eval()
    totals: Dict[str, float] = {}
    batches = 0
    for raw_batch in loader:
        batch = move_to_device(raw_batch, device)
        target_aff = make_teacher_aff(teacher, batch)
        full = bundle(
            batch,
            target_aff,
            class_weights,
            enable_partner,
            enable_partner,
        )
        for name, value in full.items():
            totals[name] = totals.get(name, 0.0) + float(value)
        if enable_partner:
            self_only = bundle(batch, target_aff, class_weights, False, False)
            totals["partner_gain"] = totals.get("partner_gain", 0.0) + float(
                self_only["h1"] - full["h1"]
            )
        batches += 1
    return {name: value / max(batches, 1) for name, value in totals.items()}


@torch.no_grad()
def state_curves(model) -> Dict[str, object]:
    speaker_ids = torch.full((1, 2), -1, dtype=torch.long, device=next(model.parameters()).device)
    _, tau = model.personal(speaker_ids)
    tau = tau[0, 0]
    groups = torch.tensor_split(tau, model.personal.num_timescales)
    group_tau = torch.stack([value.mean() for value in groups])
    time = torch.linspace(0.0, 300.0, 61, device=tau.device)
    decay = torch.exp(-time[:, None] / group_tau[None])
    accumulation = []
    value = torch.zeros_like(group_tau)
    one_second_decay = torch.exp(-1.0 / group_tau)
    for _ in range(33):
        value = value * one_second_decay + 1.0
        accumulation.append(value.clone())
    return {
        "tau_seconds": group_tau.cpu().tolist(),
        "decay_time_seconds": time.cpu().tolist(),
        "decay": decay.transpose(0, 1).cpu().tolist(),
        "unit_impulse_accumulation": torch.stack(accumulation)
        .transpose(0, 1)
        .cpu()
        .tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate emotion observation and dyadic state models")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--observation-checkpoint", type=Path, required=True)
    parser.add_argument("--heads-checkpoint", type=Path, required=True)
    parser.add_argument("--ema-checkpoint", type=Path, required=True)
    parser.add_argument("--dynamics-checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    cfg = load_config(args.config, args.opts)
    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else "cpu")
    stores = build_feature_stores(cfg)
    speaker_vocab = build_speaker_vocabulary(stores)
    utterances = UnifiedUtteranceDataset(stores, args.split, speaker_vocab)
    utterance_loader = DataLoader(
        utterances,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.DATA.NUM_WORKERS,
    )
    encoder = ObservationEncoder.from_config(cfg).to(device)
    encoder.load_state_dict(load_component_state(args.observation_checkpoint, "encoder"))
    heads = ObservationSupervisionHeads(
        cfg.MODEL.OBSERVATION_DIM,
        len(speaker_vocab),
        cfg.MODEL.NUM_DOMAINS,
    ).to(device)
    heads.load_state_dict(load_component_state(args.heads_checkpoint, "heads"))
    metrics, evidence = evaluate_observation(encoder, heads, utterance_loader, device)
    if args.split != "train":
        probe_dataset = UnifiedUtteranceDataset(stores, "train", speaker_vocab)
        probe_loader = DataLoader(
            probe_dataset,
            batch_size=cfg.TRAIN.BATCH_SIZE,
            shuffle=False,
            num_workers=cfg.DATA.NUM_WORKERS,
        )
        _, evidence = evaluate_observation(encoder, heads, probe_loader, device)
    metrics["speaker_leakage_accuracy"] = linear_leakage_probe(
        evidence["aff"], evidence["speaker"], cfg.SEED
    )
    metrics["domain_leakage_accuracy"] = linear_leakage_probe(
        evidence["aff"], evidence["domain"], cfg.SEED
    )

    if args.dynamics_checkpoint:
        cfg_for_dynamics = cfg.clone()
        cfg_for_dynamics.defrost()
        cfg_for_dynamics.TRAIN.OBSERVATION_CHECKPOINT = str(args.observation_checkpoint)
        cfg_for_dynamics.TRAIN.EMOTION_HEADS_CHECKPOINT = str(args.heads_checkpoint)
        cfg_for_dynamics.freeze()
        bundle = initialize_dynamics_bundle(cfg_for_dynamics, len(speaker_vocab)).to(device)
        bundle.encoder.load_state_dict(
            load_component_state(args.dynamics_checkpoint, "encoder")
        )
        bundle.state_model.load_state_dict(
            load_component_state(args.dynamics_checkpoint, "state_model")
        )
        bundle.decoder.load_state_dict(load_component_state(args.dynamics_checkpoint, "decoder"))
        teacher = ObservationEncoder.from_config(cfg).to(device)
        teacher.load_state_dict(load_component_state(args.ema_checkpoint, "teacher"))
        teacher.eval()
        windows = DialogueWindowDataset(
            stores,
            args.split,
            speaker_vocab,
            cfg.DATA.WINDOW_LENGTH,
            cfg.DATA.WINDOW_STRIDE,
        )
        window_loader = DataLoader(
            windows,
            batch_size=cfg.TRAIN.SEQUENCE_BATCH_SIZE,
            shuffle=False,
            num_workers=cfg.DATA.NUM_WORKERS,
        )
        dynamics = evaluate_dynamics(
            bundle,
            teacher,
            window_loader,
            windows.class_weights().to(device),
            device,
            cfg.DYNAMICS.ENABLE_PARTNER,
        )
        metrics.update({f"dynamics_{name}": value for name, value in dynamics.items()})
        metrics["dynamics_corrected_teacher_forced_h1"] = dynamics.get("h1", 0.0)
        for horizon in cfg.DYNAMICS.HORIZONS:
            metrics[f"dynamics_open_loop_h{horizon}"] = dynamics.get(
                f"h{horizon}", 0.0
            )
        metrics["state_curves"] = state_curves(bundle.state_model)
    text = json.dumps(metrics, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        output_dir = ensure_output_directory(
            args.output.parent,
            [
                cfg.DATA.ROOT,
                cfg.DATA.EMOTIONTALK_ROOT,
                cfg.DATA.IEMOCAP_RAW_ROOT,
                cfg.DATA.DUALTALK_ROOT,
            ],
        )
        (output_dir / args.output.name).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
