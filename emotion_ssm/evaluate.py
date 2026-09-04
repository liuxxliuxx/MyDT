from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
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
    subset_vad_prediction: List[List[torch.Tensor]] = [list() for _ in SUBSET_NAMES]
    subset_vad_mask: List[List[torch.Tensor]] = [list() for _ in SUBSET_NAMES]
    subset_vad_target: List[List[torch.Tensor]] = [list() for _ in SUBSET_NAMES]
    alignment = {
        "AV": {"same_sum": 0.0, "same_count": 0, "random_sum": 0.0, "random_count": 0},
        "AT": {"same_sum": 0.0, "same_count": 0, "random_sum": 0.0, "random_count": 0},
        "VT": {"same_sum": 0.0, "same_count": 0, "random_sum": 0.0, "random_count": 0},
    }
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
        for subset in range(len(SUBSET_NAMES)):
            subset_vad_prediction[subset].append(predictions["vad"][:, subset].cpu())
            subset_vad_target[subset].append(batch["vad"].cpu())
            subset_vad_mask[subset].append(
                (batch["vad_mask"] & output.valid_subsets[:, subset, None]).cpu()
            )
        if output.modality_aff is not None:
            actual_mask = batch["modality_mask"].bool()
            for name, left, right in (("AV", 0, 1), ("AT", 0, 2), ("VT", 1, 2)):
                same_valid = actual_mask[:, left] & actual_mask[:, right]
                same = F.cosine_similarity(
                    output.modality_aff[:, left], output.modality_aff[:, right], dim=-1
                )
                alignment[name]["same_sum"] += float(same[same_valid].sum().cpu())
                alignment[name]["same_count"] += int(same_valid.sum().cpu())
                if len(actual_mask) > 1:
                    permutation = torch.roll(
                        torch.arange(len(actual_mask), device=device), shifts=1
                    )
                    random_valid = actual_mask[:, left] & actual_mask[permutation, right]
                    random = F.cosine_similarity(
                        output.modality_aff[:, left],
                        output.modality_aff[permutation, right],
                        dim=-1,
                    )
                    alignment[name]["random_sum"] += float(
                        random[random_valid].sum().cpu()
                    )
                    alignment[name]["random_count"] += int(random_valid.sum().cpu())
        full_valid = output.valid_subsets[:, -1]
        full_aff.append(output.aff[full_valid, -1].cpu())
        speakers.append(batch["speaker"][full_valid].cpu())
        domains.append(batch["dataset_id"][full_valid].cpu())
    metrics: Dict[str, float] = {}
    for subset, name in enumerate(SUBSET_NAMES):
        values = classification_metrics(matrices[subset])
        metrics[f"{name}_macro_f1"] = values["macro_f1"]
        metrics[f"{name}_uar"] = values["uar"]
        metrics[f"{name}_intensity_mae"] = float(
            intensity_error[subset] / counts[subset].clamp_min(1)
        )
        metrics[f"{name}_vad_ccc"] = ccc_value(
            torch.cat(subset_vad_prediction[subset]),
            torch.cat(subset_vad_target[subset]),
            torch.cat(subset_vad_mask[subset]),
        )
    metrics["mean_subset_f1"] = sum(
        metrics[f"{name}_macro_f1"] for name in SUBSET_NAMES
    ) / len(SUBSET_NAMES)
    for name, values in alignment.items():
        same_count = values["same_count"]
        random_count = values["random_count"]
        same = values["same_sum"] / max(same_count, 1)
        random = values["random_sum"] / max(random_count, 1)
        metrics[f"{name}_same_utterance_cosine"] = same
        metrics[f"{name}_random_utterance_cosine"] = random
        metrics[f"{name}_alignment_margin"] = same - random
        metrics[f"{name}_alignment_pairs"] = float(min(same_count, random_count))
    evidence = {
        "aff": torch.cat(full_aff),
        "speaker": torch.cat(speakers),
        "domain": torch.cat(domains),
    }
    # These diagnostics make the collapse and alignment contracts visible in
    # a real run instead of limiting them to synthetic unit tests.
    if evidence["aff"].shape[0] > 1:
        latent_std = evidence["aff"].std(dim=0, unbiased=False)
        evidence["latent_std_mean"] = latent_std.mean()
        evidence["latent_near_constant_dimensions"] = (latent_std < 1e-3).sum()
    else:
        evidence["latent_std_mean"] = torch.tensor(0.0)
        evidence["latent_near_constant_dimensions"] = torch.tensor(
            evidence["aff"].shape[-1]
        )
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
    # The split and probe initialization are both fixed so leakage comparisons
    # between checkpoints are reproducible.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        classifier = nn.Linear(features.shape[-1], int(target.max()) + 1)
        optimizer = torch.optim.Adam(classifier.parameters(), lr=0.03)
        for _ in range(100):
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(classifier(x_train), target[train_index])
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            return float(
                (classifier(x_test).argmax(-1) == target[test_index]).float().mean()
            )


def chance_accuracy(target: torch.Tensor) -> float:
    """Uniform random-guess accuracy for labels actually present in a split."""
    valid = target[target >= 0]
    classes = valid.unique()
    return 1.0 / len(classes) if len(classes) else 0.0


def observation_contract_failures(
    metrics: Mapping[str, object],
    reference_metrics: Optional[Mapping[str, object]] = None,
    min_alignment_margin: float = 0.0,
    max_latent_constant_dimensions: int = 0,
    max_leakage_excess: float = 0.1,
    max_emotion_f1_drop: float = 0.02,
) -> List[str]:
    """Return failed real-run observation contracts without hiding metrics."""

    def number(name: str) -> Optional[float]:
        value = metrics.get(name)
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        return value if math.isfinite(value) else None

    failures = []
    for name in ("AV", "AT", "VT"):
        pairs = number(f"{name}_alignment_pairs")
        margin = number(f"{name}_alignment_margin")
        if pairs is None or pairs < 1:
            failures.append(f"{name}: no valid same/random cross-modal pairs")
        elif margin is None or margin < min_alignment_margin:
            failures.append(
                f"{name}: alignment margin {margin} is below {min_alignment_margin}"
            )
    collapsed = number("latent_near_constant_dimensions")
    if collapsed is None or collapsed > max_latent_constant_dimensions:
        failures.append(
            "latent_near_constant_dimensions exceeds "
            f"{max_latent_constant_dimensions}"
        )
    for name in ("A", "V", "T", "AVT"):
        if number(f"{name}_vad_ccc") is None:
            failures.append(f"{name}: VAD CCC is unavailable")
    for name in ("speaker", "domain"):
        leakage = number(f"{name}_leakage_accuracy")
        chance = number(f"{name}_chance")
        if leakage is None or chance is None:
            failures.append(f"{name}: leakage or chance metric is unavailable")
        elif leakage > chance + max_leakage_excess:
            failures.append(
                f"{name}: leakage {leakage} exceeds chance {chance} by more than "
                f"{max_leakage_excess}"
            )
    if reference_metrics is not None:
        current_f1 = number("mean_subset_f1")
        try:
            reference_f1 = float(reference_metrics["mean_subset_f1"])
        except (KeyError, TypeError, ValueError):
            reference_f1 = float("nan")
        if current_f1 is None or not math.isfinite(reference_f1):
            failures.append("mean_subset_f1 is unavailable for the reference comparison")
        elif current_f1 < reference_f1 - max_emotion_f1_drop:
            failures.append(
                f"mean_subset_f1 dropped from {reference_f1} to {current_f1}, "
                f"more than {max_emotion_f1_drop}"
            )
    return failures


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
    parser.add_argument("--reference-metrics", type=Path)
    parser.add_argument("--assert-observation-contracts", action="store_true")
    parser.add_argument("--min-alignment-margin", type=float, default=0.0)
    parser.add_argument("--max-latent-constant-dimensions", type=int, default=0)
    parser.add_argument("--max-leakage-excess", type=float, default=0.1)
    parser.add_argument("--max-emotion-f1-drop", type=float, default=0.02)
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
    probe_evidence = evidence
    if args.split != "train":
        probe_dataset = UnifiedUtteranceDataset(stores, "train", speaker_vocab)
        probe_loader = DataLoader(
            probe_dataset,
            batch_size=cfg.TRAIN.BATCH_SIZE,
            shuffle=False,
            num_workers=cfg.DATA.NUM_WORKERS,
        )
        _, probe_evidence = evaluate_observation(encoder, heads, probe_loader, device)
    metrics["speaker_leakage_accuracy"] = linear_leakage_probe(
        probe_evidence["aff"], probe_evidence["speaker"], cfg.SEED
    )
    metrics["domain_leakage_accuracy"] = linear_leakage_probe(
        probe_evidence["aff"], probe_evidence["domain"], cfg.SEED
    )
    metrics["latent_std_mean"] = float(evidence["latent_std_mean"])
    metrics["latent_near_constant_dimensions"] = int(
        evidence["latent_near_constant_dimensions"]
    )
    metrics["speaker_chance"] = chance_accuracy(probe_evidence["speaker"])
    metrics["domain_chance"] = chance_accuracy(probe_evidence["domain"])
    metrics["speaker_leakage_excess"] = metrics["speaker_leakage_accuracy"] - metrics["speaker_chance"]
    metrics["domain_leakage_excess"] = metrics["domain_leakage_accuracy"] - metrics["domain_chance"]

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
    reference_metrics = None
    if args.reference_metrics is not None:
        if not args.reference_metrics.is_file():
            raise FileNotFoundError(args.reference_metrics)
        reference_metrics = json.loads(args.reference_metrics.read_text(encoding="utf-8"))
    failures = []
    if args.assert_observation_contracts:
        failures = observation_contract_failures(
            metrics,
            reference_metrics,
            args.min_alignment_margin,
            args.max_latent_constant_dimensions,
            args.max_leakage_excess,
            args.max_emotion_f1_drop,
        )
        metrics["observation_contract_passed"] = not failures
        metrics["observation_contract_failures"] = failures
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
    if failures:
        raise SystemExit("Observation contracts failed: " + "; ".join(failures))


if __name__ == "__main__":
    main()
