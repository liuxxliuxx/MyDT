from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from emotion_ssm.config import load_config
from emotion_ssm.models import DyadicEmotionSSM
from emotion_ssm.schema import DyadicState, EventObservation
from emotion_ssm.train.dynamics_core import load_component_state
from emotion_ssm.utils.paths import ensure_output_directory


def _load(path: Path):
    kwargs = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = False
    return torch.load(str(path), **kwargs)


def _event(value, turn: int, candidates: int, device: torch.device) -> EventObservation:
    def expand(name: str, default_dim: int) -> torch.Tensor:
        tensor = value.get(name)
        if tensor is None:
            return torch.zeros(candidates, default_dim, device=device)
        tensor = tensor[turn].to(device).float()
        return tensor[None].expand(candidates, -1)

    return EventObservation(
        aff=expand("aff", 128),
        event=expand("event", 128),
        action=expand("action", 128),
        reliability=expand("reliability", 3),
        modality_mask=expand("modality_mask", 3).bool(),
    )


@torch.no_grad()
def rollout_candidates(model, payload, device: torch.device):
    user = payload["user_observation"]
    candidate_actions = payload["candidate_actions"].to(device).float()
    if candidate_actions.ndim != 3:
        raise ValueError("candidate_actions must be [candidates, turns, 128]")
    candidates, turns, _ = candidate_actions.shape
    state = model.initialize(
        torch.full((candidates, 2), -1, dtype=torch.long, device=device)
    )
    user_z = []
    avatar_z = []
    relations = []
    dt = payload.get("dt", torch.ones(turns)).to(device).float()
    for turn in range(turns):
        user_event = _event(user, turn, candidates, device)
        user_step = model.step(
            state,
            user_event,
            torch.zeros(candidates, dtype=torch.long, device=device),
            dt[turn].expand(candidates) * 0.5,
            enable_partner=True,
            correct=True,
        )
        avatar_event = EventObservation(
            aff=torch.zeros(candidates, 128, device=device),
            event=torch.zeros(candidates, 128, device=device),
            action=candidate_actions[:, turn],
            reliability=torch.zeros(candidates, 3, device=device),
            modality_mask=torch.zeros(candidates, 3, dtype=torch.bool, device=device),
        )
        avatar_step = model.step(
            user_step.next_prior,
            avatar_event,
            torch.ones(candidates, dtype=torch.long, device=device),
            dt[turn].expand(candidates) * 0.5,
            enable_partner=True,
            correct=False,
        )
        state = avatar_step.next_prior
        user_z.append(state.z[:, 0])
        avatar_z.append(state.z[:, 1])
        relations.append(state.relation)
    user_z = torch.stack(user_z, dim=1)
    avatar_z = torch.stack(avatar_z, dim=1)
    relation = torch.stack(relations, dim=1)
    user_aff = model.state_to_aff(user_z)
    avatar_aff = model.state_to_aff(avatar_z)
    if "target_user_aff" in payload:
        target = payload["target_user_aff"].to(device).float()
        if target.ndim == 1:
            target = target[None].expand(candidates, -1)
        score = F.cosine_similarity(user_aff[:, -1], target, dim=-1)
    else:
        score = F.cosine_similarity(user_aff, avatar_aff, dim=-1).mean(dim=-1)
    order = score.argsort(descending=True)
    return {
        "user_z": user_z.cpu(),
        "avatar_z": avatar_z.cpu(),
        "relation": relation.cpu(),
        "user_aff": user_aff.cpu(),
        "avatar_aff": avatar_aff.cpu(),
        "score": score.cpu(),
        "ranking": order.cpu(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank candidate avatar action trajectories")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    cfg = load_config(args.config, args.opts)
    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else "cpu")
    model = DyadicEmotionSSM.from_config(cfg, num_speakers=1).to(device)
    state = load_component_state(args.checkpoint, "state_model")
    state = {
        name: value
        for name, value in state.items()
        if name not in {"personal.baseline_delta.weight", "personal.tau_delta.weight"}
    }
    model.load_state_dict(state, strict=False)
    model.eval()
    payload = _load(args.input)
    output = rollout_candidates(model, payload, device)
    output_dir = ensure_output_directory(
        args.output.parent,
        [
            cfg.DATA.ROOT,
            cfg.DATA.EMOTIONTALK_ROOT,
            cfg.DATA.IEMOCAP_RAW_ROOT,
            cfg.DATA.DUALTALK_ROOT,
        ],
    )
    output_path = output_dir / args.output.name
    torch.save(output, output_path)
    summary = {
        "ranking": output["ranking"].tolist(),
        "score": output["score"].tolist(),
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
