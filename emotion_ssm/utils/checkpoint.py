from __future__ import annotations

import random
import inspect
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np
import torch

from .distributed import unwrap_model


def capture_rng_state() -> Dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_training_checkpoint(
    path: Path,
    epoch: int,
    global_step: int,
    models: Mapping[str, Any],
    optimizer=None,
    scheduler=None,
    scaler=None,
    metrics: Optional[Mapping[str, float]] = None,
    config: Optional[Mapping[str, Any]] = None,
) -> None:
    payload = {
        "epoch": epoch,
        "global_step": global_step,
        "models": {
            name: unwrap_model(model).state_dict() for name, model in models.items()
        },
        "optimizer": None if optimizer is None else optimizer.state_dict(),
        "scheduler": None if scheduler is None else scheduler.state_dict(),
        "scaler": None if scaler is None else scaler.state_dict(),
        "metrics": dict(metrics or {}),
        "config": config,
        "rng_state": capture_rng_state(),
    }
    torch.save(payload, str(path))


def load_training_checkpoint(
    path: Path,
    models: Mapping[str, Any],
    optimizer=None,
    scheduler=None,
    scaler=None,
    strict: bool = True,
) -> Dict[str, Any]:
    load_kwargs = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        load_kwargs["weights_only"] = False
    checkpoint = torch.load(str(path), **load_kwargs)
    for name, model in models.items():
        if name not in checkpoint["models"]:
            raise KeyError(f"Checkpoint does not contain model '{name}'")
        unwrap_model(model).load_state_dict(checkpoint["models"][name], strict=strict)
    if optimizer is not None and checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler is not None and checkpoint.get("scaler") is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    if checkpoint.get("rng_state") is not None:
        restore_rng_state(checkpoint["rng_state"])
    return checkpoint
