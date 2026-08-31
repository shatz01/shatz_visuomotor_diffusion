"""Shared utilities for training and loading policies."""

import random
import re
from pathlib import Path

import numpy as np
import torch

from src.train.model import SimpleTrajectoryModel


def save_checkpoint(
    path: Path,
    model: SimpleTrajectoryModel,
    metadata: dict[str, object],
    epoch: int | None,
    validation_metrics: dict[str, float] | None,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            **metadata,
            "epoch": epoch,
            "validation_metrics": validation_metrics,
        },
        path,
    )


def create_run_output_dir(output_root: Path, run_name: str, run_id: str) -> Path:
    """Create a unique, filesystem-safe directory for one W&B run."""
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "-", run_name).strip("._-") or "run"
    safe_id = re.sub(r"[^A-Za-z0-9._-]+", "-", run_id).strip("._-")
    if not safe_id:
        raise ValueError("The W&B run ID cannot be empty.")
    output_dir = output_root / f"{safe_name}-{safe_id}"
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(requested)
