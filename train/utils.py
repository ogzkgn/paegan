"""Training helpers for config loading, reproducibility, and artifacts."""

from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from torchvision.utils import make_grid, save_image


def load_config(config_path: str | Path) -> dict:
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_run_directories(base_dir: str | Path, experiment_name: str) -> dict[str, Path]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = Path(base_dir) / experiment_name / timestamp
    directories = {
        "run_root": run_root,
        "checkpoints": run_root / "checkpoints",
        "samples": run_root / "samples",
        "logs": run_root / "logs",
    }
    for path in directories.values():
        path.mkdir(parents=True, exist_ok=True)
    return directories


def save_checkpoint(
    checkpoint_path: str | Path,
    generator: torch.nn.Module,
    discriminator: torch.nn.Module,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    config: dict,
) -> None:
    checkpoint = {
        "generator": generator.state_dict(),
        "discriminator": discriminator.state_dict(),
        "optimizer_g": optimizer_g.state_dict(),
        "optimizer_d": optimizer_d.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "config": config,
    }
    torch.save(checkpoint, checkpoint_path)


def save_sample_grid(
    generator: torch.nn.Module,
    fixed_noise: torch.Tensor,
    sample_path: str | Path,
    device: torch.device,
    nrow: int = 4,
) -> None:
    generator.eval()
    with torch.no_grad():
        images = generator(fixed_noise.to(device)).cpu()
    images = (images + 1.0) / 2.0
    grid = make_grid(images.clamp(0.0, 1.0), nrow=nrow)
    save_image(grid, sample_path)
    generator.train()


def append_metrics(log_path: str | Path, metrics: dict) -> None:
    with Path(log_path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(metrics) + "\n")
