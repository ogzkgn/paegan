"""Sanity check for the CelebA dataloader."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.build import build_dataloader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check PAE-GAN data pipeline.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/base_celeba32.yaml"),
        help="Path to the experiment config file.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Override dataloader workers for portable sanity checks.",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config["data"]["num_workers"] = args.num_workers

    dataloader = build_dataloader(
        data_cfg=config["data"],
        batch_size=int(config["train"]["batch_size"]),
        seed=int(config["seed"]),
        shuffle=True,
        drop_last=True,
    )

    batch = next(iter(dataloader))

    print(f"dataset_size={len(dataloader.dataset)}")
    print(f"batch_shape={tuple(batch.shape)}")
    print(f"dtype={batch.dtype}")
    print(f"min={batch.min().item():.4f}")
    print(f"max={batch.max().item():.4f}")
    print(f"mean={batch.mean().item():.4f}")
    print(f"std={batch.std().item():.4f}")
    print(f"is_finite={torch.isfinite(batch).all().item()}")


if __name__ == "__main__":
    main()
