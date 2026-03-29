"""Train the baseline PAE-GAN setup."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train.trainer import train_gan
from train.utils import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PAE-GAN.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/base_celeba32.yaml"),
        help="Path to the experiment config file.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional limit for quick smoke tests.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    run_root = train_gan(config=config, max_steps=args.max_steps)
    print(f"run_root={run_root}")


if __name__ == "__main__":
    main()
