"""Smoke test for the discriminator and generator."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import CNNDiscriminator, ProgressiveGenerator


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main() -> None:
    config = load_config(Path("configs/base_celeba32.yaml"))
    model_cfg = config["model"]
    data_cfg = config["data"]
    batch_size = 4

    generator = ProgressiveGenerator(
        latent_dim=int(model_cfg["latent_dim"]),
        base_channels=int(model_cfg["base_channels"]),
        image_size=int(model_cfg["image_size"]),
        stage_resolutions=list(model_cfg["stage_resolutions"]),
        attention_schedule=list(model_cfg["attention_schedule"]),
        blocks_per_stage=int(model_cfg["blocks_per_stage"]),
        out_channels=int(data_cfg["channels"]),
        use_positional_embeddings=bool(model_cfg.get("use_positional_embeddings", True)),
    )
    discriminator = CNNDiscriminator(
        in_channels=int(data_cfg["channels"]),
        base_channels=64,
    )

    z = torch.randn(batch_size, int(model_cfg["latent_dim"]))
    fake_images = generator(z)
    logits = discriminator(fake_images)

    print(f"fake_images_shape={tuple(fake_images.shape)}")
    print(f"fake_images_range=({fake_images.min().item():.4f}, {fake_images.max().item():.4f})")
    print(f"logits_shape={tuple(logits.shape)}")


if __name__ == "__main__":
    main()
