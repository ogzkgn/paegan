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
        out_channels=int(data_cfg["channels"]),
        attention_enabled=bool(model_cfg.get("attention_enabled", False)),
        attention_mode=str(model_cfg.get("attention_mode", "fixed")),
        fixed_attention_type=str(model_cfg.get("fixed_attention_type", "global")),
        attention_resolution=int(model_cfg.get("attention_resolution", 16)),
        attention_num_heads=int(model_cfg.get("attention_num_heads", 4)),
        progressive_attention_schedule=list(model_cfg.get("progressive_attention_schedule", [])),
    )
    if hasattr(generator, "set_epoch"):
        generator.set_epoch(1)
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
    if hasattr(generator, "current_attention_type"):
        print(f"attention_type={generator.current_attention_type}")


if __name__ == "__main__":
    main()
