"""FID evaluation helpers."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import torch
from PIL import Image
from pytorch_fid.fid_score import calculate_fid_given_paths
from torchvision.transforms import functional as TF

from models import ProgressiveGenerator


def _load_checkpoint(checkpoint_path: str | Path) -> dict:
    return torch.load(checkpoint_path, map_location="cpu")


def _build_generator_from_config(config: dict) -> ProgressiveGenerator:
    model_cfg = config["model"]
    data_cfg = config["data"]
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
    return generator


def _load_generator_weights(generator: ProgressiveGenerator, state_dict: dict) -> None:
    missing_keys, unexpected_keys = generator.load_state_dict(state_dict, strict=False)
    allowed_missing_prefix = "positional_embeddings."

    disallowed_missing = [key for key in missing_keys if not key.startswith(allowed_missing_prefix)]
    if disallowed_missing or unexpected_keys:
        raise RuntimeError(
            f"Checkpoint/model mismatch. Missing: {disallowed_missing}, unexpected: {unexpected_keys}"
        )


def _prepare_real_cache(real_root: str | Path, cache_dir: str | Path, image_size: int, limit: int | None) -> Path:
    real_root = Path(real_root)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        path
        for path in real_root.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )
    if limit is not None:
        image_paths = image_paths[:limit]

    existing_files = list(cache_dir.glob("*.png"))
    if len(existing_files) == len(image_paths) and len(existing_files) > 0:
        return cache_dir

    for old_file in existing_files:
        old_file.unlink()

    for index, image_path in enumerate(image_paths):
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image = image.resize((image_size, image_size), Image.BILINEAR)
            image.save(cache_dir / f"{index:06d}.png")

    return cache_dir


def _generate_fake_images(
    checkpoint: dict,
    output_dir: str | Path,
    num_samples: int,
    batch_size: int,
    device: torch.device,
) -> Path:
    config = checkpoint["config"]
    latent_dim = int(config["model"]["latent_dim"])

    output_dir = Path(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generator = _build_generator_from_config(config)
    _load_generator_weights(generator, checkpoint["generator"])
    generator.to(device)
    generator.eval()

    image_index = 0
    with torch.no_grad():
        while image_index < num_samples:
            current_batch = min(batch_size, num_samples - image_index)
            z = torch.randn(current_batch, latent_dim, device=device)
            images = generator(z).cpu()
            images = ((images + 1.0) / 2.0).clamp(0.0, 1.0)

            for image in images:
                TF.to_pil_image(image).save(output_dir / f"{image_index:06d}.png")
                image_index += 1

    return output_dir


def compute_fid_for_checkpoint(
    checkpoint_path: str | Path,
    real_root: str | Path,
    output_root: str | Path,
    num_samples: int,
    batch_size: int = 64,
    device: str = "cuda",
) -> dict:
    checkpoint_path = Path(checkpoint_path)
    checkpoint = _load_checkpoint(checkpoint_path)
    config = checkpoint["config"]
    image_size = int(config["data"]["image_size"])

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    torch_cache_dir = output_root / "torch_cache"
    torch_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_HOME"] = str(torch_cache_dir)

    real_cache_dir = output_root / f"real_cache_{image_size}"
    fake_dir = output_root / f"generated_{checkpoint_path.stem}"
    real_dir = _prepare_real_cache(real_root, real_cache_dir, image_size=image_size, limit=num_samples)
    fake_dir = _generate_fake_images(checkpoint, fake_dir, num_samples=num_samples, batch_size=batch_size, device=torch.device(device))

    fid_value = calculate_fid_given_paths(
        [str(real_dir), str(fake_dir)],
        batch_size=batch_size,
        device=device,
        dims=2048,
        num_workers=0,
    )

    result = {
        "checkpoint_path": str(checkpoint_path),
        "real_dir": str(real_dir),
        "fake_dir": str(fake_dir),
        "num_samples": num_samples,
        "batch_size": batch_size,
        "fid": float(fid_value),
    }

    result_path = output_root / f"fid_{checkpoint_path.stem}.json"
    with result_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    return result
