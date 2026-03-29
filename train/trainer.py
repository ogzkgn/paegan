"""End-to-end GAN training loop."""

from __future__ import annotations

import time
from pathlib import Path

import torch
from torch import optim
from torch.amp import GradScaler, autocast

from data import build_dataloader
from models import CNNDiscriminator, ProgressiveGenerator
from train.losses import discriminator_hinge_loss, generator_hinge_loss
from train.utils import (
    append_metrics,
    prepare_run_directories,
    resolve_device,
    save_checkpoint,
    save_sample_grid,
    set_seed,
)


def build_models(config: dict, device: torch.device) -> tuple[ProgressiveGenerator, CNNDiscriminator]:
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
    ).to(device)

    discriminator = CNNDiscriminator(
        in_channels=int(data_cfg["channels"]),
        base_channels=64,
    ).to(device)

    return generator, discriminator


def train_gan(config: dict, max_steps: int | None = None) -> Path:
    seed = int(config["seed"])
    train_cfg = config["train"]
    model_cfg = config["model"]
    eval_cfg = config["eval"]

    set_seed(seed)
    device = resolve_device()
    print({"selected_device": str(device)}, flush=True)
    use_amp = bool(train_cfg.get("use_amp", device.type == "cuda"))
    amp_dtype = torch.float16 if device.type == "cuda" else torch.float32
    scaler_g = GradScaler(device="cuda", enabled=use_amp and device.type == "cuda")
    scaler_d = GradScaler(device="cuda", enabled=use_amp and device.type == "cuda")

    dataloader = build_dataloader(
        data_cfg=config["data"],
        batch_size=int(train_cfg["batch_size"]),
        seed=seed,
        shuffle=True,
        drop_last=True,
    )
    generator, discriminator = build_models(config, device)

    optimizer_g = optim.Adam(
        generator.parameters(),
        lr=float(train_cfg["lr_g"]),
        betas=tuple(float(beta) for beta in train_cfg["betas"]),
    )
    optimizer_d = optim.Adam(
        discriminator.parameters(),
        lr=float(train_cfg["lr_d"]),
        betas=tuple(float(beta) for beta in train_cfg["betas"]),
    )

    directories = prepare_run_directories(
        base_dir=eval_cfg["save_dir"],
        experiment_name=config["experiment_name"],
    )
    metrics_path = directories["logs"] / "metrics.jsonl"

    fixed_noise = torch.randn(16, int(model_cfg["latent_dim"]), device=device)
    global_step = 0
    epochs = int(train_cfg["epochs"])
    n_critic = int(train_cfg.get("n_critic", 1))
    log_every = int(train_cfg["log_every"])
    sample_every = int(train_cfg["sample_every"])
    checkpoint_every = int(train_cfg["checkpoint_every"])

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        for real_images in dataloader:
            global_step += 1
            real_images = real_images.to(device, non_blocking=True)
            batch_size = real_images.size(0)

            z = torch.randn(batch_size, int(model_cfg["latent_dim"]), device=device)
            with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                fake_images = generator(z).detach()

            optimizer_d.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                real_logits = discriminator(real_images)
                fake_logits = discriminator(fake_images)
                loss_d = discriminator_hinge_loss(real_logits, fake_logits)
            scaler_d.scale(loss_d).backward()
            scaler_d.step(optimizer_d)
            scaler_d.update()

            loss_g = None
            if global_step % n_critic == 0:
                z = torch.randn(batch_size, int(model_cfg["latent_dim"]), device=device)
                optimizer_g.zero_grad(set_to_none=True)
                with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    generated_images = generator(z)
                    generated_logits = discriminator(generated_images)
                    loss_g = generator_hinge_loss(generated_logits)
                scaler_g.scale(loss_g).backward()
                scaler_g.step(optimizer_g)
                scaler_g.update()

            if global_step % log_every == 0 or global_step == 1:
                metrics = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "loss_d": float(loss_d.item()),
                    "loss_g": None if loss_g is None else float(loss_g.item()),
                    "device": str(device),
                    "use_amp": use_amp,
                }
                if torch.cuda.is_available():
                    metrics["gpu_memory_mb"] = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                append_metrics(metrics_path, metrics)
                print(metrics, flush=True)

            if global_step % sample_every == 0 or global_step == 1:
                sample_path = directories["samples"] / f"step_{global_step:06d}.png"
                save_sample_grid(
                    generator=generator,
                    fixed_noise=fixed_noise,
                    sample_path=sample_path,
                    device=device,
                )

            if max_steps is not None and global_step >= max_steps:
                checkpoint_path = directories["checkpoints"] / f"step_{global_step:06d}.pt"
                save_checkpoint(
                    checkpoint_path=checkpoint_path,
                    generator=generator,
                    discriminator=discriminator,
                    optimizer_g=optimizer_g,
                    optimizer_d=optimizer_d,
                    epoch=epoch,
                    global_step=global_step,
                    config=config,
                )
                return directories["run_root"]

        if epoch % checkpoint_every == 0:
            checkpoint_path = directories["checkpoints"] / f"epoch_{epoch:03d}.pt"
            save_checkpoint(
                checkpoint_path=checkpoint_path,
                generator=generator,
                discriminator=discriminator,
                optimizer_g=optimizer_g,
                optimizer_d=optimizer_d,
                epoch=epoch,
                global_step=global_step,
                config=config,
            )

        print(
            {
                "epoch": epoch,
                "global_step": global_step,
                "epoch_time_sec": round(time.time() - epoch_start, 2),
            },
            flush=True,
        )

    return directories["run_root"]
