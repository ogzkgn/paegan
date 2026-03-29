"""Factories for datasets and dataloaders."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from data.celeba import CelebAImageFolderDataset


def build_image_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


def build_dataset(data_cfg: dict) -> CelebAImageFolderDataset:
    dataset_name = data_cfg["dataset_name"].lower()
    root = Path(data_cfg["root"])
    image_size = int(data_cfg["image_size"])

    if dataset_name != "celeba":
        raise ValueError(f"Unsupported dataset_name: {data_cfg['dataset_name']}")

    transform = build_image_transform(image_size)
    return CelebAImageFolderDataset(root=root, transform=transform)


def build_dataloader(
    data_cfg: dict,
    batch_size: int,
    seed: int,
    shuffle: bool = True,
    drop_last: bool = True,
) -> DataLoader:
    dataset = build_dataset(data_cfg)
    generator = torch.Generator().manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=int(data_cfg.get("num_workers", 0)),
        pin_memory=bool(data_cfg.get("pin_memory", False)),
        drop_last=drop_last,
        generator=generator,
    )
