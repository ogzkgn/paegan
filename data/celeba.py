"""CelebA dataset helpers."""

from __future__ import annotations

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


class CelebAImageFolderDataset(Dataset):
    """Loads image files from a flat CelebA directory."""

    VALID_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, root: str | Path, transform=None) -> None:
        self.root = Path(root)
        self.transform = transform

        if not self.root.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.root}")

        self.image_paths = sorted(
            path
            for path in self.root.iterdir()
            if path.is_file() and path.suffix.lower() in self.VALID_SUFFIXES
        )

        if not self.image_paths:
            raise RuntimeError(f"No supported image files found in: {self.root}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        image_path = self.image_paths[index]

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)

        return image
