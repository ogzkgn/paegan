"""DCGAN-style generator with a single progressive attention module."""

from __future__ import annotations

import torch
from torch import nn

from models.attention import ProgressiveAttentionModule


class PAMGenerator(nn.Module):
    """DCGAN-style generator with one attention module at an intermediate resolution."""

    def __init__(
        self,
        latent_dim: int = 128,
        base_channels: int = 64,
        image_size: int = 32,
        out_channels: int = 3,
        attention_enabled: bool = True,
        attention_mode: str = "fixed",
        fixed_attention_type: str = "global",
        attention_resolution: int = 16,
        attention_num_heads: int = 4,
        progressive_attention_schedule: list[dict] | None = None,
        **_: dict,
    ) -> None:
        super().__init__()
        if image_size != 32:
            raise ValueError("This PAM-GAN implementation currently targets 32x32 outputs.")

        self.latent_dim = latent_dim
        self.attention_enabled = attention_enabled
        self.attention_mode = attention_mode
        self.fixed_attention_type = fixed_attention_type if attention_enabled else "none"
        self.attention_resolution = attention_resolution
        self.current_epoch = 1
        self.current_attention_type = self.fixed_attention_type
        self.progressive_attention_schedule = progressive_attention_schedule or [
            {"epoch_start": 1, "attention_type": "window_4"},
            {"epoch_start": 11, "attention_type": "window_8"},
            {"epoch_start": 21, "attention_type": "global"},
        ]

        c = base_channels
        self.project = nn.Sequential(
            nn.Linear(latent_dim, c * 8 * 4 * 4),
            nn.BatchNorm1d(c * 8 * 4 * 4),
            nn.ReLU(inplace=True),
        )

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(c * 8, c * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(c * 4),
            nn.ReLU(inplace=True),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(c * 4, c * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(c * 2),
            nn.ReLU(inplace=True),
        )
        self.attention = ProgressiveAttentionModule(c * 2, num_heads=attention_num_heads)
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(c * 2, c, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )
        self.to_rgb = nn.Sequential(
            nn.Conv2d(c, out_channels, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = epoch
        if not self.attention_enabled:
            self.current_attention_type = "none"
            return

        if self.attention_mode == "fixed":
            self.current_attention_type = self.fixed_attention_type
            return
        if self.attention_mode != "progressive":
            raise ValueError(f"Unsupported attention_mode: {self.attention_mode}")

        current_type = self.progressive_attention_schedule[0]["attention_type"]
        for item in self.progressive_attention_schedule:
            if epoch >= int(item["epoch_start"]):
                current_type = item["attention_type"]
        self.current_attention_type = current_type

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        bsz = z.size(0)
        x = self.project(z).view(bsz, -1, 4, 4)
        x = self.up1(x)
        x = self.up2(x)
        if x.shape[-1] != self.attention_resolution:
            raise ValueError(
                f"Attention module expected resolution {self.attention_resolution}, got {x.shape[-1]}"
            )
        x = self.attention(x, self.current_attention_type)
        x = self.up3(x)
        return self.to_rgb(x)


ProgressiveGenerator = PAMGenerator
