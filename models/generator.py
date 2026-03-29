"""Stage-based generator for PAE-GAN."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from models.attention import TransformerStageBlock


class ProgressiveGenerator(nn.Module):
    """Projects a latent vector into an 8x8 token map and upsamples by stage."""

    def __init__(
        self,
        latent_dim: int = 128,
        base_channels: int = 256,
        image_size: int = 32,
        stage_resolutions: list[int] | None = None,
        attention_schedule: list[str] | None = None,
        blocks_per_stage: int = 1,
        out_channels: int = 3,
        num_heads: int = 8,
        use_positional_embeddings: bool = True,
    ) -> None:
        super().__init__()
        if stage_resolutions is None:
            stage_resolutions = [8, 16, 32]
        if attention_schedule is None:
            attention_schedule = ["window_4", "window_4", "window_4"]
        if len(stage_resolutions) != len(attention_schedule):
            raise ValueError("stage_resolutions and attention_schedule must have the same length")
        if stage_resolutions[-1] != image_size:
            raise ValueError("Final stage resolution must match image_size")

        self.image_size = image_size
        self.base_channels = base_channels
        self.stage_resolutions = stage_resolutions
        self.use_positional_embeddings = use_positional_embeddings

        self.input_proj = nn.Linear(latent_dim, base_channels * stage_resolutions[0] * stage_resolutions[0])

        stages = []
        to_rgb_layers = []
        positional_embeddings = []
        for stage_idx, attention_type in enumerate(attention_schedule):
            resolution = stage_resolutions[stage_idx]
            blocks = [
                TransformerStageBlock(
                    dim=base_channels,
                    num_heads=num_heads,
                    attention_type=attention_type,
                )
                for _ in range(blocks_per_stage)
            ]
            stages.append(nn.Sequential(*blocks))
            positional_embeddings.append(
                nn.Parameter(torch.zeros(1, base_channels, resolution, resolution))
            )
            to_rgb_layers.append(
                nn.Sequential(
                    nn.GroupNorm(num_groups=8, num_channels=base_channels),
                    nn.SiLU(),
                    nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1),
                )
            )

        self.stages = nn.ModuleList(stages)
        self.positional_embeddings = nn.ParameterList(positional_embeddings)
        self.to_rgb_layers = nn.ModuleList(to_rgb_layers)

    def forward(self, z):
        initial_size = self.stage_resolutions[0]
        x = self.input_proj(z)
        x = x.view(z.size(0), self.base_channels, initial_size, initial_size)

        for stage_idx, resolution in enumerate(self.stage_resolutions):
            if stage_idx > 0:
                x = F.interpolate(x, size=(resolution, resolution), mode="nearest")
            if self.use_positional_embeddings:
                x = x + self.positional_embeddings[stage_idx]
            x = self.stages[stage_idx](x)

        rgb = self.to_rgb_layers[-1](x)
        return rgb.tanh()
