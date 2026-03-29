"""Single attention module for PAM-GAN."""

from __future__ import annotations

import torch
from torch import nn


class ProgressiveAttentionModule(nn.Module):
    """Applies self-attention over a feature map using local or global context."""

    def __init__(self, channels: int, num_heads: int = 4) -> None:
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(f"channels={channels} must be divisible by num_heads={num_heads}")

        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5

        self.norm = nn.BatchNorm2d(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor, attention_type: str) -> torch.Tensor:
        if attention_type == "none":
            return x

        bsz, channels, height, width = x.shape
        h = self.norm(x)

        if attention_type == "global":
            out = self._apply_attention(h.flatten(2).transpose(1, 2))
            out = out.transpose(1, 2).reshape(bsz, channels, height, width)
        elif attention_type.startswith("window_"):
            window = int(attention_type.split("_", maxsplit=1)[1])
            out = self._apply_window_attention(h, window)
        else:
            raise ValueError(f"Unsupported attention type: {attention_type}")

        return x + out

    def _apply_window_attention(self, x: torch.Tensor, window: int) -> torch.Tensor:
        bsz, channels, height, width = x.shape
        if height % window != 0 or width % window != 0:
            raise ValueError(f"window={window} must divide feature map {height}x{width}")

        x = x.reshape(
            bsz,
            channels,
            height // window,
            window,
            width // window,
            window,
        )
        x = x.permute(0, 2, 4, 3, 5, 1).reshape(-1, window * window, channels)
        x = self._apply_attention(x)
        x = x.reshape(bsz, height // window, width // window, window, window, channels)
        x = x.permute(0, 5, 1, 3, 2, 4).reshape(bsz, channels, height, width)
        return x

    def _apply_attention(self, tokens: torch.Tensor) -> torch.Tensor:
        bsz, num_tokens, channels = tokens.shape
        spatial = int(num_tokens ** 0.5)
        qkv = self.qkv(tokens.transpose(1, 2).reshape(bsz, channels, spatial, spatial))
        qkv = qkv.flatten(2).transpose(1, 2)
        qkv = qkv.reshape(bsz, num_tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(dim=0)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(bsz, num_tokens, channels)
        out = self.proj(out.transpose(1, 2).reshape(bsz, channels, spatial, spatial))
        return out.flatten(2).transpose(1, 2)
