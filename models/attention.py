"""Attention modules for the generator."""

from __future__ import annotations

import math

import torch
from torch import nn


class StageAttention(nn.Module):
    """Self-attention that can operate globally or within local windows."""

    def __init__(self, dim: int, num_heads: int, attention_type: str) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.attention_type = attention_type
        self.window_size = self._parse_window_size(attention_type)

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    @staticmethod
    def _parse_window_size(attention_type: str) -> int | None:
        if attention_type == "global":
            return None
        if attention_type.startswith("window_"):
            return int(attention_type.split("_", maxsplit=1)[1])
        raise ValueError(f"Unsupported attention_type: {attention_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, channels, height, width = x.shape
        tokens = x.flatten(2).transpose(1, 2)

        if self.window_size is None:
            attended = self._apply_attention(tokens)
        else:
            if height != width:
                raise ValueError("Window attention expects square spatial inputs.")
            if height % self.window_size != 0 or width % self.window_size != 0:
                raise ValueError(
                    f"window_size={self.window_size} must divide spatial size {height}x{width}"
                )
            attended = self._apply_window_attention(tokens, height, width)

        return attended.transpose(1, 2).reshape(bsz, channels, height, width)

    def _apply_window_attention(self, tokens: torch.Tensor, height: int, width: int) -> torch.Tensor:
        bsz, _, channels = tokens.shape
        window = self.window_size

        x = tokens.transpose(1, 2).reshape(bsz, channels, height, width)
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
        return x.flatten(2).transpose(1, 2)

    def _apply_attention(self, tokens: torch.Tensor) -> torch.Tensor:
        bsz, num_tokens, _ = tokens.shape
        qkv = self.qkv(tokens)
        qkv = qkv.reshape(bsz, num_tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(dim=0)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(bsz, num_tokens, self.dim)
        return self.proj(out)


class TransformerStageBlock(nn.Module):
    """A minimal pre-norm transformer block over image tokens."""

    def __init__(self, dim: int, num_heads: int, attention_type: str, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)

        self.norm1 = nn.LayerNorm(dim)
        self.attn = StageAttention(dim=dim, num_heads=num_heads, attention_type=attention_type)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, channels, height, width = x.shape

        tokens = x.flatten(2).transpose(1, 2)
        normed = self.norm1(tokens).transpose(1, 2).reshape(bsz, channels, height, width)
        x = x + self.attn(normed)

        tokens = x.flatten(2).transpose(1, 2)
        x = x + self.mlp(self.norm2(tokens)).transpose(1, 2).reshape(bsz, channels, height, width)
        return x
