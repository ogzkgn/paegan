"""CNN discriminator for PAE-GAN."""

from __future__ import annotations

from torch import nn
from torch.nn.utils import spectral_norm


def sn_conv2d(*args, **kwargs) -> nn.Conv2d:
    return spectral_norm(nn.Conv2d(*args, **kwargs))


class CNNDiscriminator(nn.Module):
    """A compact spectral-normalized discriminator for 32x32 images."""

    def __init__(self, in_channels: int = 3, base_channels: int = 64) -> None:
        super().__init__()
        c = base_channels

        self.features = nn.Sequential(
            sn_conv2d(in_channels, c, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            sn_conv2d(c, c * 2, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            sn_conv2d(c * 2, c * 4, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            sn_conv2d(c * 4, c * 8, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.head = spectral_norm(nn.Linear(c * 8 * 2 * 2, 1))

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.head(x)
