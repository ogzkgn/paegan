"""Model components for PAE-GAN."""

from models.discriminator import CNNDiscriminator
from models.generator import ProgressiveGenerator

__all__ = ["CNNDiscriminator", "ProgressiveGenerator"]
