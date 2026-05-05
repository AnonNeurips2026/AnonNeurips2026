"""Network factories bundled with the anonymous review package."""

from nqs_nets.net import ViT as ViT
from nqs_nets.net.wrappers import ViTNd as ViTNd

__all__ = ["ViT", "ViTNd"]
