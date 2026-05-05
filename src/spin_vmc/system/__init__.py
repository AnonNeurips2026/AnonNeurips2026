"""Paper spin systems."""

from spin_vmc.system.base import BaseSystem, SpinSystem, SpinonSystem
from spin_vmc.system.system import ShastrySutherland, SquareHeisenberg

__all__ = [
    "BaseSystem",
    "ShastrySutherland",
    "SpinSystem",
    "SpinonSystem",
    "SquareHeisenberg",
]
