"""Public BetterHuman API."""

from .bodies import SMPL, SMPLX, load
from .core import ParametricBody

__version__ = "0.1.0.dev0"

__all__ = ["ParametricBody", "SMPL", "SMPLX", "__version__", "load"]
