"""Concrete parametric bodies and the public registry."""

from .registry import load
from .smpl import SMPL, SMPLClassic
from .smplx import SMPLX, SMPLXClassic

__all__ = ["SMPL", "SMPLClassic", "SMPLX", "SMPLXClassic", "load"]
