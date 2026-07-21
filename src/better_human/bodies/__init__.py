"""Concrete parametric bodies and the public registry."""

from .registry import load
from .mhr import MHR, MHRClassic
from .smpl import SMPL, SMPLClassic
from .smplx import SMPLX, SMPLXClassic

__all__ = ["MHR", "MHRClassic", "SMPL", "SMPLClassic", "SMPLX", "SMPLXClassic", "load"]
