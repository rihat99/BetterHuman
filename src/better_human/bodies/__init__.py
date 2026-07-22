"""Concrete parametric bodies and the public registry."""

from .registry import load
from .smpl_family import SMPL, SMPLClassic, SMPLX, SMPLXClassic
from .mhr import MHR, MHRClassic

__all__ = ["MHR", "MHRClassic", "SMPL", "SMPLClassic", "SMPLX", "SMPLXClassic", "load"]
