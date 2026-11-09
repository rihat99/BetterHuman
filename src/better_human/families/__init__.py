"""Human families: loaders that turn weight files into :class:`~better_human.core.FamilySpec`."""

from .mhr_family import mhr
from .mhr_rig import MHRRig
from .regressed_rig import RegressedRig, drop_joints
from .smpl_family import flame, mano, smpl, smplh, smplx
from .soma_family import soma
from .soma_rig import SOMARig

__all__ = ["MHRRig", "RegressedRig", "SOMARig", "drop_joints", "flame", "mano", "mhr", "smpl", "smplh", "smplx", "soma"]
