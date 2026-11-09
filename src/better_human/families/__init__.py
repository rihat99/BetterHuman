"""Human families: loaders that turn weight files into :class:`~better_human.core.FamilySpec`."""

from .mhr import mhr
from .regressed_rig import RegressedRig, drop_joints
from .smpl_family import flame, mano, smpl, smplh, smplx
from .soma import soma

__all__ = ["RegressedRig", "drop_joints", "flame", "mano", "mhr", "smpl", "smplh", "smplx", "soma"]
