"""Body families: loaders that turn weight files into :class:`~better_human.core.FamilySpec`."""

from .mhr import mhr
from .smpl_family import flame, mano, smpl, smplh, smplx
from .soma import soma

__all__ = ["flame", "mano", "mhr", "smpl", "smplh", "smplx", "soma"]
