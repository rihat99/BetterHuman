"""Body families: loaders that turn weight files into :class:`~better_human.core.FamilySpec`."""

from .smpl_family import flame, mano, smpl, smplh, smplx

__all__ = ["flame", "mano", "smpl", "smplh", "smplx"]
