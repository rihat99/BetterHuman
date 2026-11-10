"""Public BetterHuman API: families as data, identities baked once, poses streamed."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Literal

import torch
from better_robot.exceptions import UnknownNameError

from . import families, io
from .core import FamilySpec, HumanModel, HumanState, Identity
from .core.classic import from_classic, to_classic

__version__ = "0.1.0.dev0"

_LOADERS: dict[str, Callable[..., FamilySpec]] = {
    "flame": families.flame,
    "mano": families.mano,
    "mhr": families.mhr,
    "smpl": families.smpl,
    "smplh": families.smplh,
    "smplx": families.smplx,
    "soma": families.soma,
}


def load(
    family: Literal["flame", "mano", "mhr", "smpl", "smplh", "smplx", "soma"],
    path: str | Path | None = None,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
    density: str | float | torch.Tensor | None = "Dempster",
    contact_frames: Mapping[str, tuple[int, str | None]] | None = None,
    stream_joints: bool = True,
) -> HumanModel:
    """``HumanModel`` of ``family`` with the family's default options, built in ``dtype`` on ``device``.

    ``path`` is the weight file (default: found under ``$BETTERHUMAN_MODELS_DIR``). ``density``, ``contact_frames``
    and ``stream_joints`` are :meth:`HumanModel.from_spec`'s. Family options (gender, ``num_betas``, ...) go
    through ``HumanModel.from_spec(families.smplx(path, gender=...), ...)`` (decision record 0006). Raises
    :class:`~better_robot.exceptions.UnknownNameError` for an unknown family and
    :class:`~better_robot.exceptions.DtypeMismatchError` (from the family loader) for a ``dtype`` other than
    ``torch.float32`` or ``torch.float64``, the floating dtypes BetterRobot computes in.
    """
    if family not in _LOADERS:
        raise UnknownNameError("family", family, valid=list(_LOADERS))
    spec = _LOADERS[family](path, dtype=dtype, device=device)
    return HumanModel.from_spec(spec, density=density, contact_frames=contact_frames, stream_joints=stream_joints)


__all__ = [
    "FamilySpec",
    "HumanModel",
    "HumanState",
    "Identity",
    "__version__",
    "families",
    "from_classic",
    "io",
    "load",
    "to_classic",
]
