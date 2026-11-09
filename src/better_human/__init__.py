"""Public BetterHuman API: families as data, identities baked once, poses streamed."""

from __future__ import annotations

from typing import Any

from . import families, io
from .core import FamilySpec, HumanModel, HumanState, Identity

__version__ = "0.1.0.dev0"

_LOADERS = {name: getattr(families, name) for name in families.__all__}
_MODEL_OPTIONS = {"density", "contact_frames", "stream_joints"}


def load(family: str, path: str | None = None, **options: Any) -> HumanModel:
    """``HumanModel`` of ``family`` (``"smpl"``, ``"smplx"``, ``"mhr"``, ...): loader and model options in one call."""
    try:
        loader = _LOADERS[family.lower()]
    except KeyError as error:
        raise ValueError(f"unknown family {family!r}; choose one of {sorted(_LOADERS)}") from error
    model_options = {key: options.pop(key) for key in list(options) if key in _MODEL_OPTIONS}
    return HumanModel.from_spec(loader(path, **options), **model_options)


__all__ = ["FamilySpec", "HumanModel", "HumanState", "Identity", "__version__", "families", "io", "load"]
