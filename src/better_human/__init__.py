"""Public BetterHuman API: families as data, identities baked once, poses streamed."""

from __future__ import annotations

from typing import Any

from . import families
from .core import Body, FamilySpec, HumanState, Identity

__version__ = "0.1.0.dev0"

_LOADERS = {name: getattr(families, name) for name in families.__all__}
_BODY_OPTIONS = {"mass", "density", "contact_frames", "stream_joints"}


def load(family: str, path: str | None = None, **options: Any) -> Body:
    """``Body`` of ``family`` (``"smpl"``, ``"smplx"``, ``"mhr"``, ...): loader options and ``Body`` options in one call."""
    try:
        loader = _LOADERS[family.lower()]
    except KeyError as error:
        raise ValueError(f"unknown family {family!r}; choose one of {sorted(_LOADERS)}") from error
    body_options = {key: options.pop(key) for key in list(options) if key in _BODY_OPTIONS}
    return Body(loader(path, **options), **body_options)


__all__ = ["Body", "FamilySpec", "HumanState", "Identity", "__version__", "families", "load"]
