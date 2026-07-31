"""Interactive viser studio for inspecting and authoring parametric bodies.

The public surface is :class:`~better_human.viz.studio.app.HumanStudio` and the
:func:`~better_human.viz.studio.app.launch` helper. viser is an optional
dependency (the ``studio`` extra), so ``HumanStudio`` and ``launch`` load
lazily: importing this package (or its pure submodules ``persistence``,
``posing``, ``picking``) never imports viser until the app itself is requested.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .app import HumanStudio, launch

__all__ = ["HumanStudio", "launch"]


def __getattr__(name: str) -> Any:
    if name in ("HumanStudio", "launch"):
        from . import app  # noqa: PLC0415 - lazy so the pure submodules import without viser

        return getattr(app, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
