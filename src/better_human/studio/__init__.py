"""HumanStudio: an optional browser application for posing, shaping, segmenting and frame-editing a body (record 0009).

``HumanStudio`` and ``launch`` are the public names; every submodule is the app's own. The app needs the ``studio``
extra (viser through BetterRobot's viewer), so the two load lazily: importing this package never imports viser.
Run it with ``python -m better_human.studio --data-root DIR``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .app import HumanStudio, launch

__all__ = ["HumanStudio", "launch"]


def __getattr__(name: str) -> Any:
    if name in ("HumanStudio", "launch"):
        from . import app  # noqa: PLC0415 - lazy so the package imports without viser

        return getattr(app, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
