"""HumanStudio: an optional browser application for posing, shaping, segmenting and frame-editing a body.

``HumanStudio`` is the public name; every submodule is the app's own. The app needs the ``studio`` extra (viser
through BetterRobot's viewer), so it loads lazily: importing this package never imports viser.
Run it with ``python -m better_human.studio --data-root DIR``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .app import HumanStudio

__all__ = ["HumanStudio"]


def __getattr__(name: str) -> Any:
    if name == "HumanStudio":
        from . import app  # noqa: PLC0415 - lazy so the package imports without viser

        return app.HumanStudio
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
