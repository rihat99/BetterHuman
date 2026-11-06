"""Frame sets: named mesh points, each a vertex id and an optional parent joint, stored as versioned JSON.

A frame-set file (format ``better_human.frame_set``, version 1; spec ``docs/reference/formats/frame_set.md``)::

    {"format": "better_human.frame_set", "format_version": 1,
     "frames": {"heel_left": {"vertex": 8924, "joint": null}, "knee_left": {"vertex": 3673, "joint": "left_knee"}}}

A ``null`` joint means the vertex's dominant skinning joint. An unversioned file (version 0, ``{name: vertex |
{"vertex", "joint"}}``) migrates on read. The functions here return and take the mapping
``Body(contact_frames=...)`` accepts: name -> ``(vertex_id, joint_name | None)``, in authored order. Vertex ids are
range-checked where the mesh is known, when the frames are attached to a body.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from importlib import resources
from pathlib import Path
from typing import Any

from better_robot.exceptions import UnknownNameError

from .formats import FRAME_SET, FormatError, header, upgrade


def read_frame_set(path: str | Path) -> dict[str, tuple[int, str | None]]:
    """The frames of a frame-set file: name -> ``(vertex_id, joint_name | None)``, in authored order.

    Raises :class:`~better_human.io.FormatError` for another format, a later version or a malformed entry.
    """
    path = Path(path)
    return _frames(json.loads(path.read_text()), source=str(path))


def write_frame_set(path: str | Path, frames: Mapping[str, tuple[int, str | None]]) -> None:
    """Write ``frames`` (name -> ``(vertex_id, joint_name | None)``) as a version-1 frame-set file.

    The entries pass the reader's checks before anything is written, and the file is replaced atomically.
    """
    path = Path(path)
    payload = {
        **header(FRAME_SET),
        "frames": {str(name): {"vertex": vertex, "joint": joint} for name, (vertex, joint) in frames.items()},
    }
    _frames(payload, source=str(path))
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def shipped_frame_set(name: str) -> dict[str, tuple[int, str | None]]:
    """A frame set shipped with BetterHuman, e.g. ``"smplx_contacts"`` (35 SMPL-X contact points).

    Raises :class:`~better_robot.exceptions.UnknownNameError` listing the shipped names for an unknown ``name``.
    """
    assets = resources.files(__package__) / "assets"
    shipped = sorted(entry.name.removesuffix(".json") for entry in assets.iterdir() if entry.name.endswith(".json"))
    if name not in shipped:
        raise UnknownNameError(f"unknown shipped frame set {name!r}; choose one of {shipped}")
    return _frames(json.loads((assets / f"{name}.json").read_text()), source=f"shipped frame set {name!r}")


def _frames(data: Mapping[str, Any], *, source: str) -> dict[str, tuple[int, str | None]]:
    frames = upgrade(data, format=FRAME_SET, source=source).get("frames")
    if not isinstance(frames, Mapping):
        raise FormatError(f"{source}: 'frames' must map each frame name to {{vertex, joint}}, got {frames!r}")
    return {name: _entry(name, value, source) for name, value in frames.items()}


def _entry(name: str, value: object, source: str) -> tuple[int, str | None]:
    if not isinstance(value, Mapping):
        raise FormatError(f"{source}: frame {name!r} must be {{vertex, joint}}, got {value!r}")
    vertex, joint = value.get("vertex"), value.get("joint")
    if isinstance(vertex, bool) or not isinstance(vertex, int):
        raise FormatError(f"{source}: frame {name!r}: vertex id {vertex!r} is not an int")
    if joint is not None and not isinstance(joint, str):
        raise FormatError(f"{source}: frame {name!r}: joint {joint!r} is not a joint name or null")
    return vertex, joint
