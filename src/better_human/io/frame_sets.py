"""Frame sets: named mesh points, each a vertex id and an optional parent joint, stored as versioned JSON.

A frame-set file (format ``better_human.frame_set``, version 1; spec ``docs/reference/formats/frame_set.md``)::

    {"format": "better_human.frame_set", "format_version": 1,
     "frames": {"heel_left": {"vertex_id": 8924, "joint_name": null},
                "knee_left": {"vertex_id": 3673, "joint_name": "left_knee"}}}

A ``null`` joint name means the vertex's dominant skinning joint. An unversioned file (version 0, ``{name: vertex |
{"vertex", "joint"}}``) migrates on read; a file without ``format`` is version 0 only when every value is such an
entry. The functions here return and take the mapping ``Body(contact_frames=...)`` accepts: name -> ``(vertex_id,
joint_name | None)``, in authored order. Vertex ids are range-checked where the mesh is known, when the frames are
attached to a body.
"""

from __future__ import annotations

import json
import numbers
from collections.abc import Mapping
from importlib import resources
from pathlib import Path

from better_robot.exceptions import ArgumentError, UnknownNameError

from .formats import FRAME_SET, FormatError, header, upgrade

#: How to fix a file that is not frame-shaped; the end of each such message.
FIX = "Write frame sets with better_human.io.write_frame_set (spec: docs/reference/formats/frame_set.md)."


def read_frame_set(path: str | Path) -> dict[str, tuple[int, str | None]]:
    """The frames of a frame-set file: name -> ``(vertex_id, joint_name | None)``, in authored order.

    Raises :class:`~better_human.io.FormatError` for a file that is not JSON, another format, a later version, or a
    file or entry that is not frame-shaped.
    """
    path = Path(path)
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise FormatError(f"{path}: not a JSON file ({error}). {FIX}") from error
    return _frames(data, source=str(path))


def write_frame_set(path: str | Path, frames: Mapping[str, tuple[int, str | None]]) -> None:
    """Write ``frames`` (name -> ``(vertex_id, joint_name | None)``) as a version-1 frame-set file.

    A vertex id is any integer >= 0 (``int``, ``numpy.int64``, ...). The entries pass the reader's checks before
    anything is written, and the file is replaced atomically. Raises :class:`~better_robot.exceptions.ArgumentError`
    for a value that is not a ``(vertex_id, joint_name | None)`` pair.
    """
    path = Path(path)
    for name, value in frames.items():
        if not isinstance(value, tuple | list) or len(value) != 2:
            raise ArgumentError(
                f"write_frame_set: frame {name!r} must be a pair (vertex_id, joint_name | None), got {value!r}"
            )
    checked = {
        str(name): _entry(str(name), {"vertex_id": vertex, "joint_name": joint}, str(path))
        for name, (vertex, joint) in frames.items()
    }
    payload = {
        **header(FRAME_SET),
        "frames": {name: {"vertex_id": vertex, "joint_name": joint} for name, (vertex, joint) in checked.items()},
    }
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
        raise UnknownNameError("shipped frame set", name, shipped)
    return _frames(json.loads((assets / f"{name}.json").read_text()), source=f"shipped frame set {name!r}")


def _frames(data: object, *, source: str) -> dict[str, tuple[int, str | None]]:
    if not isinstance(data, Mapping):
        raise FormatError(f"{source}: a frame set is a JSON object, got {type(data).__name__}. {FIX}")
    if "format" not in data and not all(_is_v0_entry(value) for value in data.values()):
        raise FormatError(
            f"{source}: names no format and is not a version-0 frame set ({{name: vertex | {{vertex, joint}}}}). {FIX}"
        )
    frames = upgrade(data, format=FRAME_SET, source=source).get("frames")
    if not isinstance(frames, Mapping):
        raise FormatError(
            f"{source}: 'frames' must map each frame name to {{vertex_id, joint_name}}, got {frames!r}. {FIX}"
        )
    return {name: _entry(name, value, source) for name, value in frames.items()}


def _is_v0_entry(value: object) -> bool:
    """A version-0 entry: a bare vertex id or ``{"vertex", "joint"}``."""
    return (isinstance(value, int) and not isinstance(value, bool)) or (
        isinstance(value, Mapping) and "vertex" in value
    )


def _entry(name: str, value: object, source: str) -> tuple[int, str | None]:
    if not isinstance(value, Mapping):
        raise FormatError(f"{source}: frame {name!r} must be {{vertex_id, joint_name}}, got {value!r}. {FIX}")
    vertex, joint = value.get("vertex_id"), value.get("joint_name")
    if isinstance(vertex, bool) or not isinstance(vertex, numbers.Integral) or vertex < 0:
        raise FormatError(f"{source}: frame {name!r}: vertex id {vertex!r} is not an int >= 0")
    if joint is not None and not isinstance(joint, str):
        raise FormatError(f"{source}: frame {name!r}: joint_name {joint!r} is not a string or null")
    return int(vertex), joint
