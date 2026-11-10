"""The studio's files under its data root: segmentations and poses in their versioned JSON formats, and frame sets.

A segmentation file (format ``better_human.studio.segmentation``, version 1; spec
``docs/reference/formats/studio_segmentation.md``) maps each part name to its vertex ids; a pose file (format
``better_human.studio.pose``, version 1; spec ``docs/reference/formats/studio_pose.md``) holds a configuration ``q``
with the names of the model joints it follows. Frame sets are ``better_human.io``'s format. The readers check the
format and the shape of each entry; vertex ids are range-checked against a mesh by :func:`check_vertex_ids`.
"""

from __future__ import annotations

import json
import numbers
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any, NamedTuple, final

from better_robot.exceptions import ArgumentError, UnknownNameError

from better_human.io import FormatError, header, upgrade

#: The studio's file formats (``docs/reference/formats/studio_*.md``), versioned in ``better_human.io``.
SEGMENTATION_FORMAT = "better_human.studio.segmentation"
POSE_FORMAT = "better_human.studio.pose"

#: How to fix a studio file the readers refuse; the end of each such message.
_FIX = "re-save it in the studio"

#: The families whose built-in vertex segmentation ships in ``better_human.studio.assets``.
BUILTIN_SEGMENTATIONS = ("smpl", "smplx")


@final
@dataclass(frozen=True)
class StudioPaths:
    """Directory layout for authored data under the studio's data root."""

    #: The data root; each writer creates the folders it writes into.
    root: Path

    def segmentation_dir(self, model_key: str) -> Path:
        """The segmentation directory of a mesh-topology key."""
        return self.root / "segmentations" / model_key

    def frame_dir(self, model_key: str) -> Path:
        """The frame-set directory of a mesh-topology key."""
        return self.root / "frames" / model_key

    def pose_dir(self, model_key: str) -> Path:
        """The pose directory of a mesh-topology key."""
        return self.root / "poses" / model_key


@final
class Pose(NamedTuple):
    """A saved configuration: ``q (nq,)`` and the names of the model joints, in the order ``q`` follows them."""

    joint_names: tuple[str, ...]
    q: tuple[float, ...]


def model_key(name: str, *, gender: str | None = None, lod: int | None = None) -> str:
    """The key of a distinct mesh topology: ``"smpl_neutral"``, ``"smplx_female"``, ``"mhr_lod1"``."""
    if lod is not None:
        return f"{name}_lod{lod}"
    if gender is not None:
        return f"{name}_{gender}"
    return name


def list_saved(directory: Path) -> dict[str, Path]:
    """Saved ``*.json`` files of ``directory`` by stem, sorted by name; empty when the directory does not exist."""
    if not directory.is_dir():
        return {}
    return {path.stem: path for path in sorted(directory.glob("*.json"))}


def read_segmentation(path: str | Path) -> dict[str, list[int]]:
    """The parts of a segmentation file: part name -> vertex ids, in authored order.

    An unversioned file of the pre-format studio (version 0, the bare ``{part name: vertex ids}``) migrates on read.
    Raises :class:`~better_human.io.FormatError` for a file that is not JSON, another format, a later version, or a
    part that is not a non-empty list of vertex ids.
    """
    path = Path(path)
    return _parts(_json(path), source=str(path))


def write_segmentation(path: str | Path, parts: Mapping[str, Sequence[int]]) -> None:
    """Write ``parts`` (part name -> vertex ids) as a version-1 segmentation file, creating its folder.

    Raises :class:`~better_human.io.FormatError` for a part that is not a non-empty sequence of vertex ids.
    """
    path = Path(path)
    payload = {str(name): list(ids) for name, ids in parts.items()}
    _write(path, {**header(SEGMENTATION_FORMAT), "parts": _parts_of(payload, source=str(path), fix="")})


def builtin_segmentation(family: str) -> dict[str, list[int]]:
    """The vertex segmentation shipped for ``family`` (``"smpl"``: 24 parts, ``"smplx"``: 27 parts).

    Raises :class:`~better_robot.exceptions.UnknownNameError` for another family.
    """
    if family not in BUILTIN_SEGMENTATIONS:
        raise UnknownNameError("built-in segmentation", family, valid=BUILTIN_SEGMENTATIONS)
    resource = resources.files("better_human.studio.assets") / f"{family}_vert_segmentation.json"
    return _parts(json.loads(resource.read_text()), source=f"built-in segmentation {family!r}")


def read_pose(path: str | Path) -> Pose:
    """The pose of a pose file.

    Raises :class:`~better_human.io.FormatError` for a file that is not JSON, another format, a later version, or
    ``joint_names`` / ``q`` that are not a list of strings / numbers.
    """
    path = Path(path)
    fields = upgrade(_json(path), format=POSE_FORMAT, source=str(path))
    joint_names, q = fields.get("joint_names"), fields.get("q")
    if not isinstance(joint_names, list) or not all(isinstance(name, str) for name in joint_names):
        raise FormatError(f"{path}: 'joint_names' must be a list of strings, got {joint_names!r}; {_FIX}")
    if not isinstance(q, list) or not all(_is_number(value) for value in q):
        raise FormatError(f"{path}: 'q' must be a list of numbers, got {q!r}; {_FIX}")
    return Pose(joint_names=tuple(joint_names), q=tuple(float(value) for value in q))


def write_pose(path: str | Path, pose: Pose) -> None:
    """Write ``pose`` as a version-1 pose file, creating its folder."""
    payload = {"joint_names": list(pose.joint_names), "q": [float(value) for value in pose.q]}
    _write(Path(path), {**header(POSE_FORMAT), **payload})


def check_vertex_ids(vertex_ids: Iterable[int], num_vertices: int, *, source: str) -> None:
    """Raise :class:`~better_robot.exceptions.ArgumentError` naming ``source`` unless every id is below ``num_vertices``."""
    out_of_range = sorted({int(vertex) for vertex in vertex_ids if vertex >= num_vertices})
    if out_of_range:
        raise ArgumentError(
            f"{source}: vertex ids {out_of_range[:5]} are out of range for a mesh of {num_vertices} vertices; "
            "load data authored on this body"
        )


def frame_config_snippet(frames: Mapping[str, tuple[int, str | None]]) -> str:
    """A frame set as a paste-ready ``contact_frames={...}`` argument of ``HumanModel.from_spec``.

    :param frames: frame name -> ``(vertex_id, joint_name | None)``.
    """
    rows = "".join(f"\n    {name!r}: ({vertex}, {joint!r})," for name, (vertex, joint) in frames.items())
    return f"contact_frames={{{rows}\n}}"


def _json(path: Path) -> Any:
    if not path.is_file():
        raise ArgumentError(f"{path} is not a file; pass a file the studio saved")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise FormatError(f"{path}: not a JSON file ({error}); {_FIX}") from error


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    # Replaced atomically, as io.write_frame_set does: a crash mid-write leaves the old file, never half a new one.
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _parts(data: Any, *, source: str) -> dict[str, list[int]]:
    if not isinstance(data, Mapping):
        raise FormatError(f"{source}: a segmentation is a JSON object, got {type(data).__name__}; {_FIX}")
    if "format" not in data and not all(isinstance(ids, list) for ids in data.values()):
        raise FormatError(
            f"{source}: names no format and is not a version-0 segmentation ({{part name: vertex ids}}); {_FIX}"
        )
    parts = upgrade(data, format=SEGMENTATION_FORMAT, source=source).get("parts")
    return _parts_of(parts, source=source, fix=f"; {_FIX}")


def _parts_of(parts: Any, *, source: str, fix: str) -> dict[str, list[int]]:
    """``parts`` checked; ``fix`` ends each refusal (a reader names how to fix the file, a writer nothing)."""
    if not isinstance(parts, Mapping):
        raise FormatError(f"{source}: 'parts' must map each part name to its vertex ids, got {parts!r}{fix}")
    for name, ids in parts.items():
        if not isinstance(ids, list) or not ids or not all(_is_vertex_id(vertex) for vertex in ids):
            raise FormatError(f"{source}: part {name!r} must be a non-empty list of vertex ids (ints >= 0){fix}")
    return {str(name): [int(vertex) for vertex in ids] for name, ids in parts.items()}


def _is_vertex_id(value: object) -> bool:
    return isinstance(value, numbers.Integral) and not isinstance(value, bool) and value >= 0


def _is_number(value: object) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, bool)
