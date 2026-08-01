"""File-system and packaged-asset I/O for studio-authored body data.

Round-trips vertex segmentations, named frame sets, and poses as JSON under a
``studio_data`` root, and loads the built-in segmentations shipped with
:mod:`better_human.bodies.assets`. Pure serialization with range and type
validation on load — no viser or model dependencies.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import torch

from better_human.core.contact_frames import parse_frame_set

_ASSET_PACKAGE = "better_human.bodies.assets"


@dataclass(frozen=True)
class StudioPaths:
    """Directory layout for authored data under a single ``studio_data`` root."""

    #: Root directory, created lazily by the individual save helpers.
    root: Path

    def segmentation_dir(self, model_key: str) -> Path:
        """Return the segmentation directory for a mesh-topology key."""
        return self.root / "segmentations" / model_key

    def frame_dir(self, model_key: str) -> Path:
        """Return the frame-set directory for a mesh-topology key."""
        return self.root / "frames" / model_key

    def pose_dir(self, model_key: str) -> Path:
        """Return the pose directory for a mesh-topology key."""
        return self.root / "poses" / model_key


def model_key(name: str, *, gender: str | None = None, lod: int | None = None) -> str:
    """Build the key identifying a distinct mesh topology.

    :param name: model family, one of ``"smpl" | "smplx" | "mhr"``.
    :param gender: SMPL-family gender, e.g. ``"neutral"``.
    :param lod: MHR level of detail.
    :returns: e.g. ``"smpl_neutral"``, ``"smplx_female"``, ``"mhr_lod1"``.
    """
    if lod is not None:
        return f"{name}_lod{lod}"
    if gender is not None:
        return f"{name}_{gender}"
    return name


def _checked_vertex_id(value: object, num_vertices: int, key: str) -> int:
    """Validate one vertex id, raising :class:`ValueError` keyed by ``key``."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key!r}: vertex id {value!r} is not an int")
    if not 0 <= value < num_vertices:
        raise ValueError(f"{key!r}: vertex id {value} out of range [0, {num_vertices})")
    return value


def _validate_segmentation(raw: Mapping[str, object], num_vertices: int) -> dict[str, list[int]]:
    """Validate a raw segmentation mapping into non-empty, in-range id lists."""
    parts: dict[str, list[int]] = {}
    for name, ids in raw.items():
        if not isinstance(ids, list) or not ids:
            raise ValueError(f"{name!r}: segmentation part is empty")
        parts[name] = [_checked_vertex_id(vid, num_vertices, name) for vid in ids]
    return parts


def _write_json(path: Path, payload: object) -> None:
    """Write ``payload`` as indented JSON, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def load_builtin_segmentation(filename: str, num_vertices: int) -> dict[str, list[int]]:
    """Load a shipped ``*_vert_segmentation.json`` asset by filename.

    :param filename: asset filename, e.g. ``"smpl_vert_segmentation.json"``.
    :param num_vertices: mesh vertex count used to range-check every id.
    :returns: mapping from part name to its vertex-id list.
    """
    resource = resources.files(_ASSET_PACKAGE) / filename
    raw = json.loads(resource.read_text())
    return _validate_segmentation(raw, num_vertices)


def list_saved(directory: Path) -> dict[str, Path]:
    """Return saved ``*.json`` files as a stem-keyed mapping, sorted by name."""
    if not directory.is_dir():
        return {}
    return {path.stem: path for path in sorted(directory.glob("*.json"))}


def load_segmentation(path: Path, num_vertices: int) -> dict[str, list[int]]:
    """Load and validate a saved segmentation from ``path``."""
    raw = json.loads(path.read_text())
    return _validate_segmentation(raw, num_vertices)


def save_segmentation(path: Path, parts: Mapping[str, Sequence[int]]) -> None:
    """Save a segmentation as ``{part_name: [vertex_id, ...]}`` JSON."""
    payload = {str(name): [int(vid) for vid in ids] for name, ids in parts.items()}
    _write_json(path, payload)


def load_frame_set(path: Path, num_vertices: int) -> dict[str, tuple[int, str | None]]:
    """Load a frame set, accepting the legacy and ``{vertex, joint}`` entry forms.

    :returns: mapping name -> ``(vertex_id, joint_name | None)``; the caller that
        knows the body resolves ``joint_name`` to a native joint index.
    """
    return parse_frame_set(path, num_vertices)


def save_frame_set(path: Path, frames: Mapping[str, tuple[int, str | None]]) -> None:
    """Save a frame set as ``{name: {"vertex": int, "joint": name | null}}`` JSON.

    ``frames`` maps each name to ``(vertex_id, joint_name | None)``; entries with
    no override are written with ``"joint": null``.
    """
    payload = {
        str(name): {"vertex": int(vertex), "joint": None if joint is None else str(joint)}
        for name, (vertex, joint) in frames.items()
    }
    _write_json(path, payload)


def save_pose(path: Path, q: torch.Tensor, model_key: str) -> None:
    """Save a configuration as ``{"model_key": ..., "q": [floats]}`` JSON."""
    payload = {"model_key": model_key, "q": q.detach().cpu().tolist()}
    _write_json(path, payload)


def load_pose(path: Path, model_key: str, nq: int) -> torch.Tensor:
    """Load a saved configuration, checking its key and width.

    :param path: JSON file written by :func:`save_pose`.
    :param model_key: expected mesh-topology key; a mismatch raises.
    :param nq: expected configuration width; a mismatch raises.
    :returns: the stored configuration as a ``(nq,)`` ``float32`` tensor.
    """
    raw = json.loads(path.read_text())
    saved_key = raw.get("model_key")
    if saved_key != model_key:
        raise ValueError(f"pose model_key {saved_key!r} does not match {model_key!r}")
    coordinates = raw.get("q", [])
    if len(coordinates) != nq:
        raise ValueError(f"pose has {len(coordinates)} coordinates, expected nq={nq} for {model_key!r}")
    return torch.tensor(coordinates, dtype=torch.float32)


def frame_config_snippet(frames: Mapping[str, int]) -> str:
    """Render a frame set as paste-ready ``frame_names`` / ``frame_vertex_ids`` tuples."""
    names = "".join(f'\n    "{name}",' for name in frames)
    ids = "".join(f"\n    {vid}," for vid in frames.values())
    return f"frame_names=({names}\n)\nframe_vertex_ids=({ids}\n)"
