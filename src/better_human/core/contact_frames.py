"""Contact frames — named mesh points baked onto the body as robot frames.

A *contact frame* is one mesh vertex turned into a BetterRobot frame that is
rigidly attached to a parent joint. Its placement is baked from the rest pose:
the origin sits on the vertex and the local z axis is the outward mesh normal
there. Posing the body then reports both a contact point and the surface
direction straight from forward kinematics, with no mesh work per frame.

The authored input is a frame-set JSON file, ``{name: {"vertex": int,
"joint": name | null}}``. A ``null`` joint means "use the vertex's dominant
skinning joint".
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from better_robot.lie import so3


@dataclass(frozen=True)
class ContactFrames:
    """Contact-frame spec resolved against one body."""

    #: Frame names, in the order they were authored.
    names: tuple[str, ...]
    #: Mesh vertex carrying each frame.
    vertex_ids: tuple[int, ...]
    #: Parent joint of each frame, as a native reduced-joint index.
    parent_joints: tuple[int, ...]
    #: BetterRobot frame id of each frame.
    frame_ids: tuple[int, ...]


def _parse_frame_entry(value: object, num_vertices: int, name: str) -> tuple[int, str | None]:
    """Parse one entry — a bare vertex id, or a ``{vertex, joint}`` dict."""
    joint = None
    if isinstance(value, dict):
        joint = value.get("joint")
        if joint is not None and not isinstance(joint, str):
            raise ValueError(f"{name!r}: frame joint {joint!r} is not a joint name or null")
        value = value.get("vertex")
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name!r}: vertex id {value!r} is not an int")
    if not 0 <= value < num_vertices:
        raise ValueError(f"{name!r}: vertex id {value} out of range [0, {num_vertices})")
    return value, joint


def parse_frame_set(path: str | Path, num_vertices: int) -> dict[str, tuple[int, str | None]]:
    """Load a frame-set JSON file, keeping the authored order.

    :param path: JSON file mapping each frame name to a vertex id or a
        ``{"vertex": int, "joint": name | null}`` object.
    :param num_vertices: mesh vertex count, used to range-check every id.
    :returns: mapping name -> ``(vertex_id, joint_name | None)``. The parent
        joint stays a portable name here; :func:`resolve_parents` turns it into
        an index once the body's joint names are known.
    """
    raw = json.loads(Path(path).read_text())
    return {name: _parse_frame_entry(value, num_vertices, name) for name, value in raw.items()}


def resolve_parents(
    entries: Mapping[str, tuple[int, str | None]],
    joint_names: Sequence[str],
    part_ids: torch.Tensor,
) -> list[int]:
    """Resolve the parent joint index of every frame entry.

    An explicit joint name wins. ``None`` falls back to the vertex's dominant
    skinning joint (``BodyStructure.part_ids``), which does not depend on shape.

    :param entries: parsed frame set, as returned by :func:`parse_frame_set`.
    :param joint_names: the body's native reduced joint names.
    :param part_ids: ``(V,)`` argmax skinning joint per vertex.
    :returns: one native reduced-joint index per entry, in entry order.
    """
    index_by_name = {name: index for index, name in enumerate(joint_names)}
    parents = []
    for name, (vertex, joint) in entries.items():
        if joint is None:
            parents.append(int(part_ids[vertex]))
            continue
        if joint not in index_by_name:
            raise ValueError(f"{name!r}: unknown parent joint {joint!r}")
        parents.append(index_by_name[joint])
    return parents


def vertex_normals(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Outward per-vertex normals ``(..., V, 3)``.

    Each face normal has a magnitude proportional to the triangle area, so
    accumulating them onto the face's three vertices and normalizing gives the
    usual area-weighted vertex normal.

    :param vertices: ``(..., V, 3)`` mesh vertices.
    :param faces: ``(F, 3)`` triangle indices, counter-clockwise wound.
    """
    corner_a = vertices[..., faces[:, 0], :]
    corner_b = vertices[..., faces[:, 1], :]
    corner_c = vertices[..., faces[:, 2], :]
    face_normals = torch.linalg.cross(corner_b - corner_a, corner_c - corner_a, dim=-1)
    normals = torch.zeros_like(vertices)
    normals.index_add_(-2, faces.reshape(-1), face_normals.repeat_interleave(3, dim=-2))
    return normals / normals.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def _basis_from_z(z: torch.Tensor) -> torch.Tensor:
    """Deterministic orthonormal basis ``(..., 3, 3)`` whose third column is ``z``.

    Gram-Schmidt is seeded with the world axis least aligned with ``z``, so the
    seed is never parallel to ``z`` and the basis is a pure function of ``z``.
    """
    axes = torch.eye(3, dtype=z.dtype, device=z.device)
    seed = axes[z.abs().argmin(dim=-1)]
    x = seed - (seed * z).sum(dim=-1, keepdim=True) * z
    x = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    y = torch.linalg.cross(z, x, dim=-1)
    return torch.stack((x, y, z), dim=-1)


def frame_local_placements(
    v_shaped: torch.Tensor,
    rest_joints: torch.Tensor,
    faces: torch.Tensor,
    vertex_ids: Sequence[int],
    parent_joints: Sequence[int],
) -> torch.Tensor:
    """Bake each contact frame's placement in its parent joint's coordinates.

    Rest-pose joint rotations are identity, so a parent-local direction is just
    the world direction and the translation is a plain difference.

    :param v_shaped: ``(..., V, 3)`` rest-pose vertices.
    :param rest_joints: ``(..., J, 3)`` rest-pose joints, in the same frame.
    :param faces: ``(F, 3)`` triangle indices.
    :param vertex_ids: length-``N`` mesh vertex of each frame.
    :param parent_joints: length-``N`` parent joint of each frame.
    :returns: ``(..., N, 7)`` SE3 placements, ``[translation, quaternion xyzw]``.
    """
    vertices = torch.as_tensor(vertex_ids, dtype=torch.long, device=v_shaped.device)
    parents = torch.as_tensor(parent_joints, dtype=torch.long, device=v_shaped.device)
    translation = v_shaped[..., vertices, :] - rest_joints[..., parents, :]
    normals = vertex_normals(v_shaped, faces)[..., vertices, :]
    return torch.cat((translation, so3.from_matrix(_basis_from_z(normals))), dim=-1)


__all__ = [
    "ContactFrames",
    "frame_local_placements",
    "parse_frame_set",
    "resolve_parents",
    "vertex_normals",
]
