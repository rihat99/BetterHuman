"""Contact frames — named mesh points baked onto the body as robot frames.

A *contact frame* is one mesh vertex turned into a BetterRobot frame that is
rigidly attached to a parent joint. Its placement is baked from the rest pose:
the origin sits on the vertex and the local z axis is the outward mesh normal
there. Posing the body then reports both a contact point and the surface
direction straight from forward kinematics, with no mesh work per frame.

The input is a mapping ``name -> (vertex_id, joint_name | None)``, as
``better_human.io.read_frame_set`` returns it from a frame-set file. A ``None``
joint means "use the vertex's dominant skinning joint".
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

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


def resolve_parents(
    entries: Mapping[str, tuple[int, str | None]],
    joint_names: Sequence[str],
    part_ids: torch.Tensor,
) -> list[int]:
    """Resolve the parent joint index of every frame entry.

    An explicit joint name wins. ``None`` falls back to the vertex's dominant
    skinning joint (``BodyStructure.part_ids``), which does not depend on shape.

    :param entries: name -> ``(vertex_id, joint_name | None)``, e.g. from
        ``better_human.io.read_frame_set``.
    :param joint_names: the body's native reduced joint names.
    :param part_ids: ``(V,)`` argmax skinning joint per vertex.
    :returns: one native reduced-joint index per entry, in entry order.
    :raises ValueError: for a vertex id outside ``[0, V)`` or an unknown joint name.
    """
    index_by_name = {name: index for index, name in enumerate(joint_names)}
    parents = []
    for name, (vertex, joint) in entries.items():
        if isinstance(vertex, bool) or not isinstance(vertex, int) or not 0 <= vertex < len(part_ids):
            raise ValueError(f"{name!r}: vertex id {vertex!r} is not an int in [0, {len(part_ids)})")
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
    "resolve_parents",
    "vertex_normals",
]
