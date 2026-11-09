"""Part inertias: each closed part mesh integrated at its density, in its part frame."""

from __future__ import annotations

import torch
from better_robot.exceptions import ShapeError
from better_robot.lie import se3
from better_robot.spatial import inertia_of_mesh

from .topology import PartTopology


def part_inertias(
    vertices: torch.Tensor,
    part_poses: torch.Tensor,
    *,
    topology: PartTopology,
    density: torch.Tensor,
    check_mesh: bool = False,
) -> torch.Tensor:
    """``(..., P, 10)`` BetterRobot inertias of the closed parts, each in its part frame.

    :param vertices: ``(..., V, 3)`` mesh the parts are cut from.
    :param part_poses: ``(..., P, 7)`` part frames (or ``(..., P, 3)`` positions, frames aligned with the world);
        their positions are also the authored caps, so an authored topology needs one part per cap joint.
    :param topology: the closed parts.
    :param density: ``()`` or ``(P,)`` kg/m³, on the dtype and device of ``vertices``.
    :param check_mesh: also read the values, one host sync per part, and raise
        :class:`~better_robot.exceptions.ArgumentError` for a face id outside the mesh, a density that is not finite and
        positive, or a degenerate part. ``HumanModel.from_spec`` checks once; bakes integrate unchecked.
    Raises :class:`~better_robot.exceptions.ShapeError` for ``part_poses`` that are neither positions nor poses.
    Each part is integrated about its own vertices, which is exact for the open seam parts too.
    """
    if part_poses.shape[-1] not in (3, 7):
        raise ShapeError(f"part_poses must have shape (..., P, 3) or (..., P, 7); got {tuple(part_poses.shape)}")
    per_part = density.broadcast_to((topology.num_parts,))
    positions = part_poses[..., :3]
    inertias: list[torch.Tensor] = []
    for part, (part_vertices, faces) in enumerate(topology.meshes(vertices, positions)):
        pose = part_poses[..., part, :]
        if pose.shape[-1] == 3:
            local = part_vertices - pose[..., None, :]
        else:
            local = se3.act(se3.inverse(pose)[..., None, :], part_vertices)
        inertias.append(inertia_of_mesh(local, faces, density=per_part[part], check_mesh=check_mesh))
    return torch.stack(inertias, dim=-2)
