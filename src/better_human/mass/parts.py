"""Per-part quantities of a segmented mesh: each vertex's part, each part's density, each part's inertia."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from better_robot.exceptions import ArgumentError, ShapeError, UnknownNameError
from better_robot.lie import se3
from better_robot.spatial import inertia_of_mesh

from .topology import PartTopology


def segment_vertices(weights: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """``(V,)`` long: the joint of each vertex's strongest influence.

    :param weights: ``(V, K)`` nonnegative top-K skin weights.
    :param indices: ``(V, K)`` joint of each weight.
    """
    strongest = weights.argmax(dim=-1, keepdim=True)
    return indices.gather(-1, strongest).squeeze(-1).to(dtype=torch.long)


def part_densities(
    tables: Mapping[str, Mapping[str, float]],
    name: str,
    *,
    categories: Sequence[str | None],
    part_names: Sequence[str],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """``(P,)`` kg/m³: part ``p`` takes table ``name``'s density of its category ``categories[p]``.

    :param tables: table name -> density category -> kg/m³ (``Parts.density_tables``).
    :param name: the table, e.g. ``"Dempster"``.
    :param categories: the category of each part, ``None`` for a part no category covers (``Parts.density_categories``).
    :param part_names: the name of each part, for the messages.
    Raises :class:`~better_robot.exceptions.UnknownNameError` for a table not in ``tables`` or a category the table
    lacks, and :class:`~better_robot.exceptions.ArgumentError` naming the parts without a category.
    """
    if name not in tables:
        raise UnknownNameError("density table", name, valid=sorted(tables))
    uncovered = [part for part, category in zip(part_names, categories, strict=True) if category is None]
    if uncovered:
        raise ArgumentError(
            f"density={name!r}: {len(uncovered)} of {len(part_names)} parts have no density category: {uncovered}. "
            "Pass density=None (no inertias), a uniform float in kg/m^3 or a (P,) tensor, until the family declares "
            "a category for every part."
        )
    table = tables[name]
    for category in categories:
        if category not in table:
            raise UnknownNameError(f"{name} category name", category, valid=list(table))
    return torch.tensor([table[category] for category in categories], dtype=dtype, device=device)


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
    Each part is integrated about its own vertices.
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
