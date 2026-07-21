"""Shared SMPL-family data binding without concrete-model policy."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from numbers import Real

import torch
from better_robot import Model
from better_robot.io.builders.kinematic_tree import build_kinematic_tree_model

from better_human.core import BodyValues
from better_human.core.mass import (
    PartMeshTopology,
    body_inertias_from_shape,
    build_part_topology,
    segment_vertices,
)
from better_human.core.pipeline import blend_shapes, regress_joints


def parents_from_kintree(kintree_table: torch.Tensor) -> tuple[int, ...]:
    """Resolve a two-row native-id kintree table to column parent indices."""
    joint_ids = [int(value) for value in kintree_table[1].tolist()]
    columns_by_id = {joint_id: column for column, joint_id in enumerate(joint_ids)}
    parents = [-1]
    for column in range(1, len(joint_ids)):
        parent_id = int(kintree_table[0, column])
        try:
            parents.append(columns_by_id[parent_id])
        except KeyError as error:
            raise ValueError(f"kintree parent id {parent_id} for joint column {column} does not exist") from error
    return tuple(parents)


def padded_skinning(weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compress dense ``(V, J)`` weights to padded nonzero ``(V, K)`` data."""
    nonzero = weights != 0
    max_influences = int(nonzero.sum(dim=-1).max().item())
    ordered_indices = torch.argsort(nonzero.to(torch.int8), dim=-1, descending=True, stable=True)
    indices = ordered_indices[:, :max_influences]
    selected_weights = torch.gather(weights, 1, indices)
    indices = torch.where(selected_weights != 0, indices, torch.zeros_like(indices))
    return selected_weights, indices


def make_body_values(
    v_template: torch.Tensor,
    shape_basis: torch.Tensor,
    posedirs: torch.Tensor,
    joint_regressor: torch.Tensor,
    dense_weights: torch.Tensor,
    parameter_groups: Mapping[str, int],
    density: torch.Tensor,
) -> tuple[BodyValues, torch.Tensor]:
    """Build neutral pelvis-centered values and the skinning index pattern."""
    parameters = {
        name: torch.zeros(width, dtype=v_template.dtype, device=v_template.device)
        for name, width in parameter_groups.items()
    }
    coefficients = torch.cat(tuple(parameters.values()), dim=-1)
    world_vertices = v_template + blend_shapes(coefficients, shape_basis)
    world_joints = regress_joints(joint_regressor, world_vertices)
    pelvis_offset = world_joints[0]
    skinning_weights, skinning_indices = padded_skinning(dense_weights)
    values = BodyValues(
        v_template=v_template,
        shape_basis=shape_basis,
        posedirs=posedirs.reshape(-1, posedirs.shape[-1]).T,
        joint_regressor=joint_regressor,
        skinning_weights=skinning_weights,
        shape_parameters=parameters,
        v_shaped=world_vertices - pelvis_offset,
        rest_joints=world_joints - pelvis_offset,
        pelvis_offset=pelvis_offset,
        density=density,
        body_inertias=None,
    )
    return values, skinning_indices


def density_tensor(
    density: float | torch.Tensor,
    num_parts: int,
    exemplar: torch.Tensor,
) -> torch.Tensor:
    """Validate scalar or per-part density without silent tensor conversion."""
    if isinstance(density, Real) and not isinstance(density, bool):
        value = exemplar.new_tensor(float(density))
    elif isinstance(density, torch.Tensor) and density.is_floating_point():
        if density.shape not in ((), (num_parts,)):
            raise ValueError(f"density must be scalar or have shape ({num_parts},), got {tuple(density.shape)}")
        if density.dtype != exemplar.dtype or density.device != exemplar.device:
            raise ValueError(
                f"density must use dtype={exemplar.dtype} and device={exemplar.device}; "
                f"got dtype={density.dtype}, device={density.device}"
            )
        value = density
    else:
        raise TypeError("density must be a positive real number or floating torch.Tensor")
    if not bool(torch.isfinite(value).all()) or bool((value <= 0).any()):
        raise ValueError("density must contain only finite positive values")
    return value


def bind_mass_properties(
    values: BodyValues,
    faces: torch.Tensor,
    skinning_indices: torch.Tensor,
    num_parts: int,
    compute_mass: bool,
) -> tuple[BodyValues, torch.Tensor, PartMeshTopology | None]:
    """Bind static segmentation and optional neutral shape inertias."""
    if not isinstance(compute_mass, bool):
        raise TypeError("compute_mass must be bool")
    part_ids = segment_vertices(values.skinning_weights, skinning_indices)
    if not compute_mass:
        return values, part_ids, None
    topology = build_part_topology(faces, part_ids, num_parts)
    inertias = body_inertias_from_shape(
        values.v_shaped,
        values.rest_joints,
        topology,
        values.density,
    )
    return replace(values, body_inertias=inertias), part_ids, topology


def build_family_robot(
    name: str,
    joint_names: Sequence[str],
    parents: Sequence[int],
    rest_joints: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
) -> Model:
    """Build the preserved-order free-flyer/spherical BetterRobot tree."""
    parent_tensor = torch.tensor(parents[1:], dtype=torch.long, device=rest_joints.device)
    child_offsets = rest_joints[1:] - rest_joints[parent_tensor]
    translations = torch.cat((torch.zeros_like(rest_joints[:1]), child_offsets), dim=0)
    return build_kinematic_tree_model(
        name=name,
        joint_names=joint_names,
        parents=parents,
        translations=translations.cpu(),
        root_kind="free_flyer",
        child_kind="spherical",
        preserve_joint_order=True,
        dtype=dtype,
    ).to(device=device)


def classic_tensor(
    name: str,
    value: torch.Tensor | None,
    width: int,
    exemplar: torch.Tensor,
) -> torch.Tensor | None:
    """Validate one optional classic tensor at a converter boundary."""
    if value is None:
        return None
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise TypeError(f"{name} must be a floating torch.Tensor or None")
    if value.shape[-1:] != (width,):
        raise ValueError(f"{name} must have shape (..., {width}); got {tuple(value.shape)}")
    if value.dtype != exemplar.dtype or value.device != exemplar.device:
        raise ValueError(
            f"{name} must use dtype={exemplar.dtype} and device={exemplar.device}; "
            f"got dtype={value.dtype}, device={value.device}"
        )
    return value


def broadcast_configuration(
    q: torch.Tensor,
    robot: Model,
    exemplar: torch.Tensor,
) -> tuple[torch.Tensor, tuple[int, ...]]:
    """Validate ``q`` and broadcast it against shape-baked value axes."""
    if not isinstance(q, torch.Tensor) or not q.is_floating_point():
        raise TypeError("q must be a floating torch.Tensor")
    if q.shape[-1:] != (robot.nq,):
        raise ValueError(f"q must have shape (..., {robot.nq}); got {tuple(q.shape)}")
    if q.dtype != exemplar.dtype or q.device != exemplar.device:
        raise ValueError(
            f"q must use dtype={exemplar.dtype} and device={exemplar.device}; "
            f"got dtype={q.dtype}, device={q.device}"
        )
    batch_shape = torch.broadcast_shapes(q.shape[:-1], exemplar.shape[:-1])
    return q.expand(*batch_shape, robot.nq), batch_shape


__all__ = [
    "build_family_robot",
    "broadcast_configuration",
    "bind_mass_properties",
    "classic_tensor",
    "density_tensor",
    "make_body_values",
    "padded_skinning",
    "parents_from_kintree",
]
