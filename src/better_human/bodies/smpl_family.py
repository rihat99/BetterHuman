"""Shared SMPL-family data binding without concrete-model policy."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from numbers import Real

import torch
from better_robot import Data, Model
from better_robot.io.builders.kinematic_tree import build_kinematic_tree_model

from better_human.core import BodyStructure, BodyValues, ParametricBody
from better_human.core.mass import (
    PartMeshTopology,
    body_inertias_from_shape,
    build_authored_topology,
    build_part_topology,
    segment_vertices,
)
from better_human.core.pipeline import (
    blend_shapes,
    linear_corrective,
    pose_feature_rotmat_minus_identity,
    regress_joints,
)


def bake_regressed_pelvis_shape(
    body: ParametricBody,
    expanded_parameters: Mapping[str, torch.Tensor],
) -> tuple[BodyValues, Model]:
    """Bake SMPL-family shape, regressed joints, placements, and inertias."""
    coefficients = torch.cat(tuple(expanded_parameters.values()), dim=-1)
    batch_shape = coefficients.shape[:-1]
    world_vertices = body.values.v_template + blend_shapes(coefficients, body.values.shape_basis)
    world_joints = regress_joints(body.values.joint_regressor, world_vertices)
    pelvis_offset = world_joints[..., 0, :]
    centered_vertices = world_vertices - pelvis_offset[..., None, :]
    centered_joints = world_joints - pelvis_offset[..., None, :]

    parents = torch.tensor(body.structure.parents[1:], dtype=torch.long, device=centered_joints.device)
    child_offsets = centered_joints[..., 1:, :] - centered_joints[..., parents, :]
    root_offset = torch.zeros_like(centered_joints[..., :1, :])
    native_offsets = torch.cat((root_offset, child_offsets), dim=-2)
    native_quaternions = native_offsets.new_zeros((*native_offsets.shape[:-1], 4))
    native_quaternions[..., 3] = 1.0
    native_placements = torch.cat((native_offsets, native_quaternions), dim=-1)
    universe_placement = native_placements.new_zeros((*batch_shape, 1, 7))
    universe_placement[..., 6] = 1.0
    joint_placements = torch.cat((universe_placement, native_placements), dim=-2)

    body_inertias = None
    if body.structure.part_topology is not None:
        body_inertias = body_inertias_from_shape(
            centered_vertices,
            centered_joints,
            body.structure.part_topology,
            body.values.density,
        )

    values = replace(
        body.values,
        shape_parameters=expanded_parameters,
        v_shaped=centered_vertices,
        rest_joints=centered_joints,
        pelvis_offset=pelvis_offset,
        body_inertias=body_inertias,
    )
    robot = body.robot.with_values(
        joint_placements=joint_placements,
        body_inertias=body_inertias,
    )
    return values, robot


def smpl_linear_pose_corrective(body: ParametricBody, data: Data) -> torch.Tensor:
    """Evaluate the SMPL-family quaternion feature and linear corrective."""
    root_width = body.robot.nqs[1]
    joint_quaternions = data.q[..., root_width:].reshape(
        *data.q.shape[:-1],
        len(body.structure.joint_names) - 1,
        4,
    )
    feature = pose_feature_rotmat_minus_identity(joint_quaternions)
    return linear_corrective(feature, body.values.posedirs)


def bind_smpl_body(
    body: ParametricBody,
    structure: BodyStructure,
    values: BodyValues,
    robot: Model,
) -> None:
    """Bind shared SMPL-family state and policy functions to a frozen body."""
    object.__setattr__(body, "structure", structure)
    object.__setattr__(body, "values", values)
    object.__setattr__(body, "robot", robot)
    object.__setattr__(body, "_shape_bake_policy", bake_regressed_pelvis_shape)
    object.__setattr__(body, "_pose_corrective_policy", smpl_linear_pose_corrective)


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
    closed_faces: Mapping[int, torch.Tensor] | None = None,
) -> tuple[BodyValues, torch.Tensor, PartMeshTopology | None]:
    """Bind static segmentation and optional neutral shape inertias.

    An authored ``closed_faces`` asset drives the volume-preserving joint-cap
    topology; otherwise argmax segmentation with centroid caps is used. The
    argmax ``part_ids`` are always returned for downstream part coloring.
    """
    if not isinstance(compute_mass, bool):
        raise TypeError("compute_mass must be bool")
    part_ids = segment_vertices(values.skinning_weights, skinning_indices)
    if not compute_mass:
        return values, part_ids, None
    if closed_faces is not None:
        topology = build_authored_topology(
            closed_faces,
            values.v_shaped.shape[-2],
            values.v_shaped,
            values.rest_joints,
        )
    else:
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


__all__ = [
    "bake_regressed_pelvis_shape",
    "bind_smpl_body",
    "build_family_robot",
    "bind_mass_properties",
    "density_tensor",
    "make_body_values",
    "padded_skinning",
    "parents_from_kintree",
    "smpl_linear_pose_corrective",
]
