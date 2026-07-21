"""Functionally immutable parametric-body composition."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

import better_robot as br
import torch
from better_robot import Data, Model

from .pipeline import (
    blend_shapes,
    lbs,
    linear_corrective,
    pose_feature_rotmat_minus_identity,
    regress_joints,
    skinning_transforms,
)
from .mass import PartMeshTopology, body_inertias_from_shape


@dataclass(frozen=True)
class BodyStructure:
    """Static topology and convention metadata for a parametric body."""

    #: Number of vertices in the fixed native topology.
    num_vertices: int
    #: Number of triangular faces in the fixed native topology.
    num_faces: int
    #: Triangle indices with shape ``(F, 3)``.
    faces: torch.Tensor
    #: Native LBS joint names in weight-file order.
    joint_names: tuple[str, ...]
    #: Native parent indices, with ``-1`` for the root.
    parents: tuple[int, ...]
    #: Original native joint indices retained by this spawn.
    native_joint_indices: tuple[int, ...]
    #: Padded native joint indices with shape ``(V, K)``.
    skinning_indices: torch.Tensor
    #: Argmax-skinning body-part assignment with shape ``(V,)``.
    part_ids: torch.Tensor
    #: Closed per-part mesh topology, or ``None`` when mass is disabled.
    part_topology: PartMeshTopology | None
    #: Ordered mapping from shape-group name to coefficient width.
    parameter_groups: Mapping[str, int]
    #: Optional topology-specific landmark index tables.
    landmarks: Mapping[str, torch.Tensor]
    #: Native length unit after loader normalization.
    unit: str
    #: Native world up axis.
    up_axis: str
    #: Native world forward axis.
    forward_axis: str
    #: Pose representation accepted by the classic converter.
    native_pose_format: str


@dataclass(frozen=True)
class BodyValues:
    """Loaded weights and differentiable shape-baked products."""

    #: Unshaped native template with shape ``(V, 3)``.
    v_template: torch.Tensor
    #: Concatenated named-group basis with shape ``(V, 3, S)``.
    shape_basis: torch.Tensor
    #: Linear pose-corrective matrix with shape ``(P, V * 3)``.
    posedirs: torch.Tensor
    #: Dense or sparse joint regressor with shape ``(J, V)``.
    joint_regressor: torch.Tensor
    #: Padded nonzero LBS weights with shape ``(V, K)``.
    skinning_weights: torch.Tensor
    #: Current named parameter tensors, including leading batch axes.
    shape_parameters: Mapping[str, torch.Tensor]
    #: Pelvis-centered shaped vertices with shape ``(..., V, 3)``.
    v_shaped: torch.Tensor
    #: Pelvis-centered rest joints with shape ``(..., J, 3)``.
    rest_joints: torch.Tensor
    #: Native regressed pelvis ``J_0`` with shape ``(..., 3)``.
    pelvis_offset: torch.Tensor
    #: Scalar or per-part density in kg/m^3.
    density: torch.Tensor
    #: BetterRobot packed inertias ``(..., nbodies, 10)``, if enabled.
    body_inertias: torch.Tensor | None


@dataclass(frozen=True)
class ParametricBody:
    """Compose BetterHuman deformation data with a BetterRobot model."""

    #: Static BetterHuman topology and convention metadata.
    structure: BodyStructure
    #: Loaded and shape-baked BetterHuman tensors.
    values: BodyValues
    #: BetterRobot topology paired with shape-dependent values.
    robot: Model

    def with_shape(self, **named_groups: torch.Tensor) -> ParametricBody:
        """Return a body with named shape groups baked into model values.

        Each group has shape ``(..., width)``. Omitted groups retain their
        current value, and all leading dimensions broadcast together.
        """
        unknown = set(named_groups) - set(self.structure.parameter_groups)
        if unknown:
            expected = ", ".join(self.structure.parameter_groups)
            raise ValueError(f"Unknown shape groups {sorted(unknown)}; expected: {expected}")

        parameters = dict(self.values.shape_parameters)
        parameters.update(named_groups)
        for name, width in self.structure.parameter_groups.items():
            value = parameters[name]
            if not isinstance(value, torch.Tensor) or not value.is_floating_point():
                raise TypeError(f"{name} must be a floating torch.Tensor")
            if value.shape[-1:] != (width,):
                raise ValueError(f"{name} must have shape (..., {width}); got {tuple(value.shape)}")
            if value.dtype != self.values.v_template.dtype or value.device != self.values.v_template.device:
                raise ValueError(
                    f"{name} must use dtype={self.values.v_template.dtype} and "
                    f"device={self.values.v_template.device}; got dtype={value.dtype}, device={value.device}"
                )

        batch_shape = torch.broadcast_shapes(*(value.shape[:-1] for value in parameters.values()))
        expanded_parameters = {
            name: parameters[name].expand(*batch_shape, width)
            for name, width in self.structure.parameter_groups.items()
        }
        coefficients = torch.cat(tuple(expanded_parameters.values()), dim=-1)
        world_vertices = self.values.v_template + blend_shapes(coefficients, self.values.shape_basis)
        world_joints = regress_joints(self.values.joint_regressor, world_vertices)
        pelvis_offset = world_joints[..., 0, :]
        centered_vertices = world_vertices - pelvis_offset[..., None, :]
        centered_joints = world_joints - pelvis_offset[..., None, :]

        parents = torch.tensor(self.structure.parents[1:], dtype=torch.long, device=centered_joints.device)
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
        if self.structure.part_topology is not None:
            body_inertias = body_inertias_from_shape(
                centered_vertices,
                centered_joints,
                self.structure.part_topology,
                self.values.density,
            )

        values = replace(
            self.values,
            shape_parameters=expanded_parameters,
            v_shaped=centered_vertices,
            rest_joints=centered_joints,
            pelvis_offset=pelvis_offset,
            body_inertias=body_inertias,
        )
        robot = self.robot.with_values(
            joint_placements=joint_placements,
            body_inertias=body_inertias,
        )
        return self._replace(values=values, robot=robot)

    def fk(self, q: torch.Tensor) -> Data:
        """Run BetterRobot forward kinematics for ``q (..., nq)``."""
        return br.forward_kinematics(self.robot, q)

    def vertices(self, q: torch.Tensor) -> torch.Tensor:
        """Run one FK traversal, pose correctives, and sparse LBS."""
        return self.vertices_from_data(self.fk(q))

    def vertices_from_data(self, data: Data) -> torch.Tensor:
        """Compute vertices from an already-populated BetterRobot data object."""
        if data.joint_pose_world is None:
            raise ValueError("data has no world joint poses; call body.fk(q) first")
        root_width = self.robot.nqs[1]
        joint_quaternions = data.q[..., root_width:].reshape(
            *data.q.shape[:-1],
            len(self.structure.joint_names) - 1,
            4,
        )
        feature = pose_feature_rotmat_minus_identity(joint_quaternions)
        rest_vertices = self.values.v_shaped + linear_corrective(feature, self.values.posedirs)
        transforms = skinning_transforms(data.joint_pose_world[..., 1:, :], self.values.rest_joints)
        return lbs(
            rest_vertices,
            transforms,
            self.values.skinning_weights,
            self.structure.skinning_indices,
        )

    def _replace(self, *, values: BodyValues, robot: Model) -> ParametricBody:
        """Copy the concrete body while replacing its differentiable values."""
        rebound = object.__new__(type(self))
        for name, value in vars(self).items():
            object.__setattr__(rebound, name, value)
        object.__setattr__(rebound, "values", values)
        object.__setattr__(rebound, "robot", robot)
        return rebound


__all__ = ["BodyStructure", "BodyValues", "ParametricBody"]
