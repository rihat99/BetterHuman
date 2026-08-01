"""Functionally immutable parametric-body composition."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import better_robot as br
import torch
from better_robot import Data, Model

from .mass import PartMeshTopology
from .pipeline import lbs, skinning_transforms, skinning_transforms_from_inverse


@dataclass(frozen=True)
class BodyStructure:
    """Static topology and convention metadata for a parametric body."""

    #: Number of vertices in the fixed native topology.
    num_vertices: int
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
    #: Optional native-joint to BetterRobot-joint pose gather.
    native_pose_joint_indices: torch.Tensor | None = None


@dataclass(frozen=True)
class BodyValues:
    """Loaded weights and differentiable shape-baked products."""

    #: Unshaped native template with shape ``(V, 3)``.
    v_template: torch.Tensor
    #: Concatenated named-group basis with shape ``(V, 3, S)``.
    shape_basis: torch.Tensor | None
    #: Linear pose-corrective matrix with shape ``(P, V * 3)``.
    posedirs: torch.Tensor | None
    #: Dense joint regressor with shape ``(J, V)``.
    joint_regressor: torch.Tensor | None
    #: Padded nonzero LBS weights with shape ``(V, K)``.
    skinning_weights: torch.Tensor
    #: Dense matrix-blend LBS weights with shape ``(V, J)``.
    skinning_weight_matrix: torch.Tensor
    #: Current named parameter tensors, including leading batch axes.
    shape_parameters: Mapping[str, torch.Tensor]
    #: Model-canonical shaped vertices with shape ``(..., V, 3)``.
    v_shaped: torch.Tensor
    #: Model-canonical rest joints with shape ``(..., J, 3)``.
    rest_joints: torch.Tensor
    #: Native regressed pelvis ``J_0`` with shape ``(..., 3)``.
    pelvis_offset: torch.Tensor | None
    #: Scalar or per-part density in kg/m^3.
    density: torch.Tensor
    #: BetterRobot packed inertias ``(..., nbodies, 10)``, if enabled.
    body_inertias: torch.Tensor | None
    #: Authoritative inverse bind poses ``(J, 7)``, when supplied by a model.
    inverse_bind_poses: torch.Tensor | None = None
    #: Authored bind-pose pivots ``(J, 3)`` used by scaled skinning.
    bind_joint_positions: torch.Tensor | None = None
    #: Shape-baked accumulated uniform scales ``(..., J)``.
    joint_scales: torch.Tensor | None = None


ShapeBakePolicy = Callable[
    ["ParametricBody", Mapping[str, torch.Tensor]],
    tuple[BodyValues, Model],
]
PoseCorrectivePolicy = Callable[["ParametricBody", Data], torch.Tensor]


@dataclass(frozen=True)
class ParametricBody:
    """Compose BetterHuman deformation data with a BetterRobot model."""

    #: Static BetterHuman topology and convention metadata.
    structure: BodyStructure
    #: Loaded and shape-baked BetterHuman tensors.
    values: BodyValues
    #: BetterRobot topology paired with shape-dependent values.
    robot: Model
    #: Model-specific shape bake selected by the concrete loader.
    _shape_bake_policy: ShapeBakePolicy
    #: Model-specific pose-feature and corrective function.
    _pose_corrective_policy: PoseCorrectivePolicy

    def __call__(self, q: torch.Tensor, **shape_groups: torch.Tensor) -> torch.Tensor:
        """Vertices from pose and optional shape in one call."""
        body = self.with_shape(**shape_groups) if shape_groups else self
        return body.vertices(q)

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
        values, robot = self._shape_bake_policy(self, expanded_parameters)
        return self._replace(values=values, robot=robot)

    def fk(self, q: torch.Tensor, *, compute_frames: bool = False) -> Data:
        """Run BetterRobot forward kinematics for ``q (..., nq)``.

        Set ``compute_frames`` to also fill ``Data.frame_pose_world``, e.g. for
        contact frames.
        """
        return br.forward_kinematics(self.robot, q, compute_frames=compute_frames)

    def vertices(self, q: torch.Tensor) -> torch.Tensor:
        """Run one FK traversal, pose correctives, and matrix-blend LBS."""
        return self.vertices_from_data(self.fk(q))

    def vertices_from_data(self, data: Data) -> torch.Tensor:
        """Compute vertices from an already-populated BetterRobot data object."""
        if data.joint_pose_world is None:
            raise ValueError("data has no world joint poses; call body.fk(q) first")
        rest_vertices = self.values.v_shaped + self._pose_corrective_policy(self, data)
        native_joint_poses = self.native_joint_poses(data)
        if self.values.inverse_bind_poses is None:
            transforms = skinning_transforms(native_joint_poses, self.values.rest_joints)
        else:
            transforms = skinning_transforms_from_inverse(
                native_joint_poses,
                self.values.inverse_bind_poses,
            )
        return lbs(
            rest_vertices,
            transforms,
            self.values.skinning_weight_matrix,
            joint_scales=self.values.joint_scales,
            bind_joint_positions=self.values.bind_joint_positions,
        )

    def native_joint_poses(self, data: Data) -> torch.Tensor:
        """Return world poses in the body's native skin-joint order."""
        if data.joint_pose_world is None:
            raise ValueError("data has no world joint poses; call body.fk(q) first")
        indices = self.structure.native_pose_joint_indices
        if indices is None:
            return data.joint_pose_world[..., 1:, :]
        return data.joint_pose_world.index_select(-2, indices)

    def landmark_positions(self, vertices: torch.Tensor) -> torch.Tensor:
        """Gather landmark keypoint positions from posed or shaped vertices.

        :param vertices: ``(..., V, 3)`` vertices, e.g. from :meth:`vertices`.
        :returns: ``(..., L, 3)`` landmark keypoints in native frame order.
        """
        frame_vertex_ids = self.structure.landmarks["frame_vertex_ids"]
        return vertices[..., frame_vertex_ids, :]

    def _replace(self, *, values: BodyValues, robot: Model) -> ParametricBody:
        """Copy the concrete body while replacing its differentiable values."""
        rebound = object.__new__(type(self))
        for name, value in vars(self).items():
            object.__setattr__(rebound, name, value)
        object.__setattr__(rebound, "values", values)
        object.__setattr__(rebound, "robot", robot)
        return rebound


__all__ = ["BodyStructure", "BodyValues", "ParametricBody"]
