"""Rigs: how identity turns into joint placements and how a configuration feeds the corrective.

A rig is the one behavioural part of a :class:`~better_human.core.spec.FamilySpec`. Two exist:
:class:`RegressedRig` (SMPL family: joints regressed from the shaped mesh, free-flyer root plus
spherical joints) and the authored MHR rig in ``better_human.families.mhr``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Protocol

import torch
from better_robot import Model, ModelBuilder
from better_robot.io import build_model

from .pipeline import pose_feature_rotmat_minus_identity, regress_joints

if TYPE_CHECKING:
    from .body import Tensors
    from .spec import FamilySpec

#: An operational frame to attach at build time: ``(name, parent joint name, placement (7,))``.
FrameSpec = tuple[str, str, torch.Tensor]


@dataclass(frozen=True)
class RigBake:
    """Identity-dependent kinematic values. Leading dims are the identity batch."""

    joint_placements: torch.Tensor  #: ``(..., nj, 7)`` for the BR model (joint 0 is the universe).
    rest_joints: torch.Tensor  #: ``(..., J, 3)`` native-order joint positions in the skinning rest frame.
    rest_poses: torch.Tensor  #: ``(..., J, 7)`` native-order world poses at the neutral configuration.
    pelvis: torch.Tensor  #: ``(..., 3)`` offset of the root joint from the native (uncentred) origin.
    joint_scales: torch.Tensor | None = None  #: ``(..., J)`` accumulated uniform scales (authored rigs).
    neutral_vertices: torch.Tensor | None = (
        None  #: ``(..., V, 3)`` mesh at the neutral pose when it differs from ``v_shaped``.
    )


class Rig(Protocol):
    def build_model(
        self, spec: FamilySpec, *, frames: Sequence[FrameSpec], dtype: torch.dtype, device: torch.device
    ) -> Model:
        """Unbaked BR model whose bodies are named after ``spec.joint_names``."""

    def bake(
        self, spec: FamilySpec, model: Model, v_shaped: torch.Tensor, identity: dict[str, torch.Tensor]
    ) -> RigBake:
        """Joint placements and rest joints for shaped vertices ``v_shaped (..., V, 3)`` and identity values."""

    def placements_from_rest(self, spec: FamilySpec, model: Model, rest_joints: torch.Tensor) -> torch.Tensor | None:
        """``(..., nj, 7)`` placements for shifted rest joints, or ``None`` if streams cannot move joints."""

    def stream_joint_basis(self, spec: FamilySpec, name: str) -> torch.Tensor | None:
        """``(J, 3, n)`` joint motion of stream group ``name``; ``None`` when it moves no joint."""

    def native_joint_ids(self, spec: FamilySpec, model: Model) -> torch.Tensor:
        """``(J,)`` BR joint ids in native skin-joint order."""

    def corrective_features(
        self, spec: FamilySpec, model: Model, q: torch.Tensor, *, identity: Tensors | None = None
    ) -> torch.Tensor:
        """``(..., P)`` features the corrective reads, from ``q (..., nq)``.

        ``identity`` holds the identity group values ``(I..., n)`` :meth:`Body.bake` used (``None`` when the
        caller has none), so a rig whose corrective depends on shape (STAR, SUPR) can read them.
        """

    def to(self, dtype: torch.dtype, device: torch.device) -> Rig: ...


def gravity(up_axis: str, magnitude: float = 9.81) -> tuple[float, float, float]:
    """World gravity ``(3,)`` pointing against ``up_axis`` (``"+y"``, ``"-z"``, ...)."""
    sign, axis = up_axis[0], "xyz".index(up_axis[1])
    value = -magnitude if sign == "+" else magnitude
    return tuple(value if i == axis else 0.0 for i in range(3))


@dataclass(frozen=True)
class RegressedRig:
    """SMPL-family rig: ``J = joint_regressor @ v_shaped``, root free-flyer + one spherical joint per joint.

    Everything is pelvis-centred: the root joint sits at the native origin and ``pelvis`` records
    the regressed root position, so a classic ``transl`` becomes ``transl + pelvis``.
    """

    joint_regressor: torch.Tensor  #: ``(J, V)``

    def build_model(
        self, spec: FamilySpec, *, frames: Sequence[FrameSpec], dtype: torch.dtype, device: torch.device
    ) -> Model:
        builder = ModelBuilder(spec.name, gravity=gravity(spec.up_axis))
        for name in spec.joint_names:
            builder.add_body(name)
        builder.add_free_flyer_root(spec.joint_names[0], child=spec.joint_names[0])
        for joint, parent in enumerate(spec.parents[1:], start=1):
            builder.add_spherical(
                spec.joint_names[joint], parent=spec.joint_names[parent], child=spec.joint_names[joint]
            )
        for name, parent, placement in frames:
            builder.add_frame(name, parent_body=parent, placement=placement.detach().cpu())
        return build_model(builder.finalize(), preserve_joint_order=True, dtype=dtype).to(device=device)

    def bake(
        self, spec: FamilySpec, model: Model, v_shaped: torch.Tensor, identity: dict[str, torch.Tensor]
    ) -> RigBake:
        joints = regress_joints(self.joint_regressor, v_shaped)
        pelvis = joints[..., 0, :]
        rest_joints = joints - pelvis[..., None, :]
        rest_poses = torch.cat((rest_joints, _identity_quaternions(rest_joints)), dim=-1)
        return RigBake(self.placements_from_rest(spec, model, rest_joints), rest_joints, rest_poses, pelvis)

    def placements_from_rest(self, spec: FamilySpec, model: Model, rest_joints: torch.Tensor) -> torch.Tensor:
        parents = torch.tensor(spec.parents[1:], dtype=torch.long, device=rest_joints.device)
        offsets = torch.cat(
            (torch.zeros_like(rest_joints[..., :1, :]), rest_joints[..., 1:, :] - rest_joints[..., parents, :]), dim=-2
        )
        universe = torch.zeros_like(offsets[..., :1, :])
        placements = torch.cat((universe, offsets), dim=-2)
        return torch.cat((placements, _identity_quaternions(placements)), dim=-1)

    def stream_joint_basis(self, spec: FamilySpec, name: str) -> torch.Tensor | None:
        return torch.einsum("jv,vcn->jcn", self.joint_regressor, spec.streams[name])

    def native_joint_ids(self, spec: FamilySpec, model: Model) -> torch.Tensor:
        return torch.arange(1, spec.num_joints + 1, device=model.joint_placements.device)

    def corrective_features(
        self, spec: FamilySpec, model: Model, q: torch.Tensor, *, identity: Tensors | None = None
    ) -> torch.Tensor:
        quaternions = q[..., 7:].unflatten(-1, (spec.num_joints - 1, 4))
        return pose_feature_rotmat_minus_identity(quaternions)

    def to(self, dtype: torch.dtype, device: torch.device) -> RegressedRig:
        return replace(self, joint_regressor=self.joint_regressor.to(dtype=dtype, device=device))


def _identity_quaternions(positions: torch.Tensor) -> torch.Tensor:
    quaternions = positions.new_zeros((*positions.shape[:-1], 4))
    quaternions[..., 3] = 1.0
    return quaternions


__all__ = ["FrameSpec", "RegressedRig", "Rig", "RigBake", "gravity"]
