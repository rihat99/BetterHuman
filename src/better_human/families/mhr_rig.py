"""Authored MHR rig: archive joint tree, proportion-driven placements and Euler-feature correctives."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from better_robot import Model
from better_robot.kinematics import forward_kinematics_state
from better_robot.lie import se3

from better_human.core.pipeline import lbs_affines, pose_feature_6d_from_euler, skin_affines
from better_human.core.rig import FrameSpec, RigBake

from .mhr_tree import _TreeBinding, accumulated_scales, build_mhr_tree

if TYPE_CHECKING:
    from better_human.core.spec import FamilySpec


@dataclass(frozen=True)
class AuthoredRig:
    """MHR rig built from the archive's joint tree; ``tree`` holds the BR model and parameter bindings."""

    arrays: Mapping[str, Any]  #: validated archive arrays the tree is rebuilt from on ``to``.
    tree: _TreeBinding

    def build_model(
        self, spec: FamilySpec, *, frames: Sequence[FrameSpec], dtype: torch.dtype, device: torch.device
    ) -> Model:
        if frames:
            raise ValueError("MHR does not support contact frames")
        robot = self.tree.robot
        if robot.joint_placements.dtype != dtype or robot.joint_placements.device != torch.device(device):
            robot = build_mhr_tree(self.arrays, dtype, torch.device(device)).robot
        return robot

    def bake(
        self, spec: FamilySpec, model: Model, v_shaped: torch.Tensor, identity: dict[str, torch.Tensor]
    ) -> RigBake:
        meta = spec.meta
        proportions = identity["proportions"]
        translation_delta = torch.einsum("...p,jcp->...jc", proportions, meta["translation_basis"])
        joint_scales = accumulated_scales(torch.einsum("...p,jp->...j", proportions, meta["scale_basis"]), spec.parents)
        parent_ids = torch.tensor(spec.parents[1:], dtype=torch.long, device=proportions.device)
        translations = (meta["joint_offsets"][1:] + translation_delta[..., 1:, :]) * joint_scales.index_select(
            -1, parent_ids
        )[..., None]
        quaternions = meta["joint_pre_rotations"][1:].expand(*translations.shape[:-1], 4)
        native = torch.cat((translations, quaternions), dim=-1)
        base = model.joint_placements
        placements = base.expand(*native.shape[:-2], *base.shape).index_copy(
            -2, self.tree.native_origin_joint_indices, native
        )
        baked = model.with_values(joint_placements=placements)
        state = forward_kinematics_state(baked, baked.q_neutral)
        transforms = state.joint_transform.index_select(-3, self.tree.native_pose_joint_indices)
        bottom = transforms.new_tensor([0.0, 0.0, 0.0, 1.0]).expand(*transforms.shape[:-2], 1, 4)
        rest_poses = se3.from_matrix(torch.cat((transforms, bottom), dim=-2))
        affines = skin_affines(
            transforms,
            inverse_bind_poses=meta["inverse_bind_poses"],
            joint_scales=joint_scales,
            bind_joint_positions=meta["bind_joint_positions"],
        )
        dense = v_shaped.new_zeros(spec.vertices.shape[0], spec.num_joints).scatter_add_(
            1, spec.skin_indices, spec.skin_weights
        )
        return RigBake(
            joint_placements=placements,
            rest_joints=rest_poses[..., :3],
            rest_poses=rest_poses,
            pelvis=rest_poses.new_zeros(rest_poses.shape[:-2] + (3,)),
            joint_scales=joint_scales,
            neutral_vertices=lbs_affines(v_shaped, affines, dense),
        )

    def placements_from_rest(self, spec: FamilySpec, model: Model, rest_joints: torch.Tensor) -> None:
        return None

    def stream_joint_basis(self, spec: FamilySpec, name: str) -> None:
        return None

    def native_joint_ids(self, spec: FamilySpec, model: Model) -> torch.Tensor:
        return self.tree.native_pose_joint_indices

    def corrective_features(self, spec: FamilySpec, model: Model, q: torch.Tensor) -> torch.Tensor:
        entries = model.expand_configuration(q).index_select(-1, self.tree.corrective_full_q_indices)
        euler = entries @ self.tree.corrective_euler_transform.mT
        return pose_feature_6d_from_euler(euler.unflatten(-1, (125, 3)))

    def to(self, dtype: torch.dtype, device: torch.device) -> AuthoredRig:
        return AuthoredRig(self.arrays, build_mhr_tree(self.arrays, dtype, torch.device(device)))


__all__ = ["AuthoredRig"]
