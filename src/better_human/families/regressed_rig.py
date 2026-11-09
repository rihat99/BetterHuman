"""``RegressedRig``, the SMPL-family rig, and ``drop_joints``, which removes terminal subtrees from an SMPL-family spec."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import ClassVar, final

import torch
from better_robot import Model, ModelBuilder
from better_robot.io import build_model

from better_human.core.pipeline import pose_feature_rotmat_minus_identity, regress_joints
from better_human.core.reduction import reduce_joint_data
from better_human.core.rig import Rig, RigBake, gravity
from better_human.core.spec import Corrective, FamilySpec, Parts, top_k_skin


@final
@dataclass(frozen=True)
class RegressedRig(Rig):
    """SMPL-family rig: ``J = joint_regressor @ v_shaped``, root free-flyer + one spherical joint per joint.

    Everything is root-centred: the root joint sits at the native origin and ``root_offset`` records the regressed
    root position, so a classic ``transl`` becomes ``transl + root_offset``.
    """

    moves_joints_with_streams: ClassVar[bool] = True

    joint_regressor: torch.Tensor  #: ``(J, V)``

    def build_model(self, spec: FamilySpec, *, frames: Sequence[tuple[str, str, torch.Tensor]]) -> Model:
        builder = ModelBuilder(spec.name, gravity=gravity(spec.up_axis))
        for name in spec.joint_names:
            builder.add_body(name)
        builder.add_free_flyer_root(spec.joint_names[0], child=spec.joint_names[0])
        for joint, parent in enumerate(spec.parents[1:], start=1):
            builder.add_spherical(
                spec.joint_names[joint], parent=spec.joint_names[parent], child=spec.joint_names[joint]
            )
        for name, parent, placement in frames:
            builder.add_frame(name, parent=parent, placement=placement.detach().cpu())
        return build_model(builder.finalize(), dtype=spec.dtype).to(device=spec.device)

    def bake(
        self, spec: FamilySpec, model: Model, *, shaped_vertices: torch.Tensor, identity: Mapping[str, torch.Tensor]
    ) -> RigBake:
        joints = regress_joints(self.joint_regressor, shaped_vertices)
        pelvis = joints[..., 0, :]
        rest_joints = joints - pelvis[..., None, :]
        placements = self.placements_from_rest(spec, model, rest_joint_translations=rest_joints)
        quaternions = rest_joints.new_zeros((*rest_joints.shape[:-1], 4))
        quaternions[..., 3] = 1.0
        return RigBake(placements, rest_joints, torch.cat((rest_joints, quaternions), dim=-1), pelvis)

    def joint_ids(self, spec: FamilySpec, model: Model) -> torch.Tensor:
        return torch.arange(1, spec.num_joints + 1, device=spec.device)

    def corrective_features(
        self, spec: FamilySpec, model: Model, *, q: torch.Tensor, identity: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        quaternions = q[..., 7:].unflatten(-1, (spec.num_joints - 1, 4))
        return pose_feature_rotmat_minus_identity(quaternions)

    def stream_joint_basis(self, spec: FamilySpec, name: str) -> torch.Tensor:
        return torch.einsum("jv,vcn->jcn", self.joint_regressor, spec.stream_bases[name])

    def placements_from_rest(
        self, spec: FamilySpec, model: Model, *, rest_joint_translations: torch.Tensor
    ) -> torch.Tensor:
        rest = rest_joint_translations
        parents = torch.tensor(spec.parents[1:], dtype=torch.long, device=rest.device)
        offsets = torch.cat((torch.zeros_like(rest[..., :1, :]), rest[..., 1:, :] - rest[..., parents, :]), dim=-2)
        placements = torch.cat((torch.zeros_like(offsets[..., :1, :]), offsets), dim=-2)
        quaternions = placements.new_zeros((*placements.shape[:-1], 4))
        quaternions[..., 3] = 1.0
        return torch.cat((placements, quaternions), dim=-1)

    def to(self, dtype: torch.dtype, device: torch.device) -> RegressedRig:
        return replace(self, joint_regressor=self.joint_regressor.to(dtype=dtype, device=device))


def drop_joints(spec: FamilySpec, names: Sequence[str]) -> FamilySpec:
    """Spec without the terminal subtrees rooted at ``names`` (their skin weights go to the surviving ancestor).

    Only families with a regressed rig and a linear ``(R - I)`` corrective support this (SMPL family).
    """
    if not names:
        return spec
    if not isinstance(spec.rig, RegressedRig) or spec.corrective.kind != "linear":
        raise ValueError("drop_joints needs a regressed rig with a linear corrective")
    if spec.parts.joints != tuple(range(spec.num_joints)):
        raise ValueError("drop_joints expects one part per joint")
    dropped = [spec.joint_id(name) for name in names]
    posedirs = spec.corrective.posedirs.T.reshape(spec.num_vertices, 3, -1)
    joint_names, parents, regressor, dense, posedirs, survivors = reduce_joint_data(
        spec.joint_names, spec.parents, spec.rig.joint_regressor, spec.skin_matrix(), posedirs, dropped
    )
    weights, indices = top_k_skin(dense)
    survivor_tensor = torch.tensor(survivors, dtype=torch.long, device=spec.device)
    if "native_joint_ids" in spec.meta:  # an earlier drop already renumbered joints; keep file numbering
        survivor_tensor = spec.meta["native_joint_ids"].to(spec.device)[survivor_tensor]
    old_to_new = {old: new for new, old in enumerate(survivors)}
    part_joints = tuple(old_to_new[j] for j in spec.parts.joints if j in old_to_new)
    ancestor = torch.arange(spec.num_joints, device=spec.device)
    for joint in range(spec.num_joints):  # every dropped joint maps to its nearest surviving ancestor
        while int(ancestor[joint]) not in old_to_new:
            ancestor[joint] = spec.parents[int(ancestor[joint])]
    part_ids = torch.tensor([old_to_new[int(ancestor[j])] for j in range(spec.num_joints)], device=spec.device)
    return replace(
        spec,
        joint_names=joint_names,
        parents=parents,
        skin_weights=weights,
        skin_indices=indices,
        corrective=Corrective("linear", posedirs=posedirs.reshape(spec.num_vertices * 3, -1).T),
        rig=replace(spec.rig, joint_regressor=regressor),
        parts=Parts(part_ids[spec.parts.ids], part_joints),
        meta={**spec.meta, "native_joint_ids": survivor_tensor},
    )


__all__ = ["RegressedRig", "drop_joints"]
