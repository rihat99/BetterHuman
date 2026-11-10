"""``RegressedRig``, the SMPL-family rig, and ``drop_joints``, which removes terminal subtrees from an SMPL-family spec."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import NamedTuple, final

import torch
from better_robot import Model, ModelBuilder
from better_robot.exceptions import ArgumentError
from better_robot.io import build_model
from better_robot.lie import so3

from better_human.core.reduction import reduce_joint_data
from better_human.core.rig import Rig, RigBake, gravity
from better_human.core.skinning import top_k_skin
from better_human.core.spec import FamilySpec, LinearCorrective, Parts


@final
@dataclass(frozen=True)
class RegressedRig(Rig):
    """SMPL-family rig: ``J = joint_regressor @ v_shaped``, root free-flyer + one spherical joint per joint.

    Everything is root-centred: the root joint sits at the native origin and ``root_offset`` records the regressed
    root position, so a classic ``transl`` becomes ``transl + root_offset``.
    """

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
        return build_model(builder.finalize(), dtype=spec.dtype, device=spec.device)

    def bake(
        self, spec: FamilySpec, model: Model, *, shaped_vertices: torch.Tensor, identity: Mapping[str, torch.Tensor]
    ) -> RigBake:
        joints = torch.einsum("jv,...vc->...jc", self.joint_regressor, shaped_vertices)
        root_offset = joints[..., 0, :]
        rest_joints = joints - root_offset[..., None, :]
        eye = torch.eye(3, dtype=rest_joints.dtype, device=rest_joints.device).expand(*rest_joints.shape[:-1], 3, 3)
        return RigBake(
            joint_placements=self.placements_from_rest(spec, model, rest_joint_translations=rest_joints),
            rest_vertices=shaped_vertices - root_offset[..., None, :],
            rest_joint_translations=rest_joints,
            inverse_bind_affines=torch.cat((eye, -rest_joints[..., None]), dim=-1),  # identity-oriented at the joint
            root_offset=root_offset,
        )

    def joint_ids(self, spec: FamilySpec, model: Model) -> torch.Tensor:
        return torch.arange(1, spec.num_joints + 1, device=spec.device)

    def corrective_features(
        self, spec: FamilySpec, model: Model, *, q: torch.Tensor, identity: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        quaternions = q[..., 7:].unflatten(-1, (spec.num_joints - 1, 4))
        return _pose_feature_rotmat_minus_identity(quaternions)

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

    def without_joints(self, spec: FamilySpec, dropped: Sequence[int]) -> FamilySpec:
        if not isinstance(spec.corrective, LinearCorrective):
            raise ArgumentError("drop_joints needs a linear (R - I) corrective, as the SMPL family loaders build")
        data = _JointData(
            spec.joint_names,
            spec.parents,
            spec.file_joint_ids,
            self.joint_regressor,
            spec.skin_weights,
            spec.skin_indices,
            spec.corrective.posedirs,
            spec.parts,
        )
        reduced = _reduce_joints(data, dropped)
        return replace(
            spec,
            joint_names=reduced.joint_names,
            parents=reduced.parents,
            file_joint_ids=reduced.file_joint_ids,
            skin_weights=reduced.skin_weights,
            skin_indices=reduced.skin_indices,
            corrective=LinearCorrective(reduced.posedirs),
            rig=RegressedRig(reduced.joint_regressor),
            parts=reduced.parts,
        )


@final
class _JointData(NamedTuple):
    """The per-joint arrays of an SMPL-family spec that dropping joints reduces."""

    joint_names: tuple[str, ...]
    parents: tuple[int, ...]
    file_joint_ids: torch.Tensor  #: ``(J,)``
    joint_regressor: torch.Tensor  #: ``(J, V)``
    skin_weights: torch.Tensor  #: ``(V, K)``
    skin_indices: torch.Tensor  #: ``(V, K)``
    posedirs: torch.Tensor  #: ``(P, V * 3)`` of the linear corrective
    parts: Parts


def drop_joints(spec: FamilySpec, names: Sequence[str]) -> FamilySpec:
    """Spec without the terminal subtrees rooted at ``names`` (their skin weights go to the surviving ancestor).

    Each dropped joint's part merges into its surviving ancestor's: authored closed parts (SMPL) join their faces and
    lose the cap they shared, so the total volume is kept and every other part is unchanged. The merged part keeps
    the survivor's density category, so the dropped volume takes the survivor's density: the total mass is kept only
    where the merged parts share a density (a uniform density; the loaders' ``hands=False`` and ``face=False``). At
    "Dempster", dropping SMPL's left forearm (``["left_elbow", "left_wrist", "left_hand"]``) loses 0.085 kg.

    The spec's rig drops the joints (``Rig.without_joints``); of the shipped rigs only the SMPL family's
    ``RegressedRig`` can. Raises :class:`~better_robot.exceptions.ArgumentError` for a rig that cannot, for a bare
    string (pass a list) and for a repeated name, and :class:`~better_robot.exceptions.UnknownNameError` for an
    unknown name.
    """
    if isinstance(names, str):
        raise ArgumentError(f"drop_joints takes a list of joint names, got the string {names!r}; pass [{names!r}]")
    repeated = sorted(name for name, count in Counter(names).items() if count > 1)
    if repeated:
        raise ArgumentError(f"drop_joints names {repeated} more than once; name each joint once")
    reduced = spec.rig.without_joints(spec, [spec.joint_id(name) for name in names])
    if reduced is None:
        raise ArgumentError(f"drop_joints: {type(spec.rig).__name__} cannot drop joints; the SMPL family's rig can")
    return reduced


def _reduce_joints(data: _JointData, dropped: Sequence[int]) -> _JointData:
    """``data`` without the terminal subtrees rooted at the joints ``dropped``, in ``data``'s dtype.

    The SMPL loader reduces its float64 arrays before it builds the spec at the requested dtype (record 0005).
    """
    if not dropped:
        return data
    if data.parts.joints != tuple(range(len(data.joint_names))):
        raise ArgumentError("drop_joints expects one part per joint")
    num_joints, device = len(data.joint_names), data.skin_weights.device
    num_vertices = data.skin_weights.shape[0]
    dense = data.skin_weights.new_zeros((num_vertices, num_joints)).scatter_add_(
        1, data.skin_indices, data.skin_weights
    )
    posedirs = data.posedirs.T.reshape(num_vertices, 3, -1)
    joint_names, parents, regressor, dense, posedirs, survivors = reduce_joint_data(
        data.joint_names, data.parents, data.joint_regressor, dense, posedirs, dropped
    )
    weights, indices = top_k_skin(dense)
    survivor_tensor = torch.tensor(survivors, dtype=torch.long, device=device)
    old_to_new = {old: new for new, old in enumerate(survivors)}
    ancestor = torch.arange(num_joints, device=device)
    for joint in range(num_joints):  # every dropped joint maps to its nearest surviving ancestor
        while int(ancestor[joint]) not in old_to_new:
            ancestor[joint] = data.parents[int(ancestor[joint])]
    part_ids = torch.tensor([old_to_new[int(ancestor[j])] for j in range(num_joints)], device=device)
    part_joints = tuple(old_to_new[j] for j in data.parts.joints if j in old_to_new)
    categories = data.parts.density_categories
    closed_faces = None
    if data.parts.closed_faces is not None:
        closed_faces = {
            new: _merged_closed_faces(data, [j for j in range(num_joints) if int(ancestor[j]) == old], old_to_new)
            for new, old in enumerate(survivors)
        }
    return _JointData(
        joint_names,
        parents,
        data.file_joint_ids[survivor_tensor],  # an earlier drop already renumbered joints: keep file numbering
        regressor,
        weights,
        indices,
        posedirs.reshape(num_vertices * 3, -1).T,
        Parts(
            part_ids[data.parts.ids],
            part_joints,
            tuple(
                category for joint, category in zip(data.parts.joints, categories, strict=True) if joint in old_to_new
            ),
            data.parts.density_tables,
            closed_faces,
        ),
    )


def _merged_closed_faces(data: _JointData, members: Sequence[int], old_to_new: Mapping[int, int]) -> torch.Tensor:
    """The authored closed faces of the parts ``members`` merged into one part, its cap joints renumbered.

    A cap two merged parts share appears once in each (as the same three indices) and closes neither part any more,
    so both copies go; the rest is the union's closed surface. Raises
    :class:`~better_robot.exceptions.ArgumentError` for a cap left at a dropped joint.
    """
    num_vertices = data.skin_weights.shape[0]
    faces = torch.cat([data.parts.closed_faces[member] for member in members])
    _, inverse, counts = torch.unique(faces.sort(dim=1).values, dim=0, return_inverse=True, return_counts=True)
    faces = faces[counts[inverse] == 1]
    renumbered = torch.full((len(data.joint_names),), -1, dtype=faces.dtype, device=faces.device)
    for old, new in old_to_new.items():
        renumbered[old] = new
    caps = faces >= num_vertices
    joints = renumbered[(faces - num_vertices).clamp(min=0)]
    if bool((caps & (joints < 0)).any()):
        raise ArgumentError(
            f"drop_joints: the authored part of {data.joint_names[members[0]]} keeps a dropped joint cap"
        )
    return torch.where(caps, num_vertices + joints, faces)


def _pose_feature_rotmat_minus_identity(joint_quaternions: torch.Tensor) -> torch.Tensor:
    """Return flattened non-root ``(R - I)`` pose features.

    ``joint_quaternions`` contains only non-root joints and has shape
    ``(..., J - 1, 4)`` in scalar-last order. The result has shape
    ``(..., (J - 1) * 9)``.
    """
    rotation_matrices = so3.to_matrix(so3.normalize(joint_quaternions))
    identity = torch.eye(3, dtype=joint_quaternions.dtype, device=joint_quaternions.device)
    return (rotation_matrices - identity).flatten(start_dim=-3)


__all__ = ["RegressedRig", "drop_joints"]
