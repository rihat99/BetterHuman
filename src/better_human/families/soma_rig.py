"""SOMA rig: skeleton fit to the shaped mesh, repose to the template bind, twist helpers, corrective features.

Mirrors NVIDIA SOMA-X v0.3.4 (``SOMALayer`` with procedural twist joints, mid LOD). Native joints are the 110 skin
joints of the template rig, ``Root`` first. The 78 public joints are BR joints (``Root`` is the universe, always at the
origin; ``Hips`` a free flyer; the rest spherical); the 32 twist helpers are fixed joints whose rotation about their x
axis is set after FK from the swing-twist of their limb segment (``joint_transforms``).

``q_neutral`` is the template T-pose. A public joint with upstream axis-angle ``a`` (relative to the T-pose) has
``q = exp(orient^T a)``, ``orient`` its T-pose world rotation, and its BR body frame is the upstream world frame. The
free flyer is world-aligned instead: ``q[:3]`` is the Hips world position, ``q[3:7] = exp(a)``, and
``joint_transforms`` turns the Hips body frame by ``orient`` into the upstream Hips frame.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, final

import torch
from better_robot import JointFreeFlyer, KinematicState, Model, ModelBuilder
from better_robot.exceptions import ArgumentError
from better_robot.io import build_model
from better_robot.lie import so3
from better_robot.model import JointFixed, JointSpherical

from better_human.core.rig import Rig, RigBake, gravity
from better_human.core.skinning import linear_blend_skinning
from better_human.io import FormatError

if TYPE_CHECKING:
    from better_human.core.spec import FamilySpec

NEWTON_SCHULZ_STEPS = 30  #: upstream ``align_vectors`` iteration count.


@final
@dataclass(frozen=True, eq=False)
class SOMARig(Rig):
    """SOMA rig tensors. ``P = 78`` public joints, ``J = 110`` native joints, ``T = 32`` twist helpers."""

    # Skeleton fit (upstream ``SkeletonTransfer``, ``rotation_method="auto"``) on the public joints.
    joint_regressor: torch.Tensor  #: ``(P, V)`` public joint positions from the shaped mesh.
    public_ids: torch.Tensor  #: ``(P,)`` native id of every public joint.
    public_parents: torch.Tensor  #: ``(P,)`` public parents; ``Root`` is its own parent.
    fit_vertices: torch.Tensor  #: ``(N,)`` vertex of every (joint, skinned vertex) pair of the inverse-LBS fit.
    fit_joints: torch.Tensor  #: ``(N,)`` public joint of every pair.
    fit_offsets: torch.Tensor  #: ``(N, 3)`` template bind vertex minus template bind joint position.
    fit_first: torch.Tensor  #: ``(P, 2)`` first two pairs of every fitted joint (virtual normal).
    fitted_ids: torch.Tensor  #: ``(F,)`` fitted joints: those with children, except ``Root``.
    child_offsets: torch.Tensor  #: ``(P, 3)`` template bind offset of every joint from its parent.
    single_child: torch.Tensor  #: ``(P,)`` bool: the joint has one child, aligned by the shortest arc.
    only_child: torch.Tensor  #: ``(P,)`` that child (``0`` elsewhere).
    multi_children: torch.Tensor  #: ``(M,)`` children of joints with several, aligned by Procrustes.
    multi_first: torch.Tensor  #: ``(P, 2)`` first two ``multi_children`` entries of every such joint.
    multi_parents: torch.Tensor  #: ``(K,)`` joints with several children.
    fitted_source: torch.Tensor  #: ``(P,)`` joint whose fitted rotation each joint takes (end joints: the parent).
    public_weights: torch.Tensor  #: ``(V, P)`` skin weights with the twist helpers folded into their parents.
    ancestors: torch.Tensor  #: ``(P, P)`` one where the column joint is the row joint or one of its ancestors.
    # Bind and BR placements of the native joints.
    parents: torch.Tensor  #: ``(J,)`` native parents; ``Root`` is its own parent.
    expansion: torch.Tensor  #: ``(J, P)`` native bind positions from public ones (helpers interpolate the segment).
    bind_rotations: torch.Tensor  #: ``(J, 3, 3)`` template bind (A-pose) world rotations.
    inverse_bind_quaternions: torch.Tensor  #: ``(J, 4)`` their inverses.
    repose_mask: torch.Tensor  #: ``(P, 3)`` zero where upstream drops a local bind offset (Hips x and z).
    placement_frames: torch.Tensor  #: ``(J, 3, 3)`` world to BR parent frame at the template bind, per bone.
    placement_quaternions: torch.Tensor  #: ``(J, 4)`` T-pose local rotations; bind local for the twist helpers.
    # Twist helpers (upstream ``aligned_x_swing_twist`` on 8 limb segments) and the Hips frame.
    helper_ids: torch.Tensor  #: ``(T,)`` native id of every twist helper.
    turned_ids: torch.Tensor  #: ``(1 + T,)`` the Hips and the twist helpers, whose BR body frames get turned.
    twist_joints: torch.Tensor  #: ``(24,)`` native ids of the segment end, start and parent joints.
    twist_offsets: torch.Tensor  #: ``(24, 4)`` ``conj(q_bind) q_align`` from BR body frames; Hips: ``orient`` times it.
    local_twist: torch.Tensor  #: ``(8, T)`` helper angles per radian of start-to-end twist.
    inherited_twist: torch.Tensor  #: ``(8, T)`` helper angles per radian of parent-to-start twist.
    # Pose conventions.
    t_pose_rotations: torch.Tensor  #: ``(P, 3, 3)`` T-pose world rotations; upstream poses are relative to them.
    corrective_frames: torch.Tensor  #: ``(P, 3, 3)`` corrective bind transposed times T-pose local rotation (Hips: I).

    def __repr__(self) -> str:
        vertices, public = self.public_weights.shape
        return (
            f"SOMARig(joints={self.parents.shape[0]}, public_joints={public}, vertices={vertices}, "
            f"dtype={self.public_weights.dtype}, device={self.public_weights.device})"
        )

    @classmethod
    def from_arrays(cls, arrays: Mapping[str, Any], *, dtype: torch.dtype, device: torch.device) -> SOMARig:
        """Rig from validated ``soma_mid.npz`` arrays (``tools/convert/soma``), derived in float64, built in ``dtype``.

        The archive's rotations are rounded to float32; they are projected onto the nearest rotations first.
        """

        def floats(name: str) -> torch.Tensor:
            return arrays[name].double()

        parents, public = arrays["joint_parents"].clamp_min(0), arrays["public_joint_ids"]
        public_parents = arrays["public_parents"].clamp_min(0).tolist()
        count = len(public)
        bind, t_pose = floats("bind_pose_world"), floats("t_pose_world")
        bind_rotations, positions = _nearest_rotations(bind[:, :3, :3]), bind[public, :3, 3]
        t_rotations = _nearest_rotations(t_pose[:, :3, :3])
        children = [[child for child in range(1, count) if public_parents[child] == joint] for joint in range(count)]

        fitted = torch.tensor([joint > 0 and bool(kids) for joint, kids in enumerate(children)])
        fit_joints, fit_vertices = ((arrays["public_skin_weights"] > 0.01) & fitted).T.nonzero().T
        multi = [child for kids in children if len(kids) > 1 for child in kids]
        fitted_source = [joint if kids else public_parents[joint] for joint, kids in enumerate(children)]
        ancestors = torch.eye(count)
        for joint in range(1, count):  # parents precede children
            ancestors[joint] += ancestors[public_parents[joint]]
        repose_mask = torch.ones(count, 3)
        repose_mask[1, 0::2] = 0.0  # upstream align_translation: the Hips keep only their height above Root

        helpers, segment, fraction = arrays["twist_joint_ids"], arrays["twist_segment"], floats("twist_fraction")
        start, end, parent, reverse = (arrays[f"segment_{role}"] for role in ("start", "end", "parent", "reverse"))
        targets = torch.cat((end, start[reverse])).tolist()
        if len(set(targets)) < len(targets):
            raise FormatError(
                "SOMA twist segments must end on distinct joints that no reverse segment starts on; "
                "convert the archive again with tools/convert/soma"
            )
        expansion = torch.zeros(len(parents), count, dtype=torch.float64)
        expansion[public, torch.arange(count)] = 1.0
        expansion[helpers, start[segment]] = 1.0 - fraction
        expansion[helpers, end[segment]] = fraction
        matrix = floats("twist_rotation_matrix")

        public_bind = bind_rotations[public]
        roles = torch.cat((end, start, parent))
        align = _segment_alignments(public_bind[start], positions[end] - positions[start])
        offsets = so3.compose(so3.inverse(so3.from_matrix(public_bind[roles])), align.repeat(3, 1))
        # The free flyer is world-aligned (identity origin): the Hips' T-pose orientation moves out of their placement
        # into their children's placements, their twist offsets and the frame ``joint_transforms`` reports.
        hips, on_hips = t_rotations[1], parents == 1
        local = t_rotations[parents].mT @ t_rotations
        local[helpers] = bind_rotations[parents][helpers].mT @ bind_rotations[helpers]
        local[1] = torch.eye(3, dtype=torch.float64)
        corrective_frames = _nearest_rotations(floats("corrective_bind")).mT @ local[public]
        local[on_hips] = hips @ local[on_hips]
        frames = bind_rotations[parents].mT
        frames[on_hips] = hips @ frames[on_hips]
        offsets[roles == 1] = so3.compose(so3.from_matrix(hips), offsets[roles == 1])
        tensors = {
            "joint_regressor": floats("joint_regressor"),
            "public_ids": public,
            "public_parents": torch.tensor(public_parents),
            "fit_vertices": fit_vertices,
            "fit_joints": fit_joints,
            "fit_offsets": floats("bind_vertices")[fit_vertices] - positions[fit_joints],
            "fit_first": _first_two(fit_joints, count),
            "fitted_ids": fitted.nonzero().squeeze(-1),
            "child_offsets": positions - positions[public_parents],
            "single_child": torch.tensor([joint > 0 and len(kids) == 1 for joint, kids in enumerate(children)]),
            "only_child": torch.tensor([kids[0] if len(kids) == 1 else 0 for kids in children]),
            "multi_children": torch.tensor(multi),
            "multi_first": _first_two(torch.tensor(public_parents)[multi], count),
            "multi_parents": torch.tensor([joint for joint, kids in enumerate(children) if len(kids) > 1]),
            "fitted_source": torch.tensor(fitted_source),
            "public_weights": floats("public_skin_weights"),
            "ancestors": ancestors,
            "parents": parents,
            "expansion": expansion,
            "bind_rotations": bind_rotations,
            "inverse_bind_quaternions": so3.from_matrix(bind_rotations.mT),
            "repose_mask": repose_mask,
            "placement_frames": frames,
            "placement_quaternions": so3.from_matrix(local),
            "helper_ids": helpers,
            "turned_ids": torch.cat((torch.tensor([1]), helpers)),
            "twist_joints": public[roles],
            "twist_offsets": offsets,
            "local_twist": matrix[:, end].T,
            "inherited_twist": (matrix[:, start] * reverse).T,
            "t_pose_rotations": t_rotations[public],
            "corrective_frames": corrective_frames,
        }
        return cls(
            **{
                name: value.to(device=device, dtype=dtype if value.is_floating_point() else value.dtype)
                for name, value in tensors.items()
            }
        )

    def build_model(self, spec: FamilySpec, *, frames: Sequence[tuple[str, str, torch.Tensor]]) -> Model:
        """BR model with the template (mean shape) placements; ``bake`` replaces them per identity."""
        if frames:
            raise ArgumentError("SOMA does not support contact frames; load it with contact_frames=None")
        placements = self._placements(self._fit_and_repose(spec.template_vertices)[0]).cpu()
        names, helpers = spec.joint_names, set(self.helper_ids.tolist())
        builder = ModelBuilder(spec.name, gravity=gravity(spec.up_axis))
        for name in names[1:]:
            builder.add_body(name)
        # world-aligned: an identity origin
        builder.add_joint(names[1], kind=JointFreeFlyer(), parent="world", child=names[1])
        for joint in range(2, len(names)):
            kind = JointFixed() if joint in helpers else JointSpherical()
            parent = names[spec.parents[joint]]
            builder.add_joint(names[joint], kind=kind, parent=parent, child=names[joint], origin=placements[joint])
        return build_model(builder.finalize(), dtype=spec.dtype, device=spec.device)

    def bake(
        self, spec: FamilySpec, model: Model, *, shaped_vertices: torch.Tensor, identity: Mapping[str, torch.Tensor]
    ) -> RigBake:
        """Fit and repose the skeleton to ``shaped_vertices (I..., V, 3)``; the reposed mesh is the skinning rest mesh."""
        positions, repose = self._fit_and_repose(shaped_vertices)
        rotations = so3.to_matrix(self.inverse_bind_quaternions).expand(*positions.shape[:-1], 3, 3)
        translations = -(self.bind_rotations.mT @ positions[..., None])
        return RigBake(
            joint_placements=self._placements(positions),
            rest_vertices=linear_blend_skinning(shaped_vertices, repose, weights=self.public_weights),
            rest_joint_translations=positions,
            inverse_bind_affines=torch.cat((rotations, translations), dim=-1),
            root_offset=positions.new_zeros((*positions.shape[:-2], 3)),
        )

    def joint_ids(self, spec: FamilySpec, model: Model) -> torch.Tensor:
        return torch.arange(spec.num_joints, device=self.parents.device)

    def joint_transforms(self, spec: FamilySpec, model: Model, *, kinematics: KinematicState) -> torch.Tensor:
        """``(..., J, 3, 4)`` upstream world frames: BR body frames, the Hips turned by their T-pose orientation.

        Each twist helper also turns about its x axis by the swing-twist of its segment; a segment twisted past 180°
        wraps to the opposite angle, as upstream.
        """
        transforms = kinematics.joint_transforms
        angle = self._twist_angles(so3.from_matrix(transforms.index_select(-3, self.twist_joints)[..., :3]))
        cos, sin = angle.cos(), angle.sin()
        one, zero = torch.ones_like(cos), torch.zeros_like(cos)
        spin = torch.stack((one, zero, zero, zero, cos, -sin, zero, sin, cos), dim=-1).unflatten(-1, (3, 3))
        turns = torch.cat((self.t_pose_rotations[1].expand(*spin.shape[:-3], 1, 3, 3), spin), dim=-3)
        turned = transforms.index_select(-3, self.turned_ids)
        turned = torch.cat((turned[..., :3] @ turns, turned[..., 3:]), dim=-1)
        return transforms.index_copy(-3, self.turned_ids, turned)

    def corrective_features(
        self, spec: FamilySpec, model: Model, *, q: torch.Tensor, identity: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        """``(..., P * 6)``: first two columns of ``corrective_bind^T R_local`` minus identity, ``Root`` included."""
        rotations = so3.to_matrix(so3.normalize(q[..., 3:].unflatten(-1, (-1, 4))))
        # Rows 0-11 of corrective_first_layer are zero: the Root and Hips features have no effect; kept as upstream.
        hips = rotations[..., :1, :, :] @ self.t_pose_rotations[1]  # the upstream Hips world rotation
        root = torch.eye(3, dtype=q.dtype, device=q.device).expand_as(hips)
        columns = self.corrective_frames @ torch.cat((root, hips, rotations[..., 1:, :, :]), dim=-3)[..., :2]
        eye = torch.eye(3, 2, dtype=q.dtype, device=q.device)
        return (columns - eye).flatten(-3)

    def _fit_and_repose(self, v_shaped: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Bind positions ``(..., J, 3)`` and public repose affines ``(..., P, 3, 4)`` of ``v_shaped (..., V, 3)``.

        Upstream ``prepare_identity``: regress and orient the public skeleton, re-pose it to the template bind
        rotations with the Hips over the origin and the lowest joint at ``y = 0``, pin ``Root`` to the origin, then
        place each twist helper on its segment. The affines skin ``v_shaped`` (``public_weights``) to the rest mesh.
        """
        parents = self.public_parents
        bind = self.bind_rotations[self.public_ids]
        positions = torch.einsum("jv,...vc->...jc", self.joint_regressor, v_shaped)
        rotations = self._fit_rotations(v_shaped, positions, bind)
        local = (rotations[..., parents, :, :].mT @ (positions - positions[..., parents, :])[..., None]).squeeze(-1)
        offsets = (bind[parents] @ (local * self.repose_mask)[..., None]).squeeze(-1)
        reposed = torch.einsum("jk,...kc->...jc", self.ancestors, offsets)
        reposed = reposed - reposed[..., 1].amin(-1)[..., None, None] * reposed.new_tensor((0.0, 1.0, 0.0))
        reposed = torch.cat((torch.zeros_like(reposed[..., :1, :]), reposed[..., 1:, :]), dim=-2)
        delta = bind @ rotations.mT
        affines = torch.cat((delta, (reposed - (delta @ positions[..., None]).squeeze(-1))[..., None]), dim=-1)
        return torch.einsum("jp,...pc->...jc", self.expansion, reposed), affines

    def _fit_rotations(self, v_shaped: torch.Tensor, positions: torch.Tensor, bind: torch.Tensor) -> torch.Tensor:
        """``(..., P, 3, 3)`` fitted bind rotations: inverse-LBS Procrustes, then child-direction alignment."""
        parents = self.public_parents
        target = v_shaped[..., self.fit_vertices, :] - positions[..., self.fit_joints, :]
        covariance = _covariance(target, self.fit_offsets, self.fit_joints, self.fit_first)
        initial = _group_rotations(covariance, self.fitted_ids)
        target = positions - positions[..., parents, :]
        source = (initial[..., parents, :, :] @ self.child_offsets[..., None]).squeeze(-1)
        single = _rodrigues(target, source).index_select(-3, self.only_child)
        members = self.multi_children
        covariance = _covariance(target[..., members, :], source[..., members, :], parents[members], self.multi_first)
        multi = _group_rotations(covariance, self.multi_parents)
        align = torch.where(self.single_child[:, None, None], single, multi)
        return (align @ initial @ bind).index_select(-3, self.fitted_source)

    def _placements(self, positions: torch.Tensor) -> torch.Tensor:
        """``(..., J, 7)`` BR placements from native bind positions ``(..., J, 3)``; BR joint ids are native ids."""
        bones = positions - positions[..., self.parents, :]
        translation = (self.placement_frames @ bones[..., None]).squeeze(-1)
        # Root is the universe and the Hips position is the free flyer's q.
        translation = torch.cat((torch.zeros_like(translation[..., :2, :]), translation[..., 2:, :]), dim=-2)
        return torch.cat((translation, self.placement_quaternions.expand(*translation.shape[:-1], 4)), dim=-1)

    def _twist_angles(self, quaternions: torch.Tensor) -> torch.Tensor:
        """``(..., T)`` helper angles from the BR world quaternions ``(..., 24, 4)`` of the ``twist_joints``."""
        virtual = so3.compose(quaternions, self.twist_offsets)
        end, start, parent = virtual.unflatten(-2, (3, -1)).unbind(-3)
        local = _twist_x(so3.compose(so3.inverse(start), end))
        inherited = _twist_x(so3.compose(so3.inverse(parent), start))
        return local @ self.local_twist + inherited @ self.inherited_twist


def _first_two(groups: torch.Tensor, count: int) -> torch.Tensor:
    """``(count, 2)`` positions in sorted ``groups (N,)`` of the first two members of every group; zeros if fewer."""
    sizes = torch.bincount(groups, minlength=count)
    starts = sizes.cumsum(0) - sizes
    return torch.where((sizes >= 2)[:, None], torch.stack((starts, starts + 1), -1), 0)


def _covariance(target: torch.Tensor, source: torch.Tensor, groups: torch.Tensor, first: torch.Tensor) -> torch.Tensor:
    """``(..., G, 3, 3)`` sums of ``target source^T`` per group plus the virtual normal of its first two members.

    ``target (..., N, 3)`` and ``source (..., N, 3)`` pair up; ``groups (N,)`` in ``[0, G)``; ``first (G, 2)``.
    """
    outer = target[..., :, None] * source[..., None, :]
    covariance = outer.new_zeros((*outer.shape[:-3], len(first), 3, 3)).index_add(-3, groups, outer)
    a0, a1, b0, b1 = (vectors[..., first[:, k], :] for vectors in (target, source) for k in (0, 1))
    normal_a, normal_b = torch.linalg.cross(a0, a1), torch.linalg.cross(b0, b1)
    length_a, length_b = normal_a.norm(dim=-1, keepdim=True), normal_b.norm(dim=-1, keepdim=True)
    normal_a = normal_a * (a0.norm(dim=-1, keepdim=True) / (length_a + 1e-8))
    normal_b = normal_b * (b0.norm(dim=-1, keepdim=True) / (length_b + 1e-8))
    valid = (length_a > 1e-9) & (length_b > 1e-9)
    return covariance + torch.where(valid[..., None], normal_a[..., :, None] * normal_b[..., None, :], 0.0)


def _group_rotations(covariance: torch.Tensor, groups: torch.Tensor) -> torch.Tensor:
    """``(..., G, 3, 3)`` ``_auto_rotation`` of ``covariance (..., G, 3, 3)`` at ``groups (K,)``, identity elsewhere.

    The other groups are empty; an SVD of their zero covariance would have NaN gradients.
    """
    eye = torch.eye(3, dtype=covariance.dtype, device=covariance.device).expand_as(covariance)
    return eye.index_copy(-3, groups, _auto_rotation(covariance.index_select(-3, groups)))


def _auto_rotation(covariance: torch.Tensor) -> torch.Tensor:
    """Rotation ``(..., 3, 3)`` best aligning a covariance, as upstream ``align_vectors(method="auto")``.

    A weak identity prior conditions rank-deficient covariances; Newton-Schulz gives the polar factor; Kabsch (SVD)
    takes over for reflections and for iterations that did not converge to a rotation.
    """
    eye = torch.eye(3, dtype=covariance.dtype, device=covariance.device)
    scale = covariance.abs().sum(-1).amax(-1).clamp_min(1e-8)
    volume = torch.linalg.det(covariance).abs() / scale**3
    covariance = covariance + (0.05 * ((1e-6 - volume) / 1e-6).clamp(0.0, 1.0) * scale)[..., None, None] * eye
    rotation = covariance / covariance.abs().sum(-1).amax(-1).clamp_min(1e-8)[..., None, None]
    for _ in range(NEWTON_SCHULZ_STEPS):
        rotation = rotation @ (3.0 * eye - rotation.mT @ rotation) * 0.5
    flip = torch.linalg.det(rotation)[..., None, None] < 0
    rotation = torch.cat((rotation[..., :2], torch.where(flip, -rotation[..., 2:], rotation[..., 2:])), dim=-1)
    error = (rotation.mT @ rotation - eye).abs().amax((-2, -1))
    valid = ((torch.linalg.det(rotation) - 1.0).abs() <= 1e-2) & (error <= 1e-2)  # False for NaN too
    kabsch = ~valid | (torch.linalg.det(covariance) < 0)
    u, _, vh = torch.linalg.svd(covariance)
    flip = torch.linalg.det(u @ vh)[..., None, None] < 0
    u = torch.cat((u[..., :2], torch.where(flip, -u[..., 2:], u[..., 2:])), -1)
    return torch.where(kabsch[..., None, None], u @ vh, rotation)


def _rodrigues(target: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    """Shortest-arc rotation ``(..., 3, 3)`` turning ``source (..., 3)`` towards ``target (..., 3)``.

    Opposite vectors (cosine below ``-1 + 1e-6``) get upstream's half turn about an axis normal to ``source``.
    """
    a = source / source.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    b = target / target.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    x, y, z = torch.linalg.cross(a, b).unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), dim=-1).unflatten(-1, (3, 3))
    cosine = (a * b).sum(-1).clamp(-1.0, 1.0)
    opposite = cosine < -1.0 + 1e-6
    eye = torch.eye(3, dtype=a.dtype, device=a.device)
    shortest = eye + skew + skew @ skew / torch.where(opposite, 1.0, 1.0 + cosine)[..., None, None]
    axis = torch.linalg.cross(a, torch.where(a[..., :1].abs() > 0.6, eye[1], eye[0]))
    axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-8)  # Root's zero source has no axis
    half_turn = 2.0 * axis[..., :, None] * axis[..., None, :] - eye
    return torch.where(opposite[..., None, None], half_turn, shortest)


def _nearest_rotations(matrices: torch.Tensor) -> torch.Tensor:
    """Nearest rotations ``(..., 3, 3)`` (polar factors) of nearly orthonormal ``matrices (..., 3, 3)``."""
    u, _, vh = torch.linalg.svd(matrices)
    return u @ vh


def _twist_x(quaternion: torch.Tensor) -> torch.Tensor:
    """Upstream ``quaternion_twist_angle_xyzw`` about x: ``4 atan2(x, w + 1)`` of the unit quaternion, ``w >= 0``."""
    quaternion = so3.normalize(quaternion)
    quaternion = torch.where(quaternion[..., 3:] < 0, -quaternion, quaternion)
    return 4.0 * torch.atan2(quaternion[..., 0], quaternion[..., 3] + 1.0)


def _segment_alignments(start_rotations: torch.Tensor, spans: torch.Tensor) -> torch.Tensor:
    """``(S, 4)`` upstream segment frames: x along the bind span, y the start joint's y made orthogonal to it."""
    spans = spans / spans.norm(dim=-1, keepdim=True)
    x = torch.where((start_rotations[..., 0] * spans).sum(-1, keepdim=True) >= 0, spans, -spans)
    z = torch.linalg.cross(x, start_rotations[..., 1])
    z = z / z.norm(dim=-1, keepdim=True)
    y = torch.linalg.cross(z, x)
    return so3.from_matrix(torch.stack((x, y / y.norm(dim=-1, keepdim=True), z), dim=-1))


__all__ = ["SOMARig"]
