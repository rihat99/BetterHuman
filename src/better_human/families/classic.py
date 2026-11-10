"""Every family's classic parameters (``*Classic``) and their formats (decision record 0004).

``SMPLFamilyFormat`` is one data-driven format for SMPL, SMPL-H, SMPL-X, MANO and FLAME; ``MHRFormat`` and
``SOMAFormat`` map the MHR and SOMA reference inputs. Convert with ``better_human.from_classic(human, classic)`` and
``better_human.to_classic(identity, q, streams=...)``. Every field is keyword-only and optional: an omitted field is
zero. The expression of SMPL-X, FLAME and MHR is a stream group, not shape state.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, ClassVar, final

import torch
from better_robot.exceptions import ArgumentError
from better_robot.lie import so3

from better_human.core.classic import ClassicFormat

from .mhr_rig import ROOT_COUNT

if TYPE_CHECKING:
    from better_human.core.human_model import HumanModel, Identity
    from better_human.core.spec import FamilySpec

_MHR_PARAMETER_COUNT = 204
_SOMA_POSE_JOINT_COUNT = 77  # public joints after the virtual Root, Hips first


@final
@dataclass(frozen=True, kw_only=True)
class SMPLClassic:
    """Classic SMPL parameters (the smplx package's ``SMPL`` inputs) with arbitrary leading batch dimensions."""

    betas: torch.Tensor | None = None  #: ``(..., num_betas)``
    global_orient: torch.Tensor | None = None  #: ``(..., 3)``
    body_pose: torch.Tensor | None = None  #: ``(..., 69)``
    transl: torch.Tensor | None = None  #: ``(..., 3)``


@final
@dataclass(frozen=True, kw_only=True)
class SMPLHClassic:
    """Classic SMPL+H parameters; hand poses are full axis-angle blocks (the file ships no hand PCA)."""

    betas: torch.Tensor | None = None  #: ``(..., num_betas)``
    global_orient: torch.Tensor | None = None  #: ``(..., 3)``
    body_pose: torch.Tensor | None = None  #: ``(..., 63)``
    left_hand_pose: torch.Tensor | None = None  #: ``(..., 45)``
    right_hand_pose: torch.Tensor | None = None  #: ``(..., 45)``
    transl: torch.Tensor | None = None  #: ``(..., 3)``


@final
@dataclass(frozen=True, kw_only=True)
class SMPLXClassic:
    """Classic SMPL-X parameters; the hand PCA width is the model's (``families.smplx(..., num_pca_comps=)``)."""

    betas: torch.Tensor | None = None  #: ``(..., num_betas)``
    global_orient: torch.Tensor | None = None  #: ``(..., 3)``
    body_pose: torch.Tensor | None = None  #: ``(..., 63)``
    jaw_pose: torch.Tensor | None = None  #: ``(..., 3)``
    leye_pose: torch.Tensor | None = None  #: ``(..., 3)``
    reye_pose: torch.Tensor | None = None  #: ``(..., 3)``
    left_hand_pose: torch.Tensor | None = None  #: ``(..., 45)``, or ``(..., num_pca_comps)``
    right_hand_pose: torch.Tensor | None = None  #: ``(..., 45)``, or ``(..., num_pca_comps)``
    expression: torch.Tensor | None = None  #: ``(..., num_expression)``
    transl: torch.Tensor | None = None  #: ``(..., 3)``


@final
@dataclass(frozen=True, kw_only=True)
class MANOClassic:
    """Classic MANO parameters; the hand PCA width is the model's (``families.mano(..., num_pca_comps=)``)."""

    betas: torch.Tensor | None = None  #: ``(..., num_betas)``
    global_orient: torch.Tensor | None = None  #: ``(..., 3)``
    hand_pose: torch.Tensor | None = None  #: ``(..., 45)``, or ``(..., num_pca_comps)``
    transl: torch.Tensor | None = None  #: ``(..., 3)``


@final
@dataclass(frozen=True, kw_only=True)
class FLAMEClassic:
    """Classic FLAME parameters (neck, jaw and eye axis-angles, the expression stream)."""

    betas: torch.Tensor | None = None  #: ``(..., num_betas)``
    global_orient: torch.Tensor | None = None  #: ``(..., 3)``
    neck_pose: torch.Tensor | None = None  #: ``(..., 3)``
    jaw_pose: torch.Tensor | None = None  #: ``(..., 3)``
    leye_pose: torch.Tensor | None = None  #: ``(..., 3)``
    reye_pose: torch.Tensor | None = None  #: ``(..., 3)``
    expression: torch.Tensor | None = None  #: ``(..., num_expression)``
    transl: torch.Tensor | None = None  #: ``(..., 3)``


@final
@dataclass(frozen=True, kw_only=True)
class MHRClassic:
    """The MHR reference inputs."""

    identity_coeffs: torch.Tensor | None = None  #: ``(..., 45)``
    model_parameters: torch.Tensor | None = None  #: ``(..., 204)``: root, pose and proportion parameters
    expression: torch.Tensor | None = None  #: ``(..., 72)``


@final
@dataclass(frozen=True, kw_only=True)
class SOMAClassic:
    """The upstream ``SOMALayer`` inputs."""

    identity_coeffs: torch.Tensor | None = None  #: ``(..., 128)``
    #: ``(..., 77, 3)`` axis-angle per public joint relative to its T-pose orientation, ``Hips`` first.
    poses: torch.Tensor | None = None
    transl: torch.Tensor | None = None  #: ``(..., 3)`` world position of the ``Hips`` joint in metres


@final
@dataclass(frozen=True, eq=False)
class SMPLFamilyFormat(ClassicFormat):
    """The SMPL family's classic parameters as data (SMPL, SMPL-H, SMPL-X, MANO, FLAME).

    Identity and stream fields are named like the spec's groups (``betas``, ``expression``); ``global_orient`` and
    ``transl`` place the root, which pivots about the pelvis (``transl + root_offset``). ``pose_fields`` maps every
    axis-angle field ``(..., 3 n)`` to its ``n`` native joints, fields in native joint order after the root. A hand
    field in ``hand_components`` holds ``(..., C)`` coefficients of its ``(C, 3 n)`` PCA basis instead, and
    ``hand_means[field] (3 n,)`` is added to its axis-angles; both are fixed at load (``num_pca_comps``,
    ``flat_hand_mean``). A joint the model was loaded without reads zero in its field, before the hand mean; a PCA
    field cannot lose some of its joints (``field_shapes`` raises :class:`~better_robot.exceptions.ArgumentError`, and
    so does ``families.drop_joints``). Axis-angles keep their angle in ``[0, 2 pi]``: a rotation by more than ``pi``
    comes back as given, not as its principal equivalent.
    """

    classic_type: type
    pose_fields: Mapping[str, tuple[str, ...]]
    hand_means: Mapping[str, torch.Tensor] = field(default_factory=dict)
    hand_components: Mapping[str, torch.Tensor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("pose_fields", "hand_means", "hand_components"):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))

    def field_shapes(self, spec: FamilySpec) -> dict[str, tuple[int, ...]]:
        shapes = {"global_orient": (3,), "transl": (3,)}
        shapes.update({name: (width,) for name, width in (*spec.identity_groups.items(), *spec.stream_groups.items())})
        for name, joints in self.pose_fields.items():
            dropped = [joint for joint in joints if joint not in spec.joint_names]
            if len(dropped) == len(joints):
                continue
            components = self.hand_components.get(name)
            if components is None:
                shapes[name] = (3 * len(joints),)
                continue
            if dropped:  # a PCA coefficient moves every joint of the hand, so none can be left out
                raise ArgumentError(
                    f"{name} holds {components.shape[0]} hand PCA coefficients, which move every joint of the hand, so "
                    f"{dropped} cannot be left out; load the model with num_pca_comps=None (full axis-angle hands) "
                    "to drop single hand joints"
                )
            shapes[name] = (components.shape[0],)
        return shapes

    def from_fields(
        self, human: HumanModel, fields: Mapping[str, torch.Tensor]
    ) -> tuple[Identity, torch.Tensor, dict[str, torch.Tensor]]:
        spec = human.spec
        identity = human.bake(**{name: fields[name] for name in spec.identity_bases})
        batch = torch.broadcast_shapes(*(value.shape[:-1] for value in fields.values()))
        streams = {name: fields[name].expand(*batch, width) for name, width in spec.stream_groups.items()}
        axes = []
        for name, joints in self.pose_fields.items():
            if name not in fields:  # every joint of the field was dropped: no pose, no hand mean
                axes.append(identity.root_offset.new_zeros((*batch, len(joints), 3)))
                continue
            value = fields[name].expand(*batch, fields[name].shape[-1])
            if name in self.hand_components:
                value = value @ self.hand_components[name]
            dropped = [index for index, joint in enumerate(joints) if joint not in spec.joint_names]
            # checked before the hand mean: a dropped joint reads zero in the classic, like to_classic writes it
            if dropped and bool(value.unflatten(-1, (len(joints), 3))[..., dropped, :].any()):
                raise ArgumentError(
                    f"from_classic: {name} sets {[joints[index] for index in dropped]}, which this model was loaded "
                    "without (drop_joints); leave them zero"
                )
            if name in self.hand_means:
                value = value + self.hand_means[name]
            axes.append(value.unflatten(-1, (len(joints), 3)))
        reduced = torch.cat(axes, dim=-2).index_select(-2, spec.file_joint_ids[1:] - 1)
        pelvis = _pelvis(identity, streams).expand(*batch, 3)
        root = torch.cat(
            (fields["transl"].expand(*batch, 3) + pelvis, so3.exp(fields["global_orient"].expand(*batch, 3))), dim=-1
        )
        return identity, torch.cat((root, so3.exp(reduced).flatten(-2)), dim=-1), streams

    def to_fields(
        self, identity: Identity, q: torch.Tensor, *, streams: Mapping[str, torch.Tensor]
    ) -> dict[str, torch.Tensor | None]:
        spec = identity.human.spec
        batch = q.shape[:-1]
        fields: dict[str, torch.Tensor | None] = {
            "global_orient": _axis_angle(q[..., 3:7]),
            "transl": q[..., :3] - _pelvis(identity, streams).expand(*batch, 3),
            **{name: identity.values[name].expand(*batch, width) for name, width in spec.identity_groups.items()},
            **streams,
        }
        num_axes = sum(len(joints) for joints in self.pose_fields.values())
        reduced = _axis_angle(q[..., 7:].unflatten(-1, (-1, 4)))
        axes = q.new_zeros((*batch, num_axes, 3)).index_copy(-2, spec.file_joint_ids[1:] - 1, reduced)
        start = 0
        for name, joints in self.pose_fields.items():
            value = axes[..., start : start + len(joints), :].flatten(-2)
            start += len(joints)
            kept = torch.tensor([joint in spec.joint_names for joint in joints], device=q.device)
            if not bool(kept.any()):
                continue
            if name in self.hand_means:  # a dropped joint keeps the mean, so it reads zero, as from_classic requires
                value = value - self.hand_means[name] * kept.repeat_interleave(3)
            components = self.hand_components.get(name)
            if components is not None:
                delta = value
                value = delta @ torch.linalg.pinv(components)
                if not torch.allclose(value @ components, delta, rtol=2e-5, atol=2e-5):
                    raise ArgumentError(
                        f"to_classic: {name} is outside the model's first {components.shape[0]} hand PCA components; "
                        "load the model with num_pca_comps=None for full axis-angle hands"
                    )
            fields[name] = value
        return fields


@final
@dataclass(frozen=True)
class MHRFormat(ClassicFormat):
    """MHR's reference inputs: ``identity_coeffs``, ``model_parameters`` (root, pose and proportion parameters) and
    ``expression``. The root's Euler angles come back on BetterRobot's principal branch."""

    classic_type: ClassVar[type] = MHRClassic

    def field_shapes(self, spec: FamilySpec) -> dict[str, tuple[int, ...]]:
        return {
            "identity_coeffs": (spec.identity_groups["identity"],),
            "model_parameters": (_MHR_PARAMETER_COUNT,),
            **{name: (width,) for name, width in spec.stream_groups.items()},
        }

    def from_fields(
        self, human: HumanModel, fields: Mapping[str, torch.Tensor]
    ) -> tuple[Identity, torch.Tensor, dict[str, torch.Tensor]]:
        rig = human.spec.rig
        batch = torch.broadcast_shapes(*(value.shape[:-1] for value in fields.values()))
        coefficients = fields["identity_coeffs"].expand(*batch, fields["identity_coeffs"].shape[-1])
        model_parameters = fields["model_parameters"].expand(*batch, _MHR_PARAMETER_COUNT)
        identity = human.bake(
            identity=coefficients, proportions=model_parameters.index_select(-1, rig.proportion_columns)
        )
        streams = {name: fields[name].expand(*batch, width) for name, width in human.spec.stream_groups.items()}
        root = model_parameters[..., :ROOT_COUNT] @ rig.root_transform.mT
        pose = model_parameters.index_select(-1, rig.pose_columns) * rig.host_weights
        return identity, torch.cat((root[..., :3], so3.from_euler(root[..., 3:]), pose), dim=-1), streams

    def to_fields(
        self, identity: Identity, q: torch.Tensor, *, streams: Mapping[str, torch.Tensor]
    ) -> dict[str, torch.Tensor | None]:
        rig = identity.human.spec.rig
        batch = q.shape[:-1]
        root_quaternion = so3.normalize(q[..., 3:7])
        root_euler = so3.to_euler(root_quaternion)
        flip = (so3.from_euler(root_euler) * root_quaternion).sum(dim=-1, keepdim=True) < 0
        roll = root_euler[..., :1] + torch.where(
            flip, root_euler.new_tensor(2.0 * torch.pi), root_euler.new_tensor(0.0)
        )
        root_channels = torch.cat((q[..., :3], roll, root_euler[..., 1:]), dim=-1)
        root_parameters = root_channels @ torch.linalg.inv(rig.root_transform).mT
        model_parameters = torch.cat(
            (root_parameters, q.new_zeros((*batch, _MHR_PARAMETER_COUNT - ROOT_COUNT))), dim=-1
        )
        model_parameters = model_parameters.index_copy(-1, rig.pose_columns, q[..., 7:] / rig.host_weights)
        proportions = identity.values["proportions"].expand(*batch, len(rig.proportion_columns))
        model_parameters = model_parameters.index_copy(-1, rig.proportion_columns, proportions)
        coefficients = identity.values["identity"]
        return {
            "identity_coeffs": coefficients.expand(*batch, coefficients.shape[-1]),
            "model_parameters": model_parameters,
            **streams,
        }


@final
@dataclass(frozen=True)
class SOMAFormat(ClassicFormat):
    """SOMA's reference inputs. ``q`` holds the world-aligned Hips free flyer ``(transl, exp(a_Hips))`` then
    ``exp(orient_j^T a_j)`` per public joint, ``orient_j`` its T-pose world rotation, so ``q_neutral`` is the T-pose.
    The identity bakes over ``identity_coeffs``' own batch; angles come back on the principal branch."""

    classic_type: ClassVar[type] = SOMAClassic

    def field_shapes(self, spec: FamilySpec) -> dict[str, tuple[int, ...]]:
        return {
            "identity_coeffs": (spec.identity_groups["shape"],),
            "poses": (_SOMA_POSE_JOINT_COUNT, 3),
            "transl": (3,),
        }

    def from_fields(
        self, human: HumanModel, fields: Mapping[str, torch.Tensor]
    ) -> tuple[Identity, torch.Tensor, dict[str, torch.Tensor]]:
        poses, transl = fields["poses"], fields["transl"]
        orient = human.spec.rig.t_pose_rotations[2:]
        rotations = so3.exp(torch.cat((poses[..., :1, :], (orient.mT @ poses[..., 1:, :, None]).squeeze(-1)), dim=-2))
        batch = torch.broadcast_shapes(transl.shape[:-1], rotations.shape[:-2])
        q = torch.cat(
            (transl.expand(*batch, 3), rotations.expand(*batch, _SOMA_POSE_JOINT_COUNT, 4).flatten(-2)), dim=-1
        )
        return human.bake(shape=fields["identity_coeffs"]), q, {}

    def to_fields(
        self, identity: Identity, q: torch.Tensor, *, streams: Mapping[str, torch.Tensor]
    ) -> dict[str, torch.Tensor | None]:
        coefficients = identity.values["shape"]
        orient = identity.human.spec.rig.t_pose_rotations[2:]
        rotations = so3.log(so3.normalize(q[..., 3:].unflatten(-1, (_SOMA_POSE_JOINT_COUNT, 4))))
        poses = torch.cat((rotations[..., :1, :], (orient @ rotations[..., 1:, :, None]).squeeze(-1)), dim=-2)
        return {
            "identity_coeffs": coefficients.expand(*q.shape[:-1], coefficients.shape[-1]),
            "poses": poses,
            "transl": q[..., :3].clone(),
        }


def _pelvis(identity: Identity, streams: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """``(..., 3)`` root offset of ``identity`` with the joint motion of the stream groups that move joints."""
    pelvis = identity.root_offset
    for name, value in streams.items():
        basis = identity.human.stream_joint_bases.get(name)
        if basis is not None:
            pelvis = pelvis + torch.einsum("...n,cn->...c", value, basis[0])
    return pelvis


def _axis_angle(quaternion: torch.Tensor) -> torch.Tensor:
    """``(..., 3)`` axis-angles of scalar-last quaternions ``(..., 4)``, keeping the sign of the quaternion.

    Unlike the principal SO(3) logarithm, a negative-scalar quaternion maps to an angle in ``(pi, 2 pi]``, so
    ``so3.exp`` gives back the quaternion itself, not only its rotation.
    """
    quaternion = so3.normalize(quaternion)
    vector = quaternion[..., :3]
    scalar = quaternion[..., 3:4]
    sin_half = vector.norm(dim=-1, keepdim=True)
    positive_singularity = (sin_half < 1e-7) & (scalar >= 0)
    safe_sin_half = torch.where(positive_singularity, torch.ones_like(sin_half), sin_half)
    angle = 2.0 * torch.atan2(safe_sin_half, scalar)
    full_factor = angle / safe_sin_half.clamp(min=1e-30)
    taylor_factor = 2.0 + (2.0 / 3.0) * sin_half.square()
    axis_angle = torch.where(positive_singularity, taylor_factor, full_factor) * vector

    negative_singularity = (sin_half < 1e-7) & (scalar < 0)
    fallback = torch.zeros_like(axis_angle)
    fallback[..., 0] = 2.0 * torch.pi
    return torch.where(negative_singularity.expand_as(axis_angle), fallback, axis_angle)


__all__ = [
    "FLAMEClassic",
    "MANOClassic",
    "MHRClassic",
    "MHRFormat",
    "SMPLClassic",
    "SMPLFamilyFormat",
    "SMPLHClassic",
    "SMPLXClassic",
    "SOMAClassic",
    "SOMAFormat",
]
