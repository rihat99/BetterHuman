"""``HumanModel`` (a family ready to bake), ``Identity`` (one subject, baked once), ``HumanState`` (one posed batch).

All three are frozen and compare by identity (``eq=False``): they hold tensors, whose ``==`` is elementwise.
``HumanState`` computes its outputs on first use and keeps them in ``_memo``, keyed by the output and the grad mode
(``"<key>/<torch.is_grad_enabled()>"``): ``joint_transforms``, ``skin_affines``, ``corrective_features``,
``vertices``. A value computed under ``no_grad`` never serves a grad-enabled caller.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from numbers import Real

import torch
from better_robot import Model
from better_robot.exceptions import UnknownNameError
from better_robot.kinematics import KinematicState, forward_kinematics
from better_robot.lie import so3

from .contact_frames import ContactFrames, frame_local_placements, resolve_parents
from .mass import PartMeshTopology, body_inertias_from_shape, build_authored_topology, build_part_topology
from .skinning import blend_shapes, linear_blend_skinning, skin_affines
from .rig import RigBake
from .spec import FamilySpec

Tensors = Mapping[str, torch.Tensor]


@dataclass(frozen=True, eq=False)
class HumanModel:
    """A body family ready to bake subjects: the BetterRobot model, skinning, segmentation, mass and contact frames.

    Build it with :meth:`from_spec` (or ``better_human.load``); the dtype and device are the spec's.
    """

    spec: FamilySpec
    model: Model  #: the BetterRobot model at zero identity, contact frames attached.
    skin_matrix: torch.Tensor  #: ``(V, J)`` dense skin weights.
    joint_ids: torch.Tensor  #: ``(J,)`` BetterRobot joint id of each native joint.
    part_body_ids: torch.Tensor  #: ``(P,)`` BetterRobot body id carrying each part.
    density: torch.Tensor | None  #: ``()`` or ``(P,)`` kg/m³; ``None``: the baked models carry no inertias.
    part_topology: PartMeshTopology | None  #: closed part meshes for the inertias; ``None`` without density.
    contact_frames: ContactFrames | None  #: frames attached to the model, or ``None``.
    stream_joint_bases: Tensors  #: stream group -> ``(J, 3, n)`` joint motion; groups that move no joint are absent.

    @classmethod
    def from_spec(
        cls,
        spec: FamilySpec,
        *,
        density: float | torch.Tensor | None = 1000.0,
        contact_frames: Mapping[str, tuple[int, str | None]] | None = None,
        stream_joints: bool = True,
    ) -> HumanModel:
        """Compute every input from ``spec``, then construct once.

        ``density`` is kg/m³: a scalar, one value per part ``(P,)``, or ``None`` for no inertias. ``contact_frames``
        maps ``name -> (vertex_id, parent joint name | None)`` (a file: ``better_human.io.read_frame_set``); each
        becomes an operational frame of the model, re-placed on every bake. ``stream_joints=False`` lets stream groups
        (expression) move vertices but not joints: faster, slightly off upstream.
        """
        model = spec.rig.build_model(spec, frames=())
        density = None if density is None else _density(density, spec.parts.count, spec.template_vertices)
        zero = None
        if contact_frames is not None or density is not None:
            zero = _bake(spec, model, _checked_groups({}, spec.identity_groups, spec.template_vertices))
        frames = None
        if contact_frames is not None:
            model, frames = _attach(spec, contact_frames, zero)
        topology = None
        if density is not None:
            closed = spec.parts.closed_faces
            topology = (
                build_authored_topology(closed, spec.num_vertices, zero.rest_vertices, zero.rest_joint_translations)
                if closed is not None
                else build_part_topology(spec.faces, spec.parts.ids, spec.parts.count)
            )
        stream_bases = {}
        if stream_joints:
            for name in spec.stream_bases:
                basis = spec.rig.stream_joint_basis(spec, name)
                if basis is not None:
                    stream_bases[name] = basis
        part_body_ids = [model.body_id(spec.joint_names[joint]) for joint in spec.parts.joints]
        return cls(
            spec=spec,
            model=model,
            skin_matrix=spec.skin_matrix(),
            joint_ids=spec.rig.joint_ids(spec, model),
            part_body_ids=torch.tensor(part_body_ids, dtype=torch.long, device=spec.device),
            density=density,
            part_topology=topology,
            contact_frames=frames,
            stream_joint_bases=stream_bases,
        )

    def bake(self, **identity: torch.Tensor) -> Identity:
        """Bake identity groups ``(..., n)`` (omitted groups are zero; leading dims broadcast) into an :class:`Identity`.

        Raises :class:`~better_robot.exceptions.UnknownNameError` for a group the family does not have.
        """
        spec = self.spec
        for name in identity:
            if name not in spec.identity_bases:
                raise UnknownNameError("identity group", name, list(spec.identity_bases))
        values = _checked_groups(identity, spec.identity_groups, spec.template_vertices)
        bake = _bake(spec, self.model, values)
        model = self.model.with_values(joint_placements=bake.joint_placements)
        model_values = {}
        if self.part_topology is not None:
            # The parts are integrated in the neutral configuration: their frames and the mesh posed there.
            neutral = forward_kinematics(model, model.q_neutral)
            transforms = spec.rig.joint_transforms(spec, model, kinematics=neutral)
            vertices = linear_blend_skinning(
                bake.rest_vertices, skin_affines(transforms, bake.inverse_bind_affines), weights=self.skin_matrix
            )
            part_joints = self.joint_ids.index_select(0, torch.as_tensor(spec.parts.joints, device=spec.device))
            part_poses = neutral.joint_poses.index_select(-2, part_joints)
            native = body_inertias_from_shape(vertices, part_poses, self.part_topology, self.density)
            inertias = native.new_zeros((*native.shape[:-2], self.model.nbodies, 10))
            model_values["body_inertias"] = inertias.index_copy(-2, self.part_body_ids, native[..., 1:, :])
        if self.contact_frames is not None:
            frames = self.contact_frames
            local = frame_local_placements(
                bake.rest_vertices, bake.rest_joint_translations, spec.faces, frames.vertex_ids, frames.parent_joints
            )
            placements = local.new_zeros((*local.shape[:-2], self.model.nframes, 7))
            placements[..., 6] = 1.0
            placements[..., torch.as_tensor(frames.frame_ids, device=spec.device), :] = local
            model_values["frame_placements"] = placements
        return Identity(
            human=self,
            values=values,
            rest_vertices=bake.rest_vertices,
            rest_joint_translations=bake.rest_joint_translations,
            inverse_bind_affines=bake.inverse_bind_affines,
            root_offset=bake.root_offset,
            model=model.with_values(**model_values) if model_values else model,
        )

    def __repr__(self) -> str:
        spec = self.spec
        return (
            f"HumanModel({spec.name}, vertices={spec.num_vertices}, joints={spec.num_joints}, "
            f"dtype={spec.dtype}, device={spec.device})"
        )


@dataclass(frozen=True, eq=False)
class Identity:
    """One subject (or an identity batch ``I...``): rest mesh, rest joints and the baked BetterRobot model."""

    human: HumanModel
    values: Tensors  #: identity group values ``(I..., n)``.
    rest_vertices: torch.Tensor  #: ``(I..., V, 3)`` in the skinning rest frame (root at the origin).
    rest_joint_translations: torch.Tensor  #: ``(I..., J, 3)`` native joints in the skinning rest frame.
    inverse_bind_affines: torch.Tensor  #: ``(I..., J, 3, 4)`` rest frame into each joint's frame (``RigBake``).
    root_offset: torch.Tensor  #: ``(I..., 3)`` root from the native origin; ``transl + root_offset`` places the root.
    model: Model  #: placements, inertias and contact frames baked.

    def forward(self, q: torch.Tensor, streams: Tensors | None = None) -> HumanState:
        """Pose ``q (B..., nq)`` with optional stream groups ``(B..., n)``; ``B...`` broadcasts with ``I...``.

        Raises :class:`~better_robot.exceptions.UnknownNameError` for a stream group the family does not have.
        """
        spec = self.human.spec
        streams = dict(streams or {})
        for name in streams:
            if name not in spec.stream_bases:
                raise UnknownNameError("stream group", name, list(spec.stream_bases))
        streams = _checked_groups(streams, spec.stream_groups, spec.template_vertices, require_all=False)
        offsets = None
        delta = None
        for name, coefficients in streams.items():
            term = blend_shapes(coefficients, spec.stream_bases[name])
            offsets = term if offsets is None else offsets + term
            basis = self.human.stream_joint_bases.get(name)
            if basis is not None:
                motion = torch.einsum("...n,jcn->...jc", coefficients, basis)
                delta = motion if delta is None else delta + motion
        model, rest_joints, root_offset = self.model, self.rest_joint_translations, self.root_offset
        inverse_binds = self.inverse_bind_affines
        if delta is not None:
            root = delta[..., :1, :]
            rest_joints = rest_joints + delta - root
            offsets = offsets - root
            root_offset = root_offset + root[..., 0, :]
            placements = spec.rig.placements_from_rest(spec, model, rest_joint_translations=rest_joints)
            model = model.with_values(joint_placements=placements)
            # A rig that moves joints with streams binds each joint at its rest translation (Rig docstring).
            rotations = inverse_binds[..., :3]
            translations = -(rotations @ rest_joints[..., None])
            inverse_binds = torch.cat((rotations.expand(*translations.shape[:-1], 3), translations), dim=-1)
        kinematics = forward_kinematics(model, q)
        return HumanState(self, q, streams, model, kinematics, rest_joints, inverse_binds, root_offset, offsets)

    def __repr__(self) -> str:
        return f"Identity({self.human.spec.name}, batch={tuple(self.root_offset.shape[:-1])})"


@dataclass(frozen=True, eq=False)
class HumanState:
    """One posed batch. Outputs are computed on first use and memoised; shapes lead with the execution batch."""

    identity: Identity
    q: torch.Tensor
    streams: Tensors  #: stream group values ``(..., n)`` as given (omitted groups are absent).
    model: Model  #: the model the kinematics were computed with (per-frame placements when streams move joints).
    kinematics: KinematicState
    rest_joint_translations: torch.Tensor  #: ``(..., J, 3)`` rest joints after the stream joint shift.
    inverse_bind_affines: torch.Tensor  #: ``(..., J, 3, 4)`` the identity's, shifted with the rest joints.
    root_offset: torch.Tensor  #: ``(..., 3)``
    stream_offsets: torch.Tensor | None  #: ``(..., V, 3)`` stream vertex offsets, if any.
    # init=False: dataclasses.replace (e.g. a new identity) starts a fresh memo instead of sharing the old one.
    _memo: dict[str, torch.Tensor] = field(default_factory=dict, init=False, repr=False)

    @property
    def joint_poses(self) -> torch.Tensor:
        """``(..., J, 7)`` world joint poses in native order (``joint_transforms`` as poses)."""
        transforms = self.joint_transforms
        return torch.cat((transforms[..., 3], so3.from_matrix(transforms[..., :3])), dim=-1)

    @property
    def joint_translations(self) -> torch.Tensor:
        """``(..., J, 3)`` world joint positions in native order."""
        return self.kinematics.joint_translations.index_select(-2, self.identity.human.joint_ids)

    @property
    def joint_transforms(self) -> torch.Tensor:
        """``(..., J, 3, 4)`` world ``[R | p]`` of the native joints, as the rig's ``joint_transforms`` reports them."""
        spec = self.identity.human.spec
        transforms = self._memoised(
            "joint_transforms", lambda: spec.rig.joint_transforms(spec, self.model, kinematics=self.kinematics)
        )
        return transforms.clone()  # the memo feeds skinning: editing the result in place must not move vertices

    @property
    def skin_affines(self) -> torch.Tensor:
        """``(..., J, 3, 4)`` affine map of every joint from the rest frame to the world."""
        return self._memoised("skin_affines", lambda: skin_affines(self.joint_transforms, self.inverse_bind_affines))

    @property
    def corrective_feature(self) -> torch.Tensor | None:
        """``(..., P)`` features the family's corrective reads, or ``None`` for a family without one."""
        spec = self.identity.human.spec
        if spec.corrective is None:
            return None
        return self._memoised(
            "corrective_features",
            lambda: spec.rig.corrective_features(spec, self.model, q=self.q, identity=self.identity.values),
        )

    def rest_vertices(self, vertex_ids: torch.Tensor | None = None) -> torch.Tensor:
        """``(..., V, 3)`` (or the ``vertex_ids`` subset) before skinning: shape + streams + corrective."""
        vertices = self.identity.rest_vertices
        offsets = self.stream_offsets
        if vertex_ids is not None:
            vertices = vertices[..., vertex_ids, :]
            offsets = None if offsets is None else offsets[..., vertex_ids, :]
        if offsets is not None:
            vertices = vertices + offsets
        feature = self.corrective_feature
        corrective = None if feature is None else self.identity.human.spec.corrective.apply(feature, vertex_ids)
        return vertices if corrective is None else vertices + corrective

    @property
    def vertices(self) -> torch.Tensor:
        """``(..., V, 3)`` posed mesh."""
        return self._memoised(
            "vertices",
            lambda: linear_blend_skinning(
                self.rest_vertices(), self.skin_affines, weights=self.identity.human.skin_matrix
            ),
        )

    def points(self, vertex_ids: torch.Tensor) -> torch.Tensor:
        """``(..., M, 3)`` posed positions of ``vertex_ids (M,)``, skinning only those vertices."""
        vertex_ids = torch.as_tensor(vertex_ids, dtype=torch.long, device=self.q.device)
        skin = self.identity.human.skin_matrix[vertex_ids]
        return linear_blend_skinning(self.rest_vertices(vertex_ids), self.skin_affines, weights=skin)

    def landmarks(self) -> torch.Tensor:
        """``(..., L, 3)`` posed landmarks of the family (``spec.landmarks``)."""
        return self.points(self.identity.human.spec.landmarks.vertex_ids)

    @property
    def frame_poses(self) -> torch.Tensor:
        """``(..., nframes, 7)`` world poses of the model frames (contact frames included)."""
        return self.kinematics.frame_poses

    def _memoised(self, key: str, compute: Callable[[], torch.Tensor]) -> torch.Tensor:
        key = f"{key}/{torch.is_grad_enabled()}"
        if key not in self._memo:
            self._memo[key] = compute()
        return self._memo[key]

    def __repr__(self) -> str:
        return f"HumanState({self.identity.human.spec.name}, batch={tuple(self.q.shape[:-1])})"


def _bake(spec: FamilySpec, model: Model, values: Tensors) -> RigBake:
    """The rig bake of identity group ``values``: the template plus the identity blend shapes, handed to the rig."""
    shaped = spec.template_vertices
    for name, coefficients in values.items():
        shaped = shaped + blend_shapes(coefficients, spec.identity_bases[name])
    return spec.rig.bake(spec, model, shaped_vertices=shaped, identity=values)


def _attach(
    spec: FamilySpec, source: Mapping[str, tuple[int, str | None]], bake: RigBake
) -> tuple[Model, ContactFrames]:
    """The model with the contact frames of ``source`` attached, and the frames resolved against it."""
    entries = dict(source)
    parents = resolve_parents(entries, spec.joint_names, spec.parts.ids)
    vertex_ids = tuple(int(vertex) for vertex, _ in entries.values())
    placements = frame_local_placements(
        bake.rest_vertices, bake.rest_joint_translations, spec.faces, vertex_ids, parents
    )
    frames = [
        (name, spec.joint_names[parent], placement) for name, parent, placement in zip(entries, parents, placements)
    ]
    model = spec.rig.build_model(spec, frames=frames)
    resolved = ContactFrames(
        names=tuple(entries),
        vertex_ids=vertex_ids,
        parent_joints=tuple(parents),
        frame_ids=tuple(model.frame_id(name) for name in entries),
    )
    return model, resolved


def _checked_groups(
    values: Tensors, widths: Mapping[str, int], exemplar: torch.Tensor, *, require_all: bool = True
) -> dict[str, torch.Tensor]:
    """Validate group tensors against ``widths``; fill omitted groups with zeros when ``require_all``."""
    out: dict[str, torch.Tensor] = {}
    for name, width in widths.items():
        value = values.get(name)
        if value is None:
            if require_all:
                out[name] = exemplar.new_zeros((width,))
            continue
        if not isinstance(value, torch.Tensor) or not value.is_floating_point():
            raise TypeError(f"{name} must be a floating torch.Tensor")
        if value.shape[-1:] != (width,):
            raise ValueError(f"{name} must have shape (..., {width}); got {tuple(value.shape)}")
        if value.dtype != exemplar.dtype or value.device != exemplar.device:
            raise ValueError(f"{name} must use dtype={exemplar.dtype} and device={exemplar.device}")
        out[name] = value
    if require_all and out:
        batch = torch.broadcast_shapes(*(value.shape[:-1] for value in out.values()))
        out = {name: value.expand(*batch, widths[name]) for name, value in out.items()}
    return out


def _density(density: float | torch.Tensor, num_parts: int, exemplar: torch.Tensor) -> torch.Tensor:
    if isinstance(density, Real) and not isinstance(density, bool):
        value = exemplar.new_tensor(float(density))
    elif isinstance(density, torch.Tensor) and density.is_floating_point():
        if density.shape not in ((), (num_parts,)):
            raise ValueError(f"density must be scalar or ({num_parts},), got {tuple(density.shape)}")
        value = density.to(dtype=exemplar.dtype, device=exemplar.device)
    else:
        raise TypeError("density must be a positive real number or floating torch.Tensor")
    if not bool(torch.isfinite(value).all()) or bool((value <= 0).any()):
        raise ValueError("density must contain only finite positive values")
    return value


__all__ = ["HumanModel", "HumanState", "Identity"]
