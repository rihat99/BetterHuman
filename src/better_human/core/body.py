"""``Body`` (a family on a device), ``Identity`` (baked once per subject), ``HumanState`` (one posed batch)."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from numbers import Real
from pathlib import Path

import torch
from better_robot import Model
from better_robot.kinematics import KinematicState, forward_kinematics_state

from .contact_frames import ContactFrames, frame_local_placements, parse_frame_set, resolve_parents
from .mass import PartMeshTopology, body_inertias_from_shape, build_authored_topology, build_part_topology
from .pipeline import blend_shapes, lbs_affines, skin_affines
from .rig import RegressedRig
from .spec import FamilySpec

Tensors = Mapping[str, torch.Tensor]


class Body:
    """A body family ready to bake subjects: BR model topology, segmentation, optional mass and contact frames.

    ``density`` is kg/m³, a scalar or one value per part. ``contact_frames`` is a frame-set JSON
    path or a mapping ``name -> (vertex_id, parent joint name | None)``; each becomes an
    operational frame of the model, re-placed on every bake. ``stream_joints=False`` lets stream
    groups (expression) move vertices but not joints: faster, slightly off upstream.
    """

    def __init__(
        self,
        spec: FamilySpec,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
        mass: bool = True,
        density: float | torch.Tensor = 1000.0,
        contact_frames: str | Path | Mapping[str, tuple[int, str | None]] | None = None,
        stream_joints: bool = True,
    ) -> None:
        self.spec = spec = spec.to(dtype=dtype, device=device)
        self.stream_joints = stream_joints
        self.skin_matrix = spec.skin_matrix()
        self.density = _density(density, spec.parts.count, spec.vertices)
        self.topology: PartMeshTopology | None = None
        self.model = spec.rig.build_model(spec, frames=(), dtype=spec.dtype, device=spec.device)
        self.contact_frames: ContactFrames | None = None
        self.part_body_ids = torch.tensor(
            [self.model.body_id(spec.joint_names[joint]) for joint in spec.parts.joints],
            dtype=torch.long,
            device=spec.device,
        )
        self.native_joint_ids = spec.rig.native_joint_ids(spec, self.model)
        if contact_frames is not None:
            self._attach_contact_frames(contact_frames)
        if mass:
            closed = spec.meta.get("closed_segmentation")
            zero = self.bake()
            self.topology = (
                build_authored_topology(closed, spec.num_vertices, zero.v_shaped, zero.rest_joints)
                if closed is not None
                else build_part_topology(spec.faces, spec.parts.ids, spec.parts.count)
            )

    # Names the viewer and the terms need without reaching into the spec.
    @property
    def faces(self) -> torch.Tensor:
        return self.spec.faces

    @property
    def joint_names(self) -> tuple[str, ...]:
        return self.spec.joint_names

    @property
    def parents(self) -> tuple[int, ...]:
        return self.spec.parents

    @property
    def num_vertices(self) -> int:
        return self.spec.num_vertices

    @property
    def identity_groups(self) -> dict[str, int]:
        return self.spec.identity_groups

    @property
    def stream_groups(self) -> dict[str, int]:
        return self.spec.stream_groups

    @property
    def dtype(self) -> torch.dtype:
        return self.spec.dtype

    @property
    def device(self) -> torch.device:
        return self.spec.device

    def _attach_contact_frames(self, source: str | Path | Mapping[str, tuple[int, str | None]]) -> None:
        if not isinstance(self.spec.rig, RegressedRig):
            raise ValueError("contact frames need a regressed rig (SMPL family)")
        spec = self.spec
        entries = dict(source) if isinstance(source, Mapping) else parse_frame_set(source, spec.num_vertices)
        parents = resolve_parents(entries, spec.joint_names, spec.parts.ids)
        vertex_ids = tuple(vertex for vertex, _ in entries.values())
        zero = self.bake()
        placements = frame_local_placements(zero.v_shaped, zero.rest_joints, spec.faces, vertex_ids, parents)
        frames = [
            (name, spec.joint_names[parent], placement) for name, parent, placement in zip(entries, parents, placements)
        ]
        self.model = spec.rig.build_model(spec, frames=frames, dtype=spec.dtype, device=spec.device)
        self.contact_frames = ContactFrames(
            names=tuple(entries),
            vertex_ids=vertex_ids,
            parent_joints=tuple(parents),
            frame_ids=tuple(self.model.frame_id(name) for name in entries),
        )

    def bake(self, **identity: torch.Tensor) -> Identity:
        """Bake identity groups ``(..., n)`` (omitted groups are zero; leading dims broadcast) into an :class:`Identity`."""
        spec = self.spec
        unknown = set(identity) - set(spec.identity)
        if unknown:
            raise ValueError(f"unknown identity groups {sorted(unknown)}; expected {list(spec.identity)}")
        values = _checked_groups(identity, spec.identity_groups, spec.vertices)
        v_shaped = spec.vertices
        for name, coefficients in values.items():
            v_shaped = v_shaped + blend_shapes(coefficients, spec.identity[name])
        bake = spec.rig.bake(spec, self.model, v_shaped, values)
        v_shaped = v_shaped - bake.pelvis[..., None, :]  # the skinning rest frame has the root at the origin

        inertias = None
        if self.topology is not None:
            vertices = bake.neutral_vertices if bake.neutral_vertices is not None else v_shaped
            part_poses = bake.rest_poses[..., torch.as_tensor(spec.parts.joints, device=spec.device), :]
            native = body_inertias_from_shape(vertices, part_poses, self.topology, self.density)
            inertias = native.new_zeros((*native.shape[:-2], self.model.nbodies, 10))
            inertias = inertias.index_copy(-2, self.part_body_ids, native[..., 1:, :])

        frame_placements = None
        if self.contact_frames is not None:
            frames = self.contact_frames
            local = frame_local_placements(
                v_shaped, bake.rest_joints, spec.faces, frames.vertex_ids, frames.parent_joints
            )
            frame_placements = local.new_zeros((*local.shape[:-2], self.model.nframes, 7))
            frame_placements[..., 6] = 1.0
            frame_placements[..., torch.as_tensor(frames.frame_ids, device=spec.device), :] = local

        model = self.model.with_values(
            joint_placements=bake.joint_placements, body_inertias=inertias, frame_placements=frame_placements
        )
        stream_joint_basis = {}
        if self.stream_joints:
            for name in spec.streams:
                basis = spec.rig.stream_joint_basis(spec, name)
                if basis is not None:
                    stream_joint_basis[name] = basis
        return Identity(
            body=self,
            params=values,
            v_shaped=v_shaped,
            rest_joints=bake.rest_joints,
            pelvis=bake.pelvis,
            model=model,
            joint_scales=bake.joint_scales,
            stream_joint_basis=stream_joint_basis,
        )


@dataclass(frozen=True)
class Identity:
    """One subject (or an identity batch ``I...``): shaped mesh, rest joints and the baked BR model."""

    body: Body
    params: Tensors  #: identity group values ``(I..., n)``.
    v_shaped: torch.Tensor  #: ``(I..., V, 3)`` in the skinning rest frame (root at the origin).
    rest_joints: torch.Tensor  #: ``(I..., J, 3)``
    pelvis: torch.Tensor  #: ``(I..., 3)`` root offset from the native origin; ``transl + pelvis`` is the root position.
    model: Model  #: placements, inertias and contact frames baked.
    joint_scales: torch.Tensor | None  #: ``(I..., J)`` for authored rigs.
    stream_joint_basis: Tensors  #: stream name -> ``(J, 3, n)`` joint motion.

    @property
    def spec(self) -> FamilySpec:
        return self.body.spec

    @property
    def nq(self) -> int:
        return self.model.nq

    @property
    def nv(self) -> int:
        return self.model.nv

    @property
    def q_neutral(self) -> torch.Tensor:
        return self.model.q_neutral

    def forward(self, q: torch.Tensor, stream: Tensors | None = None) -> HumanState:
        """Pose ``q (B..., nq)`` with optional stream groups ``(B..., n)``; ``B...`` broadcasts with ``I...``."""
        spec = self.spec
        stream = _checked_groups(dict(stream or {}), spec.stream_groups, spec.vertices, require_all=False)
        unknown = set(stream) - set(spec.streams)
        if unknown:
            raise ValueError(f"unknown stream groups {sorted(unknown)}; expected {list(spec.streams)}")
        offset = None
        delta = None
        for name, coefficients in stream.items():
            term = blend_shapes(coefficients, spec.streams[name])
            offset = term if offset is None else offset + term
            basis = self.stream_joint_basis.get(name)
            if basis is not None:
                motion = torch.einsum("...n,jcn->...jc", coefficients, basis)
                delta = motion if delta is None else delta + motion
        model, rest_joints, pelvis = self.model, self.rest_joints, self.pelvis
        if delta is not None:
            root = delta[..., :1, :]
            rest_joints = rest_joints + delta - root
            offset = offset - root
            pelvis = pelvis + root[..., 0, :]
            model = model.with_values(joint_placements=spec.rig.placements_from_rest(spec, model, rest_joints))
        kinematics = forward_kinematics_state(model, q)
        return HumanState(self, q, stream, model, kinematics, rest_joints, pelvis, offset)


@dataclass(frozen=True)
class HumanState:
    """One posed batch. Outputs are computed on first use and memoised; shapes lead with the execution batch."""

    identity: Identity
    q: torch.Tensor
    stream: Tensors
    model: Model  #: the model the kinematics were computed with (per-frame placements when streams move joints).
    kinematics: KinematicState
    rest_joints: torch.Tensor  #: ``(..., J, 3)`` rest joints after the stream joint shift.
    pelvis: torch.Tensor  #: ``(..., 3)``
    rest_offset: torch.Tensor | None  #: ``(..., V, 3)`` stream vertex offsets, if any.
    # init=False: dataclasses.replace (e.g. a new identity) starts a fresh memo instead of sharing the old one.
    _memo: dict[str, torch.Tensor] = field(default_factory=dict, init=False, repr=False, compare=False)

    @property
    def body(self) -> Body:
        return self.identity.body

    @property
    def spec(self) -> FamilySpec:
        return self.identity.spec

    @property
    def joint_poses(self) -> torch.Tensor:
        """``(..., J, 7)`` world joint poses in native order."""
        return self.kinematics.joint_pose.index_select(-2, self.body.native_joint_ids)

    @property
    def joints(self) -> torch.Tensor:
        """``(..., J, 3)`` world joint positions in native order."""
        return self.kinematics.joint_translation.index_select(-2, self.body.native_joint_ids)

    @property
    def joint_transforms(self) -> torch.Tensor:
        """``(..., J, 3, 4)`` world ``[R | p]`` of the native joints."""
        return self.kinematics.joint_transform.index_select(-3, self.body.native_joint_ids)

    @property
    def skin_affines(self) -> torch.Tensor:
        """``(..., J, 3, 4)`` affine map of every joint from the rest frame to the world."""
        spec = self.spec
        inverse_bind = spec.meta.get("inverse_bind_poses")
        return self._memoised(
            "skin_affines",
            lambda: skin_affines(
                self.joint_transforms,
                rest_joints=None if inverse_bind is not None else self.rest_joints,
                inverse_bind_poses=inverse_bind,
                joint_scales=self.identity.joint_scales,
                bind_joint_positions=spec.meta.get("bind_joint_positions"),
            ),
        )

    @property
    def corrective_feature(self) -> torch.Tensor | None:
        if self.spec.corrective.kind == "none":
            return None
        return self._memoised(
            "feature", lambda: self.spec.rig.corrective_features(self.spec, self.model, self.q, identity=self.identity.params)
        )

    def rest_vertices(self, vertex_ids: torch.Tensor | None = None) -> torch.Tensor:
        """``(..., V, 3)`` (or the ``vertex_ids`` subset) before skinning: shape + streams + corrective."""
        vertices = self.identity.v_shaped
        offset = self.rest_offset
        if vertex_ids is not None:
            vertices = vertices[..., vertex_ids, :]
            offset = None if offset is None else offset[..., vertex_ids, :]
        if offset is not None:
            vertices = vertices + offset
        corrective = (
            self.spec.corrective.apply(self.corrective_feature, vertex_ids)
            if self.corrective_feature is not None
            else None
        )
        return vertices if corrective is None else vertices + corrective

    @property
    def vertices(self) -> torch.Tensor:
        """``(..., V, 3)`` posed mesh."""
        return self._memoised(
            "vertices", lambda: lbs_affines(self.rest_vertices(), self.skin_affines, self.body.skin_matrix)
        )

    def _memoised(self, key: str, compute: Callable[[], torch.Tensor]) -> torch.Tensor:
        # A value computed under no_grad must not serve a grad-enabled caller: key by grad mode.
        key = f"{key}/{torch.is_grad_enabled()}"
        if key not in self._memo:
            self._memo[key] = compute()
        return self._memo[key]

    def points(self, vertex_ids: torch.Tensor) -> torch.Tensor:
        """``(..., M, 3)`` posed positions of ``vertex_ids (M,)``, skinning only those vertices."""
        vertex_ids = torch.as_tensor(vertex_ids, dtype=torch.long, device=self.q.device)
        return lbs_affines(self.rest_vertices(vertex_ids), self.skin_affines, self.body.skin_matrix[vertex_ids])

    def landmarks(self, name: str = "keypoints") -> torch.Tensor:
        """``(..., L, 3)`` posed landmark set ``name`` of the family."""
        return self.points(self.spec.landmarks[name])

    @property
    def frame_poses(self) -> torch.Tensor:
        """``(..., nframes, 7)`` world poses of the model frames (contact frames included)."""
        return self.kinematics.frame_pose


def _checked_groups(
    values: dict[str, torch.Tensor], widths: Mapping[str, int], exemplar: torch.Tensor, *, require_all: bool = True
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


__all__ = ["Body", "HumanState", "Identity"]
