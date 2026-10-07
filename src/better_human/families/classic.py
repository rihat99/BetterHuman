"""Classic SMPL-family parameters (axis-angle blocks, ``transl``, hand PCA) <-> ``(q, stream)``.

Every family has one ``XClassic`` dataclass, ``from_classic(identity, classic) -> (q, stream)`` and
``to_classic(identity, q, stream=None) -> XClassic``. Omitted classic fields are zero. The
expression of SMPL-X and FLAME is a stream group here, not shape state.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from better_human.core.body import Identity
from better_human.core.classic import (
    axis_angle_to_quaternion,
    classic_root_to_free_flyer,
    classic_tensor,
    free_flyer_to_classic_root,
    quaternion_to_axis_angle,
)

from . import tables

HAND_DIM = 45
Stream = dict[str, torch.Tensor]


def _pca_width(num_pca_comps: int | None) -> int:
    if num_pca_comps is None:
        return HAND_DIM
    if not isinstance(num_pca_comps, int) or isinstance(num_pca_comps, bool) or not 1 <= num_pca_comps <= HAND_DIM:
        raise ValueError(f"num_pca_comps must be in [1, {HAND_DIM}] or None")
    return num_pca_comps


def _zeros(value: torch.Tensor | None, batch: tuple[int, ...], width: int, exemplar: torch.Tensor) -> torch.Tensor:
    return exemplar.new_zeros((*batch, width)) if value is None else value.expand(*batch, width)


def _batch(identity: Identity, *values: torch.Tensor | None) -> tuple[int, ...]:
    return torch.broadcast_shapes(identity.pelvis.shape[:-1], *(v.shape[:-1] for v in values if v is not None))


def _hand_axes(
    identity: Identity, side: str, pose: torch.Tensor | None, num_pca_comps: int | None, batch: tuple[int, ...]
) -> torch.Tensor:
    """``(..., 15, 3)`` finger axis-angles from a full or PCA hand pose plus the hand mean."""
    meta = identity.spec.meta
    exemplar = identity.pelvis
    width = _pca_width(num_pca_comps)
    delta = _zeros(pose, batch, width, exemplar)
    if num_pca_comps is not None:
        delta = delta @ meta["hand_components"][side][:width]
    return (delta + meta["hand_mean"][side]).unflatten(-1, (15, 3))


def _hand_pose(identity: Identity, side: str, axes: torch.Tensor, num_pca_comps: int | None) -> torch.Tensor:
    """Inverse of :func:`_hand_axes`; raises when ``axes`` leave the requested PCA subspace."""
    meta = identity.spec.meta
    delta = axes.flatten(-2) - meta["hand_mean"][side]
    if num_pca_comps is None:
        return delta
    components = meta["hand_components"][side][: _pca_width(num_pca_comps)]
    pose = delta @ torch.linalg.pinv(components)
    if not torch.allclose(pose @ components, delta, rtol=2e-5, atol=2e-5):
        raise ValueError(f"{side} hand pose is outside the first {num_pca_comps} PCA components")
    return pose


def _assemble(
    identity: Identity,
    global_orient: torch.Tensor | None,
    axes: torch.Tensor,
    transl: torch.Tensor | None,
    stream: Stream,
) -> tuple[torch.Tensor, Stream]:
    """``q`` from the root and native non-root axis-angles ``axes (..., J_native - 1, 3)``."""
    exemplar = identity.pelvis
    batch = axes.shape[:-2]
    global_orient = _zeros(global_orient, batch, 3, exemplar)
    transl = _zeros(transl, batch, 3, exemplar)
    pelvis = identity.pelvis
    for name, value in stream.items():
        basis = identity.stream_joint_basis.get(name)
        if basis is not None:
            pelvis = pelvis + torch.einsum("...n,cn->...c", value, basis[0])
    native_ids = identity.spec.meta["native_joint_ids"]
    reduced = axes.index_select(-2, native_ids[1:] - 1)
    if reduced.shape[-2] < axes.shape[-2]:
        keep = torch.zeros(axes.shape[-2], dtype=torch.bool, device=axes.device)
        keep[native_ids[1:] - 1] = True
        if bool(axes[..., ~keep, :].any()):
            raise ValueError("pose given for joints this body was loaded without (hands=False / face=False)")
    root = classic_root_to_free_flyer(global_orient, transl, pelvis.expand(*batch, 3))
    return torch.cat((root, axis_angle_to_quaternion(reduced).flatten(-2)), dim=-1), stream


def _disassemble(
    identity: Identity, q: torch.Tensor, stream: Stream, num_native: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(global_orient, axes (..., J_native - 1, 3), transl)`` from ``q``; dropped joints read zero."""
    if q.shape[-1:] != (identity.nq,):
        raise ValueError(f"q must have shape (..., {identity.nq}); got {tuple(q.shape)}")
    batch = torch.broadcast_shapes(q.shape[:-1], identity.pelvis.shape[:-1])
    q = q.expand(*batch, identity.nq)
    pelvis = identity.pelvis
    for name, value in stream.items():
        basis = identity.stream_joint_basis.get(name)
        if basis is not None:
            pelvis = pelvis + torch.einsum("...n,cn->...c", value, basis[0])
    global_orient, transl = free_flyer_to_classic_root(q[..., :7], pelvis.expand(*batch, 3))
    reduced = quaternion_to_axis_angle(q[..., 7:].unflatten(-1, (-1, 4)))
    axes = q.new_zeros((*batch, num_native - 1, 3))
    native_ids = identity.spec.meta["native_joint_ids"]
    axes = axes.index_copy(-2, native_ids[1:] - 1, reduced)
    return global_orient, axes, transl


def _stream(identity: Identity, name: str, value: torch.Tensor | None, batch: tuple[int, ...]) -> Stream:
    width = identity.spec.stream_groups.get(name)
    if width is None:
        if value is not None:
            raise ValueError(f"{name} requires a body loaded with that stream group")
        return {}
    return {name: _zeros(classic_tensor(name, value, width, identity.pelvis), batch, width, identity.pelvis)}


# ----------------------------------------------------------------------------------------- SMPL


@dataclass(frozen=True)
class SMPLClassic:
    """Classic SMPL parameters with arbitrary leading batch dimensions."""

    global_orient: torch.Tensor | None = None  #: ``(..., 3)``
    body_pose: torch.Tensor | None = None  #: ``(..., 69)``
    transl: torch.Tensor | None = None  #: ``(..., 3)``


def smpl_from_classic(identity: Identity, classic: SMPLClassic) -> tuple[torch.Tensor, Stream]:
    exemplar = identity.pelvis
    body_pose = classic_tensor("body_pose", classic.body_pose, 69, exemplar)
    batch = _batch(identity, classic.global_orient, body_pose, classic.transl)
    axes = _zeros(body_pose, batch, 69, exemplar).unflatten(-1, (23, 3))
    return _assemble(identity, classic.global_orient, axes, classic.transl, {})


def smpl_to_classic(identity: Identity, q: torch.Tensor, stream: Stream | None = None) -> SMPLClassic:
    global_orient, axes, transl = _disassemble(identity, q, {}, 24)
    return SMPLClassic(global_orient=global_orient, body_pose=axes.flatten(-2), transl=transl)


# --------------------------------------------------------------------------------------- SMPL-H


@dataclass(frozen=True)
class SMPLHClassic:
    """Classic SMPL+H parameters; hand poses are full 45-D axis-angle blocks."""

    global_orient: torch.Tensor | None = None
    body_pose: torch.Tensor | None = None  #: ``(..., 63)``
    left_hand_pose: torch.Tensor | None = None  #: ``(..., 45)``
    right_hand_pose: torch.Tensor | None = None  #: ``(..., 45)``
    transl: torch.Tensor | None = None


def smplh_from_classic(identity: Identity, classic: SMPLHClassic) -> tuple[torch.Tensor, Stream]:
    exemplar = identity.pelvis
    body_pose = classic_tensor("body_pose", classic.body_pose, 63, exemplar)
    left = classic_tensor("left_hand_pose", classic.left_hand_pose, HAND_DIM, exemplar)
    right = classic_tensor("right_hand_pose", classic.right_hand_pose, HAND_DIM, exemplar)
    batch = _batch(identity, classic.global_orient, body_pose, left, right, classic.transl)
    axes = torch.cat(
        (
            _zeros(body_pose, batch, 63, exemplar),
            _zeros(left, batch, HAND_DIM, exemplar),
            _zeros(right, batch, HAND_DIM, exemplar),
        ),
        dim=-1,
    ).unflatten(-1, (51, 3))
    return _assemble(identity, classic.global_orient, axes, classic.transl, {})


def smplh_to_classic(identity: Identity, q: torch.Tensor, stream: Stream | None = None) -> SMPLHClassic:
    global_orient, axes, transl = _disassemble(identity, q, {}, 52)
    flat = axes.flatten(-2)
    return SMPLHClassic(
        global_orient=global_orient,
        body_pose=flat[..., :63],
        left_hand_pose=flat[..., 63:108],
        right_hand_pose=flat[..., 108:153],
        transl=transl,
    )


# --------------------------------------------------------------------------------------- SMPL-X


@dataclass(frozen=True)
class SMPLXClassic:
    """Classic SMPL-X parameters with arbitrary leading batch dimensions."""

    global_orient: torch.Tensor | None = None  #: ``(..., 3)``
    body_pose: torch.Tensor | None = None  #: ``(..., 63)``
    jaw_pose: torch.Tensor | None = None  #: ``(..., 3)``
    leye_pose: torch.Tensor | None = None  #: ``(..., 3)``
    reye_pose: torch.Tensor | None = None  #: ``(..., 3)``
    left_hand_pose: torch.Tensor | None = None  #: ``(..., 45)`` or ``(..., num_pca_comps)``
    right_hand_pose: torch.Tensor | None = None
    expression: torch.Tensor | None = None  #: ``(..., num_expression)``
    transl: torch.Tensor | None = None  #: ``(..., 3)``
    num_pca_comps: int | None = None  #: hand PCA width, or ``None`` for full axis-angle hands.


def smplx_from_classic(identity: Identity, classic: SMPLXClassic) -> tuple[torch.Tensor, Stream]:
    exemplar = identity.pelvis
    table = tables.SMPLX
    hand_width = _pca_width(classic.num_pca_comps)
    fields = {
        "body_pose": (classic.body_pose, 63),
        "jaw_pose": (classic.jaw_pose, 3),
        "leye_pose": (classic.leye_pose, 3),
        "reye_pose": (classic.reye_pose, 3),
        "left_hand_pose": (classic.left_hand_pose, hand_width),
        "right_hand_pose": (classic.right_hand_pose, hand_width),
    }
    checked = {name: classic_tensor(name, value, width, exemplar) for name, (value, width) in fields.items()}
    batch = _batch(identity, classic.global_orient, classic.transl, classic.expression, *checked.values())
    axes = torch.cat(
        (
            _zeros(checked["body_pose"], batch, 63, exemplar).unflatten(-1, (21, 3)),
            torch.stack(
                [_zeros(checked[name], batch, 3, exemplar) for name in ("jaw_pose", "leye_pose", "reye_pose")], dim=-2
            ),
            _hand_axes(identity, "left", checked["left_hand_pose"], classic.num_pca_comps, batch),
            _hand_axes(identity, "right", checked["right_hand_pose"], classic.num_pca_comps, batch),
        ),
        dim=-2,
    )
    assert axes.shape[-2] == len(table.joint_names) - 1
    stream = _stream(identity, "expression", classic.expression, batch)
    return _assemble(identity, classic.global_orient, axes, classic.transl, stream)


def smplx_to_classic(
    identity: Identity, q: torch.Tensor, stream: Stream | None = None, *, num_pca_comps: int | None = None
) -> SMPLXClassic:
    stream = dict(stream or {})
    global_orient, axes, transl = _disassemble(identity, q, stream, 55)
    face = axes[..., 21:24, :]
    return SMPLXClassic(
        global_orient=global_orient,
        body_pose=axes[..., :21, :].flatten(-2),
        jaw_pose=face[..., 0, :],
        leye_pose=face[..., 1, :],
        reye_pose=face[..., 2, :],
        left_hand_pose=_hand_pose(identity, "left", axes[..., 24:39, :], num_pca_comps),
        right_hand_pose=_hand_pose(identity, "right", axes[..., 39:54, :], num_pca_comps),
        expression=stream.get("expression"),
        transl=transl,
        num_pca_comps=num_pca_comps,
    )


# ----------------------------------------------------------------------------------------- MANO


@dataclass(frozen=True)
class MANOClassic:
    """Classic MANO parameters: ``hand_pose`` is ``(..., 45)`` or ``(..., num_pca_comps)``."""

    global_orient: torch.Tensor | None = None
    hand_pose: torch.Tensor | None = None
    transl: torch.Tensor | None = None
    num_pca_comps: int | None = None


def mano_from_classic(identity: Identity, classic: MANOClassic) -> tuple[torch.Tensor, Stream]:
    side = identity.spec.meta["side"]
    hand_pose = classic_tensor("hand_pose", classic.hand_pose, _pca_width(classic.num_pca_comps), identity.pelvis)
    batch = _batch(identity, classic.global_orient, hand_pose, classic.transl)
    axes = _hand_axes(identity, side, hand_pose, classic.num_pca_comps, batch)
    return _assemble(identity, classic.global_orient, axes, classic.transl, {})


def mano_to_classic(
    identity: Identity, q: torch.Tensor, stream: Stream | None = None, *, num_pca_comps: int | None = None
) -> MANOClassic:
    global_orient, axes, transl = _disassemble(identity, q, {}, 16)
    hand_pose = _hand_pose(identity, identity.spec.meta["side"], axes, num_pca_comps)
    return MANOClassic(global_orient=global_orient, hand_pose=hand_pose, transl=transl, num_pca_comps=num_pca_comps)


# ---------------------------------------------------------------------------------------- FLAME


@dataclass(frozen=True)
class FLAMEClassic:
    """Classic FLAME parameters (neck, jaw and eye axis-angles, expression stream)."""

    global_orient: torch.Tensor | None = None
    neck_pose: torch.Tensor | None = None
    jaw_pose: torch.Tensor | None = None
    leye_pose: torch.Tensor | None = None
    reye_pose: torch.Tensor | None = None
    expression: torch.Tensor | None = None
    transl: torch.Tensor | None = None


def flame_from_classic(identity: Identity, classic: FLAMEClassic) -> tuple[torch.Tensor, Stream]:
    exemplar = identity.pelvis
    names = ("neck_pose", "jaw_pose", "leye_pose", "reye_pose")
    checked = [classic_tensor(name, getattr(classic, name), 3, exemplar) for name in names]
    batch = _batch(identity, classic.global_orient, classic.transl, classic.expression, *checked)
    axes = torch.stack([_zeros(value, batch, 3, exemplar) for value in checked], dim=-2)
    stream = _stream(identity, "expression", classic.expression, batch)
    return _assemble(identity, classic.global_orient, axes, classic.transl, stream)


def flame_to_classic(identity: Identity, q: torch.Tensor, stream: Stream | None = None) -> FLAMEClassic:
    stream = dict(stream or {})
    global_orient, axes, transl = _disassemble(identity, q, stream, 5)
    neck, jaw, leye, reye = axes.unbind(-2)
    return FLAMEClassic(global_orient, neck, jaw, leye, reye, stream.get("expression"), transl)


__all__ = [
    "FLAMEClassic",
    "MANOClassic",
    "SMPLClassic",
    "SMPLHClassic",
    "SMPLXClassic",
    "flame_from_classic",
    "flame_to_classic",
    "mano_from_classic",
    "mano_to_classic",
    "smpl_from_classic",
    "smpl_to_classic",
    "smplh_from_classic",
    "smplh_to_classic",
    "smplx_from_classic",
    "smplx_to_classic",
]
