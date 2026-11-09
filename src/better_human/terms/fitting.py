"""Fitting terms: joints, markers, landmarks and their 2D reprojections, the ground plane, pose and shape priors."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from better_robot.exceptions import ShapeError, UnknownNameError
from better_robot.kinematics import difference
from better_robot.optim import TermFn

from better_human.core import HumanState, Identity


def joints(*, target: str, state: str = "state", joint_names: Sequence[str] | None = None) -> TermFn:
    """3D joint error against data ``target (..., J, 3)``, or the ``joint_names`` in that order. Residual ``(..., J * 3)``.

    A name the family does not have raises :class:`~better_robot.exceptions.UnknownNameError` on evaluation.
    """
    names = None if joint_names is None else tuple(joint_names)

    def residual(human: HumanState, goal: torch.Tensor) -> torch.Tensor:
        positions = human.joint_translations
        if names is not None:
            known = human.identity.human.spec.joint_names
            for name in names:
                if name not in known:
                    raise UnknownNameError("joint", name, known)
            ids = torch.tensor([known.index(name) for name in names], device=positions.device)
            positions = positions.index_select(-2, ids)
        return (positions - goal).flatten(-2)

    return TermFn(fn=residual, reads=(state, target))


def points(vertex_ids: Sequence[int] | torch.Tensor, *, target: str, state: str = "state") -> TermFn:
    """Marker fit: posed vertices ``vertex_ids (M,)`` against data ``target (..., M, 3)``. Residual ``(..., M * 3)``.

    A stationary contact is the same term with the contact positions as ``target``.
    """
    ids = torch.as_tensor(vertex_ids, dtype=torch.long)

    def residual(human: HumanState, goal: torch.Tensor) -> torch.Tensor:
        return (human.points(ids) - goal).flatten(-2)

    return TermFn(fn=residual, reads=(state, target))


def landmarks(*, target: str, state: str = "state") -> TermFn:
    """The family's landmarks (``spec.landmarks``) against data ``target (..., L, 3)``. Residual ``(..., L * 3)``."""

    def residual(human: HumanState, goal: torch.Tensor) -> torch.Tensor:
        return (human.landmarks() - goal).flatten(-2)

    return TermFn(fn=residual, reads=(state, target))


def reprojection(
    *,
    target: str,
    camera: str = "camera",
    vertex_ids: Sequence[int] | torch.Tensor | None = None,
    joints: bool = False,
    confidence: str | None = None,
    state: str = "state",
) -> TermFn:
    """2D reprojection error of the landmarks (or ``vertex_ids``, or with ``joints=True`` all joints).

    ``target (..., M, 2)`` holds the pixels; ``camera (..., 3, 4)`` is the world-to-pixel projection ``K [R | t]``,
    whose batch axes align with the *leading* batch axes of the points (a per-sequence ``camera (B, 3, 4)`` applies
    to ``(B, T, M, 3)``); ``confidence (..., M)`` scales rows. Residual ``(..., M * 2)``.
    """
    ids = None if vertex_ids is None else torch.as_tensor(vertex_ids, dtype=torch.long)

    def positions(human: HumanState) -> torch.Tensor:
        if ids is not None:
            return human.points(ids)
        return human.joint_translations if joints else human.landmarks()

    def residual(
        human: HumanState, goal: torch.Tensor, projection: torch.Tensor, *weight: torch.Tensor
    ) -> torch.Tensor:
        error = _project(positions(human), projection) - goal
        if weight:
            error = error * weight[0][..., None]
        return error.flatten(-2)

    reads = (state, target, camera) + ((confidence,) if confidence else ())
    return TermFn(fn=residual, reads=reads)


def _project(points: torch.Tensor, camera: torch.Tensor) -> torch.Tensor:
    """Pinhole projection ``(..., M, 2)`` of ``points (..., M, 3)`` with ``camera (..., 3, 4)``, batch axes leading."""
    missing = (points.ndim - 2) - (camera.ndim - 2)
    if missing < 0 or camera.shape[:-2] != points.shape[: camera.ndim - 2]:
        raise ShapeError(
            f"camera batch {tuple(camera.shape[:-2])} must lead the points batch {tuple(points.shape[:-2])}"
        )
    camera = camera.reshape(*camera.shape[:-2], *([1] * missing), 3, 4)
    homogeneous = torch.cat((points, torch.ones_like(points[..., :1])), dim=-1)
    pixels = homogeneous @ camera.mT
    return pixels[..., :2] / pixels[..., 2:3]


def ground(
    vertex_ids: Sequence[int] | torch.Tensor | None = None,
    *,
    height: float = 0.0,
    up: Sequence[float] = (0.0, 1.0, 0.0),
    hinge: bool = True,
    state: str = "state",
) -> TermFn:
    """Ground penetration of ``vertex_ids`` (default: every vertex) below the plane at ``height`` along ``up``.

    Residual ``(..., M)``: ``min(h - height, 0)`` (hinge) or the signed height for :func:`solve_constrained`.
    """
    ids = None if vertex_ids is None else torch.as_tensor(vertex_ids, dtype=torch.long)
    normal = torch.tensor(up, dtype=torch.float64)
    normal = normal / normal.norm()

    def residual(human: HumanState) -> torch.Tensor:
        positions = human.vertices if ids is None else human.points(ids)
        clearance = positions @ normal.to(positions) - height
        return clearance.clamp_max(0.0) if hinge else clearance

    return TermFn(fn=residual, reads=(state,))


def pose_prior(identity: Identity, *, q: str = "q", mean: torch.Tensor | None = None, root: bool = False) -> TermFn:
    """Tangent distance of the joint rotations to ``mean`` (default: the neutral pose). Residual ``(..., nv - 6)``.

    A Gaussian prior with unit precision; scale it with the term weight or pre-whiten ``q``.
    ``root=True`` includes the six free-flyer coordinates.
    """
    model = identity.model
    reference = model.q_neutral if mean is None else mean
    start = 0 if root else 6

    def residual(pose: torch.Tensor) -> torch.Tensor:
        return difference(model, reference.expand_as(pose), pose)[..., start:]

    return TermFn(fn=residual, reads=(q,))


def shape_prior(group: str = "betas") -> TermFn:
    """L2 prior on an identity group variable: the residual is the coefficient vector itself."""
    return TermFn(fn=lambda value: value, reads=(group,))
