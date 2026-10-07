"""Human cost terms for ``better_robot.optim``: plain functions over a cached :class:`HumanState`.

Register the state once per problem (``problem.cached(human_state(identity), name="state")``);
every term below reads it, so one forward pass serves all terms. Pose variables are ``q (B, nq)``
or trajectories ``q (B, T, nq)``; residuals keep the leading batch axes and flatten their event
axes. Identity groups may be variables too (``problem.cached(shaped(body), name="identity")``)
for shape fitting.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from better_robot.dynamics.state import inverse_dynamics
from better_robot import Model
from better_robot.kinematics import KinematicState, forward_kinematics_state
from better_robot.optim import TermFn

from .core.body import Body, HumanState, Identity


def shaped(body: Body, *groups: str) -> TermFn:
    """Cached value: ``body.bake(**groups)`` from variables named after the identity groups (default: all)."""
    names = groups or tuple(body.identity_groups)
    return TermFn(fn=lambda *values: body.bake(**dict(zip(names, values))), reads=names)


def human_state(
    identity: Identity | None = None, *, q: str = "q", stream: Sequence[str] = (), identity_name: str | None = None
) -> TermFn:
    """Cached value: ``identity.forward(q, stream)``.

    Pass ``identity`` for a fixed subject, or ``identity_name`` to read a cached/variable identity
    (see :func:`shaped`). ``stream`` lists variable or data names of per-frame groups.
    """
    if (identity is None) == (identity_name is None):
        raise ValueError("give exactly one of identity and identity_name")
    stream = tuple(stream)

    def fixed(pose: torch.Tensor, *values: torch.Tensor) -> HumanState:
        return identity.forward(pose, dict(zip(stream, values)))

    def variable(subject: Identity, pose: torch.Tensor, *values: torch.Tensor) -> HumanState:
        return subject.forward(pose, dict(zip(stream, values)))

    if identity is not None:
        return TermFn(fn=fixed, reads=(q, *stream))
    return TermFn(fn=variable, reads=(identity_name, q, *stream))


def joints(*, target: str, state: str = "state", joint_ids: Sequence[int] | None = None) -> TermFn:
    """3D joint error against data ``target (..., J, 3)`` (or the ``joint_ids`` subset). Residual ``(..., J * 3)``."""
    ids = None if joint_ids is None else torch.as_tensor(joint_ids, dtype=torch.long)

    def residual(human: HumanState, goal: torch.Tensor) -> torch.Tensor:
        positions = human.joints if ids is None else human.joints[..., ids.to(goal.device), :]
        return (positions - goal).flatten(-2)

    return TermFn(fn=residual, reads=(state, target))


def points(vertex_ids: Sequence[int] | torch.Tensor, *, target: str, state: str = "state") -> TermFn:
    """Marker fit: posed vertices ``vertex_ids (M,)`` against data ``target (..., M, 3)``. Residual ``(..., M * 3)``."""
    ids = torch.as_tensor(vertex_ids, dtype=torch.long)

    def residual(human: HumanState, goal: torch.Tensor) -> torch.Tensor:
        return (human.points(ids) - goal).flatten(-2)

    return TermFn(fn=residual, reads=(state, target))


def landmarks(*, target: str, name: str = "keypoints", state: str = "state") -> TermFn:
    """Landmark set ``name`` of the family against data ``target (..., L, 3)``. Residual ``(..., L * 3)``."""

    def residual(human: HumanState, goal: torch.Tensor) -> torch.Tensor:
        return (human.landmarks(name) - goal).flatten(-2)

    return TermFn(fn=residual, reads=(state, target))


def project(points: torch.Tensor, camera: torch.Tensor) -> torch.Tensor:
    """Pinhole projection of ``points (..., M, 3)`` with ``camera (..., 3, 4)`` (``K [R | t]``). Returns ``(..., M, 2)``.

    The camera's batch axes align with the *leading* batch axes of ``points``: a per-sequence
    ``camera (B, 3, 4)`` applies to ``points (B, T, M, 3)`` (it is not right-aligned broadcasting).
    """
    missing = (points.ndim - 2) - (camera.ndim - 2)
    if missing < 0 or camera.shape[:-2] != points.shape[: camera.ndim - 2]:
        raise ValueError(
            f"camera batch {tuple(camera.shape[:-2])} must lead the points batch {tuple(points.shape[:-2])}"
        )
    camera = camera.reshape(*camera.shape[:-2], *([1] * missing), 3, 4)
    homogeneous = torch.cat((points, torch.ones_like(points[..., :1])), dim=-1)
    pixels = homogeneous @ camera.mT
    return pixels[..., :2] / pixels[..., 2:3]


def reprojection(
    *,
    target: str,
    camera: str = "camera",
    vertex_ids: Sequence[int] | torch.Tensor | None = None,
    landmark_set: str | None = "keypoints",
    confidence: str | None = None,
    state: str = "state",
) -> TermFn:
    """2D reprojection error of landmarks (or ``vertex_ids``, or all joints) against ``target (..., M, 2)``.

    ``camera (..., 3, 4)`` is the world-to-pixel projection; ``confidence (..., M)`` scales rows.
    Residual ``(..., M * 2)``.
    """
    ids = None if vertex_ids is None else torch.as_tensor(vertex_ids, dtype=torch.long)

    def positions(human: HumanState) -> torch.Tensor:
        if ids is not None:
            return human.points(ids)
        return human.landmarks(landmark_set) if landmark_set is not None else human.joints

    def residual(
        human: HumanState, goal: torch.Tensor, projection: torch.Tensor, *weight: torch.Tensor
    ) -> torch.Tensor:
        error = project(positions(human), projection) - goal
        if weight:
            error = error * weight[0][..., None]
        return error.flatten(-2)

    reads = (state, target, camera) + ((confidence,) if confidence else ())
    return TermFn(fn=residual, reads=reads)


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
        positions = human.vertices() if ids is None else human.points(ids)
        clearance = positions @ normal.to(positions) - height
        return clearance.clamp_max(0.0) if hinge else clearance

    return TermFn(fn=residual, reads=(state,))


def contact(vertex_ids: Sequence[int] | torch.Tensor, *, target: str, state: str = "state") -> TermFn:
    """Stationary contact: vertices ``vertex_ids (M,)`` stay at data ``target (..., M, 3)`` (same as :func:`points`)."""
    return points(vertex_ids, target=target, state=state)


def pose_prior(identity: Identity, *, q: str = "q", mean: torch.Tensor | None = None, root: bool = False) -> TermFn:
    """Tangent distance of the joint rotations to ``mean`` (default: the neutral pose). Residual ``(..., nv - 6)``.

    A Gaussian prior with unit precision; scale it with the term weight or pre-whiten ``q``.
    ``root=True`` includes the six free-flyer coordinates.
    """
    model = identity.model
    reference = model.q_neutral if mean is None else mean
    start = 0 if root else 6

    def residual(pose: torch.Tensor) -> torch.Tensor:
        return model.difference(reference.expand_as(pose), pose)[..., start:]

    return TermFn(fn=residual, reads=(q,))


def shape_prior(group: str = "betas") -> TermFn:
    """L2 prior on an identity group variable: the residual is the coefficient vector itself."""
    return TermFn(fn=lambda value: value, reads=(group,))


def contact_wrenches(
    model: Model, kinematics: KinematicState, frame_ids: torch.Tensor, forces: torch.Tensor
) -> torch.Tensor:
    """``fext (..., njoints, 6)`` in joint frames from world ``forces (..., C, 3)`` applied at model frames ``frame_ids (C,)``."""
    parents = torch.as_tensor(
        [model.structure.frame_parent_joint_ids[int(f)] for f in frame_ids], dtype=torch.long, device=forces.device
    )
    joint = kinematics.joint_transform[..., parents, :, :]  # (..., C, 3, 4)
    forces = forces.expand(*joint.shape[:-2], 3)
    lever = kinematics.frame_translation[..., frame_ids, :] - joint[..., 3]
    rotation = joint[..., :3]
    linear = (rotation.mT @ forces[..., None]).squeeze(-1)
    angular = (rotation.mT @ torch.linalg.cross(lever, forces)[..., None]).squeeze(-1)
    fext = forces.new_zeros((*forces.shape[:-2], model.njoints, 6))
    return fext.index_add(-2, parents, torch.cat((linear, angular), dim=-1))


def root_wrench(
    identity: Identity, contacts: Sequence[str], *, dt: float, q: str = "q", forces: str = "forces"
) -> TermFn:
    """Root rows of inverse dynamics along a trajectory with contact forces: the physics residual of a motion.

    Reads ``q (B, T, nq)`` and ``forces (B, T, C, 3)`` (world forces at the model frames named in
    ``contacts``). Velocities and accelerations are central differences; the residual
    ``(B, T - 2, 6)`` is the free-flyer torque ``τ_root = (M a + b - Jᵀ f)[:6]`` at the interior
    knots, zero for a physically consistent motion. Time-local (``window=3``). Stream groups are
    ignored: inertias depend on identity only.
    """
    model = identity.model
    frame_ids = torch.as_tensor([model.frame_id(name) for name in contacts], dtype=torch.long)

    def residual(pose: torch.Tensor, force: torch.Tensor) -> torch.Tensor:
        if pose.ndim != 3 or pose.shape[1] < 3:
            raise ValueError(f"root_wrench needs q (B, T >= 3, nq); got {tuple(pose.shape)}")
        if force.shape[:2] != pose.shape[:2] or force.shape[-2:] != (len(contacts), 3):
            raise ValueError(f"forces must be (B, T, {len(contacts)}, 3) matching q; got {tuple(force.shape)}")
        velocity = model.difference(pose[:, :-1], pose[:, 1:]) / dt  # (B, T-1, nv) at half steps
        acceleration = (velocity[:, 1:] - velocity[:, :-1]) / dt  # (B, T-2, nv)
        mid_velocity = 0.5 * (velocity[:, 1:] + velocity[:, :-1])
        kinematics = forward_kinematics_state(model, pose[:, 1:-1])
        fext = contact_wrenches(model, kinematics, frame_ids.to(force.device), force[:, 1:-1])
        return inverse_dynamics(model, kinematics, mid_velocity, acceleration, fext=fext)[..., :6]

    return TermFn(fn=residual, reads=(q, forces), window=3)


__all__ = [
    "contact",
    "contact_wrenches",
    "ground",
    "human_state",
    "joints",
    "landmarks",
    "points",
    "pose_prior",
    "project",
    "reprojection",
    "root_wrench",
    "shape_prior",
    "shaped",
]
