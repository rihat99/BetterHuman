"""The physics of a motion: the free-flyer rows of inverse dynamics with contact forces."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from better_robot import forward_kinematics, inverse_dynamics
from better_robot.dynamics import contact_wrenches
from better_robot.exceptions import ShapeError
from better_robot.kinematics import difference
from better_robot.optim import TermFn

from better_human.core import Identity


def root_wrench(
    identity: Identity, contacts: Sequence[str], *, dt: float, q: str = "q", forces: str = "forces"
) -> TermFn:
    """Root rows of inverse dynamics along a trajectory with contact forces: the physics residual of a motion.

    Reads ``q (B, T, nq)`` and ``forces (B, T, C, 3)`` (world forces at the model frames named in
    ``contacts``). Velocities and accelerations are central differences; the residual
    ``(B, T - 2, 6)`` is the free-flyer torque ``τ_root = (M a + b - Jᵀ f)[:6]`` at the interior
    knots, zero for a physically consistent motion. Time-local (``window=3``). Stream groups are
    ignored: inertias depend on identity only. Raises :class:`~better_robot.exceptions.ShapeError` on evaluation
    for ``q`` that is not a batched trajectory of three or more knots, or ``forces`` that do not match it.
    """
    model = identity.model
    contacts = tuple(contacts)
    for name in contacts:
        model.frame_id(name)  # an unknown frame raises here, not at the first evaluation

    def residual(pose: torch.Tensor, force: torch.Tensor) -> torch.Tensor:
        if pose.ndim != 3 or pose.shape[1] < 3:
            raise ShapeError(f"root_wrench needs q (B, T >= 3, nq); got {tuple(pose.shape)}")
        if force.shape[:2] != pose.shape[:2] or force.shape[-2:] != (len(contacts), 3):
            raise ShapeError(f"forces must be (B, T, {len(contacts)}, 3) matching q; got {tuple(force.shape)}")
        velocity = difference(model, pose[:, :-1], pose[:, 1:]) / dt  # (B, T-1, nv) at half steps
        acceleration = (velocity[:, 1:] - velocity[:, :-1]) / dt  # (B, T-2, nv)
        mid_velocity = 0.5 * (velocity[:, 1:] + velocity[:, :-1])
        kinematics = forward_kinematics(model, pose[:, 1:-1])
        wrenches = contact_wrenches(model, kinematics, forces=force[:, 1:-1], frames=contacts)
        return inverse_dynamics(model, kinematics, v=mid_velocity, a=acceleration, external_forces=wrenches)[..., :6]

    return TermFn(fn=residual, reads=(q, forces), window=3)
