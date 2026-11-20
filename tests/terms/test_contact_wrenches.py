"""``terms.contact_wrenches``: a world force at a frame enters inverse dynamics as minus its frame Jacobian's transpose."""

from __future__ import annotations

import torch
from better_robot import forward_kinematics, frame_jacobian, inverse_dynamics
from better_robot.kinematics import integrate

import better_human as bh
from better_human import terms
from tests.support.random_archives import random_weight_spec


def test_a_frame_force_is_the_transpose_of_its_jacobian() -> None:
    """``inverse_dynamics(external_forces=contact_wrenches(f))`` equals ``inverse_dynamics() - J_linearᵀ f`` per frame;
    two frames on one joint add up, and a frame off its joint's origin carries the force's moment."""
    frames = {"heel": (3400, "left_ankle"), "toe": (3300, "left_ankle"), "hand": (5000, "left_wrist")}
    human = bh.HumanModel.from_spec(random_weight_spec("smpl", dtype=torch.float64), contact_frames=frames)
    model = human.bake().model
    generator = torch.Generator().manual_seed(0)
    q = integrate(model, model.q_neutral, 0.3 * torch.randn(2, model.nv, dtype=torch.float64, generator=generator))
    forces = torch.randn(2, len(frames), 3, dtype=torch.float64, generator=generator)
    state = forward_kinematics(model, q)
    zero = torch.zeros(2, model.nv, dtype=torch.float64)

    free = inverse_dynamics(model, state, v=zero, a=zero)
    wrenches = terms.contact_wrenches(model, state, forces=forces, frames=list(frames))
    pushed = inverse_dynamics(model, state, v=zero, a=zero, external_forces=wrenches)
    expected = free.clone()
    for index, name in enumerate(frames):
        linear = frame_jacobian(model, state, frame=name)[:, :3]  # (2, 3, nv), world axes at the frame's origin
        expected -= (linear.mT @ forces[:, index, :, None]).squeeze(-1)
    torch.testing.assert_close(pushed, expected, rtol=0.0, atol=1e-9)
