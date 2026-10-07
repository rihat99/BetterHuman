"""BetterRobot dynamics on shape-dependent SMPL inertias."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from better_robot.dynamics import inverse_dynamics, mass_matrix
from better_robot.kinematics import forward_kinematics_state

import better_human as bh

MODEL_PATH = Path(__file__).parents[2] / "models" / "smpl" / "SMPL_NEUTRAL.npz"


@pytest.fixture(scope="module")
def mass_body() -> bh.Body:
    """Load the licensed neutral fixture with mass properties."""
    if not MODEL_PATH.is_file():
        pytest.skip(f"licensed SMPL fixture is absent: {MODEL_PATH}")
    return bh.Body(bh.families.smpl(MODEL_PATH))


def test_gravity_inverse_dynamics_zero_velocity_and_mass_matrix_spd(mass_body: bh.Body) -> None:
    model = mass_body.bake().model
    q = model.q_neutral
    state = forward_kinematics_state(model, q)
    velocity = torch.zeros(model.nv, dtype=q.dtype)
    acceleration = torch.zeros_like(velocity)
    tau = inverse_dynamics(model, state, v=velocity, a=acceleration)
    assert torch.isfinite(tau).all()

    M = mass_matrix(model, state)
    torch.testing.assert_close(M, M.mT, rtol=2e-5, atol=2e-5)
    assert torch.linalg.eigvalsh(M).min() > 0


def test_batched_shapes_drive_batched_dynamics(mass_body: bh.Body) -> None:
    betas = torch.stack((torch.zeros(10), torch.linspace(-0.5, 0.5, 10)))
    model = mass_body.bake(betas=betas).model
    q = model.q_neutral
    velocity = torch.zeros(model.nv, dtype=q.dtype)
    acceleration = torch.zeros_like(velocity)
    state = forward_kinematics_state(model, q)
    tau = inverse_dynamics(model, state, v=velocity, a=acceleration)
    M = mass_matrix(model, state)
    assert tau.shape == (2, model.nv)
    assert M.shape == (2, model.nv, model.nv)
    assert torch.isfinite(tau).all()
    assert torch.linalg.eigvalsh(M).min() > 0


def test_betas_to_inverse_dynamics_backward_is_finite_and_nonzero(mass_body: bh.Body) -> None:
    betas = (torch.randn(10, dtype=torch.float32) * 0.03).requires_grad_()
    identity = mass_body.bake(betas=betas)
    q = identity.q_neutral
    velocity = torch.zeros(identity.nv, dtype=q.dtype)
    state = identity.forward(q)
    tau = inverse_dynamics(identity.model, state.kinematics, v=velocity, a=torch.zeros_like(velocity))
    tau.square().sum().backward()
    assert betas.grad is not None
    assert torch.isfinite(betas.grad).all()
    assert torch.count_nonzero(betas.grad) > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_mass_inverse_dynamics_and_mass_matrix_smoke() -> None:
    if not MODEL_PATH.is_file():
        pytest.skip(f"licensed SMPL fixture is absent: {MODEL_PATH}")
    model = bh.Body(bh.families.smpl(MODEL_PATH), device="cuda").bake().model
    q = model.q_neutral
    velocity = torch.zeros(model.nv, dtype=q.dtype, device=q.device)
    state = forward_kinematics_state(model, q)
    tau = inverse_dynamics(model, state, v=velocity, a=torch.zeros_like(velocity))
    M = mass_matrix(model, state)
    assert tau.is_cuda and M.is_cuda
    assert torch.isfinite(tau).all()
    assert torch.linalg.eigvalsh(M).min() > 0
