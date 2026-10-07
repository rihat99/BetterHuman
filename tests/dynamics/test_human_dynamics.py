"""BetterRobot dynamics on shape-dependent SMPL inertias."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import better_robot as br
from better_robot.dynamics import bias_forces
from better_robot.dynamics.state import inverse_dynamics
from better_robot.kinematics import forward_kinematics_state

import better_human as bh

MODEL_PATH = Path(__file__).parents[2] / "models" / "smpl" / "SMPL_NEUTRAL.npz"


@pytest.fixture(scope="module")
def mass_body() -> bh.Body:
    """Load the licensed neutral fixture with mass properties."""
    if not MODEL_PATH.is_file():
        pytest.skip(f"licensed SMPL fixture is absent: {MODEL_PATH}")
    return bh.Body(bh.families.smpl(MODEL_PATH))


def test_gravity_rnea_zero_velocity_consistency_and_crba_spd(mass_body: bh.Body) -> None:
    model = mass_body.bake().model
    q = model.q_neutral
    velocity = torch.zeros(model.nv, dtype=q.dtype)
    acceleration = torch.zeros_like(velocity)
    tau = br.rnea(model, q, velocity, acceleration)
    torch.testing.assert_close(tau, bias_forces(model, q, velocity))
    torch.testing.assert_close(inverse_dynamics(model, forward_kinematics_state(model, q), velocity, acceleration), tau)
    assert torch.isfinite(tau).all()

    mass_matrix = br.crba(model, q)
    torch.testing.assert_close(mass_matrix, mass_matrix.mT, rtol=2e-5, atol=2e-5)
    assert torch.linalg.eigvalsh(mass_matrix).min() > 0


def test_batched_shapes_drive_batched_dynamics(mass_body: bh.Body) -> None:
    betas = torch.stack((torch.zeros(10), torch.linspace(-0.5, 0.5, 10)))
    model = mass_body.bake(betas=betas).model
    q = model.q_neutral
    velocity = torch.zeros(model.nv, dtype=q.dtype)
    acceleration = torch.zeros_like(velocity)
    tau = br.rnea(model, q, velocity, acceleration)
    mass_matrix = br.crba(model, q)
    assert tau.shape == (2, model.nv)
    assert mass_matrix.shape == (2, model.nv, model.nv)
    assert torch.isfinite(tau).all()
    assert torch.linalg.eigvalsh(mass_matrix).min() > 0


def test_betas_to_rnea_backward_is_finite_and_nonzero(mass_body: bh.Body) -> None:
    betas = (torch.randn(10, dtype=torch.float32) * 0.03).requires_grad_()
    identity = mass_body.bake(betas=betas)
    q = identity.q_neutral
    velocity = torch.zeros(identity.nv, dtype=q.dtype)
    state = identity.forward(q)
    tau = inverse_dynamics(identity.model, state.kinematics, velocity, torch.zeros_like(velocity))
    tau.square().sum().backward()
    assert betas.grad is not None
    assert torch.isfinite(betas.grad).all()
    assert torch.count_nonzero(betas.grad) > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_mass_rnea_and_crba_smoke() -> None:
    if not MODEL_PATH.is_file():
        pytest.skip(f"licensed SMPL fixture is absent: {MODEL_PATH}")
    model = bh.Body(bh.families.smpl(MODEL_PATH), device="cuda").bake().model
    q = model.q_neutral
    velocity = torch.zeros(model.nv, dtype=q.dtype, device=q.device)
    tau = br.rnea(model, q, velocity, torch.zeros_like(velocity))
    mass_matrix = br.crba(model, q)
    assert tau.is_cuda and mass_matrix.is_cuda
    assert torch.isfinite(tau).all()
    assert torch.linalg.eigvalsh(mass_matrix).min() > 0
