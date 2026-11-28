"""BetterRobot dynamics on shape-dependent SMPL inertias."""

from __future__ import annotations

import pytest
import torch
from better_robot import forward_kinematics, inverse_dynamics, mass_matrix

import better_human as bh
from better_human.io import weight_path

pytestmark = pytest.mark.licensed


@pytest.fixture(scope="module")
def mass_body() -> bh.HumanModel:
    """Load the licensed neutral fixture with mass properties."""
    return bh.HumanModel.from_spec(bh.families.smpl(weight_path("smpl", "SMPL_NEUTRAL.npz")))


def test_batched_shapes_drive_batched_dynamics(mass_body: bh.HumanModel) -> None:
    """The zero identity and a shaped one: finite gravity torques at rest, a symmetric positive-definite mass matrix."""
    betas = torch.stack((torch.zeros(10), torch.linspace(-0.5, 0.5, 10)))
    model = mass_body.bake(betas=betas).model
    q = model.q_neutral
    velocity = torch.zeros(model.nv, dtype=q.dtype)
    acceleration = torch.zeros_like(velocity)
    state = forward_kinematics(model, q)
    tau = inverse_dynamics(model, state, v=velocity, a=acceleration)
    M = mass_matrix(model, state)
    assert tau.shape == (2, model.nv)
    assert M.shape == (2, model.nv, model.nv)
    assert torch.isfinite(tau).all()
    torch.testing.assert_close(M, M.mT, rtol=2e-5, atol=2e-5)
    assert torch.linalg.eigvalsh(M).min() > 0


def test_betas_to_inverse_dynamics_backward_is_finite_and_nonzero(mass_body: bh.HumanModel) -> None:
    betas = (torch.randn(10, dtype=torch.float32, generator=torch.Generator().manual_seed(0)) * 0.03).requires_grad_()
    identity = mass_body.bake(betas=betas)
    q = identity.model.q_neutral
    velocity = torch.zeros(identity.model.nv, dtype=q.dtype)
    state = identity.forward(q)
    tau = inverse_dynamics(identity.model, state.kinematics, v=velocity, a=torch.zeros_like(velocity))
    tau.square().sum().backward()
    assert betas.grad is not None
    assert torch.isfinite(betas.grad).all()
    assert torch.count_nonzero(betas.grad) > 0


@pytest.mark.cuda
def test_cuda_mass_inverse_dynamics_and_mass_matrix_smoke() -> None:
    model = (
        bh.HumanModel.from_spec(bh.families.smpl(weight_path("smpl", "SMPL_NEUTRAL.npz"), device="cuda")).bake().model
    )
    q = model.q_neutral
    velocity = torch.zeros(model.nv, dtype=q.dtype, device=q.device)
    state = forward_kinematics(model, q)
    tau = inverse_dynamics(model, state, v=velocity, a=torch.zeros_like(velocity))
    M = mass_matrix(model, state)
    assert tau.is_cuda and M.is_cuda
    assert torch.isfinite(tau).all()
    assert torch.linalg.eigvalsh(M).min() > 0
