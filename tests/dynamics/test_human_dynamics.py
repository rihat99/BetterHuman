"""BetterRobot dynamics on shape-dependent SMPL inertias."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import better_robot as br
from better_robot.dynamics import bias_forces

import better_human as bh

MODEL_PATH = Path(__file__).parents[2] / "models" / "smpl" / "SMPL_NEUTRAL.npz"


@pytest.fixture(scope="module")
def mass_body() -> bh.SMPL:
    """Load the licensed neutral fixture with mass properties."""
    if not MODEL_PATH.is_file():
        pytest.skip(f"licensed SMPL fixture is absent: {MODEL_PATH}")
    return bh.SMPL(MODEL_PATH)


def test_gravity_rnea_zero_velocity_consistency_and_crba_spd(mass_body: bh.SMPL) -> None:
    q = mass_body.robot.q_neutral
    velocity = torch.zeros(mass_body.robot.nv, dtype=q.dtype)
    acceleration = torch.zeros_like(velocity)
    tau = br.rnea(mass_body.robot, q, velocity, acceleration)
    torch.testing.assert_close(tau, bias_forces(mass_body.robot, q, velocity))
    assert torch.isfinite(tau).all()

    mass_matrix = br.crba(mass_body.robot, q)
    torch.testing.assert_close(mass_matrix, mass_matrix.mT, rtol=2e-5, atol=2e-5)
    assert torch.linalg.eigvalsh(mass_matrix).min() > 0


def test_batched_shapes_drive_batched_dynamics(mass_body: bh.SMPL) -> None:
    betas = torch.stack((torch.zeros(10), torch.linspace(-0.5, 0.5, 10)))
    shaped = mass_body.with_shape(betas=betas)
    q = mass_body.robot.q_neutral
    velocity = torch.zeros(mass_body.robot.nv, dtype=q.dtype)
    acceleration = torch.zeros_like(velocity)
    tau = br.rnea(shaped.robot, q, velocity, acceleration)
    mass_matrix = br.crba(shaped.robot, q)
    assert tau.shape == (2, mass_body.robot.nv)
    assert mass_matrix.shape == (2, mass_body.robot.nv, mass_body.robot.nv)
    assert torch.isfinite(tau).all()
    assert torch.linalg.eigvalsh(mass_matrix).min() > 0


def test_betas_to_rnea_backward_is_finite_and_nonzero(mass_body: bh.SMPL) -> None:
    betas = (torch.randn(10, dtype=torch.float32) * 0.03).requires_grad_()
    shaped = mass_body.with_shape(betas=betas)
    q = shaped.robot.q_neutral
    velocity = torch.zeros(shaped.robot.nv, dtype=q.dtype)
    tau = br.rnea(shaped.robot, q, velocity, torch.zeros_like(velocity))
    tau.square().sum().backward()
    assert betas.grad is not None
    assert torch.isfinite(betas.grad).all()
    assert torch.count_nonzero(betas.grad) > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_mass_rnea_and_crba_smoke() -> None:
    if not MODEL_PATH.is_file():
        pytest.skip(f"licensed SMPL fixture is absent: {MODEL_PATH}")
    body = bh.SMPL(MODEL_PATH, device=torch.device("cuda"))
    q = body.robot.q_neutral
    velocity = torch.zeros(body.robot.nv, dtype=q.dtype, device=q.device)
    tau = br.rnea(body.robot, q, velocity, torch.zeros_like(velocity))
    mass_matrix = br.crba(body.robot, q)
    assert tau.is_cuda and mass_matrix.is_cuda
    assert torch.isfinite(tau).all()
    assert torch.linalg.eigvalsh(mass_matrix).min() > 0
