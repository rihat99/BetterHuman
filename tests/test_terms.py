"""Human cost terms on optim2: residual shapes, camera alignment, physics residual, input validation."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import better_human as bh
from better_human import terms

MODEL_PATH = Path(__file__).parents[1] / "models" / "smpl" / "SMPL_NEUTRAL.npz"


@pytest.fixture(scope="module")
def identity() -> bh.Identity:
    if not MODEL_PATH.is_file():
        pytest.skip(f"licensed SMPL fixture is absent: {MODEL_PATH}")
    body = bh.Body(bh.families.smpl(MODEL_PATH, dtype=torch.float64), contact_frames=None)
    return body.bake()


def _trajectory(identity: bh.Identity, batch: int, length: int) -> torch.Tensor:
    q = identity.q_neutral.expand(batch, length, identity.nq).clone()
    q[..., 7:] = identity.model.integrate(q, 0.1 * torch.randn(batch, length, identity.nv, dtype=q.dtype))[..., 7:]
    return q


def test_terms_keep_leading_batch_axes(identity: bh.Identity) -> None:
    q = _trajectory(identity, 2, 3)
    state = identity.forward(q)
    assert terms.joints(target="t").fn(state, state.joints).shape == (2, 3, 24 * 3)
    assert terms.points([0, 5], target="t").fn(state, state.points(torch.tensor([0, 5]))).shape == (2, 3, 6)
    assert terms.ground([0, 1, 2]).fn(state).shape == (2, 3, 3)
    assert terms.pose_prior(identity).fn(q).shape == (2, 3, identity.nv - 6)


def test_project_aligns_camera_with_leading_batch_axes(identity: bh.Identity) -> None:
    q = _trajectory(identity, 3, 3)  # B == T so a right-aligned broadcast would silently pass
    state = identity.forward(q)
    cameras = torch.eye(3, 4, dtype=q.dtype).expand(3, 3, 4).clone()
    cameras[:, 2, 3] = torch.tensor([3.0, 5.0, 7.0], dtype=q.dtype)
    pixels = terms.project(state.joints, cameras)
    per_sequence = torch.stack([terms.project(state.joints[b], cameras[b]) for b in range(3)])
    torch.testing.assert_close(pixels, per_sequence)
    with pytest.raises(ValueError, match="camera batch"):
        terms.project(state.joints, cameras[:2])


def test_root_wrench_vanishes_in_free_fall(identity: bh.Identity) -> None:
    dt, length = 1e-3, 6
    q = identity.q_neutral.expand(1, length, identity.nq).clone()
    g = identity.model.gravity[:3]
    assert torch.allclose(g, torch.tensor([0.0, -9.81, 0.0], dtype=g.dtype)), "SMPL is y-up: gravity along -y"
    t = torch.arange(length, dtype=q.dtype) * dt
    q[0, :, :3] = 0.5 * g * t[:, None] ** 2
    term = terms.root_wrench(identity, [], dt=dt)
    residual = term.fn(q, torch.zeros(1, length, 0, 3, dtype=q.dtype))
    assert residual.shape == (1, length - 2, 6)
    assert residual.abs().max() < 1e-6


def test_root_wrench_rejects_unbatched_or_short_trajectories(identity: bh.Identity) -> None:
    term = terms.root_wrench(identity, [], dt=1e-2)
    with pytest.raises(ValueError, match="root_wrench needs"):
        term.fn(identity.q_neutral.expand(4, identity.nq), torch.zeros(4, 0, 3))
    with pytest.raises(ValueError, match="root_wrench needs"):
        term.fn(identity.q_neutral.expand(1, 2, identity.nq), torch.zeros(1, 2, 0, 3))
    with pytest.raises(ValueError, match="forces must be"):
        term.fn(identity.q_neutral.expand(1, 3, identity.nq), torch.zeros(1, 3, 1, 3))


def test_state_memo_does_not_leak_no_grad_results(identity: bh.Identity) -> None:
    q = identity.q_neutral.clone().requires_grad_(True)
    state = identity.forward(q)
    with torch.no_grad():
        assert not state.vertices().requires_grad
    assert state.vertices().requires_grad
