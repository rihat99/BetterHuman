"""Human cost terms on optim: residual shapes, camera alignment, physics residual, input validation.

They run on the random-weight SMPL archive with Dempster part inertias (no weights needed).
"""

from __future__ import annotations

import pytest
import torch

import better_human as bh
from better_robot.exceptions import ShapeError, UnknownNameError
from better_robot.kinematics import integrate

from better_human import terms
from tests.support.random_archives import random_weight_spec


@pytest.fixture(scope="module")
def identity() -> bh.Identity:
    return bh.HumanModel.from_spec(random_weight_spec("smpl", dtype=torch.float64)).bake()


def _trajectory(identity: bh.Identity, batch: int, length: int) -> torch.Tensor:
    q = identity.model.q_neutral.expand(batch, length, identity.model.nq).clone()
    generator = torch.Generator().manual_seed(0)
    step = 0.1 * torch.randn(batch, length, identity.model.nv, dtype=q.dtype, generator=generator)
    q[..., 7:] = integrate(identity.model, q, step)[..., 7:]
    return q


def test_terms_keep_leading_batch_axes(identity: bh.Identity) -> None:
    q = _trajectory(identity, 2, 3)
    state = identity.forward(q)
    assert terms.joints(target="t").fn(state, state.joint_translations).shape == (2, 3, 24 * 3)
    assert terms.points([0, 5], target="t").fn(state, state.points(torch.tensor([0, 5]))).shape == (2, 3, 6)
    assert terms.ground([0, 1, 2]).fn(state).shape == (2, 3, 3)
    assert terms.pose_prior(identity).fn(q).shape == (2, 3, identity.model.nv - 6)


def test_reprojection_aligns_camera_with_leading_batch_axes(identity: bh.Identity) -> None:
    q = _trajectory(identity, 3, 3)  # B == T so a right-aligned broadcast would silently pass
    state = identity.forward(q)
    cameras = torch.eye(3, 4, dtype=q.dtype).expand(3, 3, 4).clone()
    cameras[:, 2, 3] = torch.tensor([3.0, 5.0, 7.0], dtype=q.dtype)
    term = terms.reprojection(target="pixels", joints=True)
    origin = torch.zeros(3, 3, 24, 2, dtype=q.dtype)  # the residual is then the projected pixels
    pixels = term.fn(state, origin, cameras)
    per_sequence = torch.stack([term.fn(identity.forward(q[b]), origin[b], cameras[b]) for b in range(3)])
    torch.testing.assert_close(pixels, per_sequence)
    with pytest.raises(ShapeError, match="camera batch"):
        term.fn(state, origin, cameras[:2])


def test_joints_by_name_and_unknown_names_raise(identity: bh.Identity) -> None:
    state = identity.forward(_trajectory(identity, 2, 1))
    wrist = identity.human.spec.joint_id("left_wrist")
    goal = state.joint_translations[..., [wrist], :]
    assert terms.joints(target="t", joint_names=["left_wrist"]).fn(state, goal).abs().max() == 0
    with pytest.raises(UnknownNameError, match="unknown joint 'left_wirst'"):
        terms.joints(target="t", joint_names=["left_wirst"]).fn(state, goal)
    with pytest.raises(UnknownNameError, match="unknown stream group 'expresion'"):
        terms.human_state(identity, streams=["expresion"]).fn(state.q, torch.zeros(10, dtype=state.q.dtype))


def test_root_wrench_vanishes_in_free_fall(identity: bh.Identity) -> None:
    dt, length = 1e-3, 6
    q = identity.model.q_neutral.expand(1, length, identity.model.nq).clone()
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
    with pytest.raises(ShapeError, match="root_wrench needs"):
        term.fn(identity.model.q_neutral.expand(4, identity.model.nq), torch.zeros(4, 0, 3))
    with pytest.raises(ShapeError, match="root_wrench needs"):
        term.fn(identity.model.q_neutral.expand(1, 2, identity.model.nq), torch.zeros(1, 2, 0, 3))
    with pytest.raises(ShapeError, match="forces must be"):
        term.fn(identity.model.q_neutral.expand(1, 3, identity.model.nq), torch.zeros(1, 3, 1, 3))
