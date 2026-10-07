"""SMPL loader on the new core: identity baking, broadcasting, gathers and inertias."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import better_human as bh

MODEL_PATH = Path(__file__).parents[2] / "models" / "smpl" / "SMPL_NEUTRAL.npz"


@pytest.fixture(scope="module")
def body() -> bh.Body:
    if not MODEL_PATH.is_file():
        pytest.skip(f"licensed SMPL fixture is absent: {MODEL_PATH}")
    return bh.Body(bh.families.smpl(MODEL_PATH, dtype=torch.float64))


def test_load_without_mass(body: bh.Body) -> None:
    plain = bh.load("smpl", str(MODEL_PATH), num_betas=10, mass=False)
    assert len(plain.joint_names) == 24 and plain.topology is None


def test_bake_and_forward_validate_inputs(body: bh.Body) -> None:
    with pytest.raises(ValueError):
        body.bake(betas=torch.zeros(11, dtype=torch.float64))
    with pytest.raises(ValueError):
        body.bake(expression=torch.zeros(10, dtype=torch.float64))
    with pytest.raises(ValueError):
        body.bake(betas=torch.zeros(10, dtype=torch.float32))
    with pytest.raises(ValueError):
        body.bake().forward(torch.zeros(3, 7, dtype=torch.float64), {"expression": torch.zeros(10, dtype=torch.float64)})
    with pytest.raises(ValueError):
        bh.families.smpl(MODEL_PATH, num_betas=301)


def test_bake_broadcasts_leading_dims_and_pose_batches(body: bh.Body) -> None:
    generator = torch.Generator().manual_seed(0)
    betas = torch.randn(3, 10, generator=generator, dtype=torch.float64)
    identity = body.bake(betas=betas)
    assert identity.v_shaped.shape == (3, body.num_vertices, 3)
    single = body.bake(betas=betas[1])
    torch.testing.assert_close(identity.v_shaped[1], single.v_shaped, atol=1e-12, rtol=0)
    q = identity.q_neutral.expand(2, 1, -1).clone()
    q[0, 0, 7:10] = 0.3
    state = identity.forward(q)
    assert state.vertices().shape == (2, 3, body.num_vertices, 3)
    torch.testing.assert_close(state.vertices()[0, 1], single.forward(q[0, 0]).vertices(), atol=1e-10, rtol=0)


def test_points_and_landmarks_gather_vertices(body: bh.Body) -> None:
    identity = body.bake(betas=torch.full((10,), 0.2, dtype=torch.float64))
    state = identity.forward(identity.q_neutral)
    ids = torch.tensor([0, 411, 3500, 6889])
    torch.testing.assert_close(state.points(ids), state.vertices()[..., ids, :], atol=1e-6, rtol=0)
    keypoints = torch.as_tensor(body.spec.landmarks["keypoints"], dtype=torch.long)
    torch.testing.assert_close(state.landmarks(), state.vertices()[..., keypoints, :], atol=1e-6, rtol=0)


def test_inertias_are_finite_with_plausible_total_mass(body: bh.Body) -> None:
    inertias = body.bake().model.body_inertias
    assert torch.isfinite(inertias).all()
    masses = inertias[body.part_body_ids, 0]
    assert (masses > 0).all()
    assert 60.0 < float(masses.sum()) < 90.0
