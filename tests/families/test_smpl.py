"""SMPL loader on the new core: identity baking, gathers and inertias."""

from __future__ import annotations

import pytest
import torch
from better_robot.exceptions import UnknownNameError

import better_human as bh
from better_human.io import weight_path

pytestmark = pytest.mark.licensed


@pytest.fixture(scope="module")
def body() -> bh.HumanModel:
    return bh.HumanModel.from_spec(bh.families.smpl(weight_path("smpl", "SMPL_NEUTRAL.npz"), dtype=torch.float64))


def test_load_without_mass(body: bh.HumanModel) -> None:
    plain = bh.HumanModel.from_spec(
        bh.families.smpl(str(weight_path("smpl", "SMPL_NEUTRAL.npz")), num_betas=10), density=None
    )
    assert len(plain.spec.joint_names) == 24 and plain.part_topology is None


def test_bake_and_forward_validate_inputs(body: bh.HumanModel) -> None:
    with pytest.raises(ValueError):
        body.bake(betas=torch.zeros(11, dtype=torch.float64))
    with pytest.raises(UnknownNameError, match="unknown identity group 'expression'"):
        body.bake(expression=torch.zeros(10, dtype=torch.float64))
    with pytest.raises(ValueError):
        body.bake(betas=torch.zeros(10, dtype=torch.float32))
    with pytest.raises(UnknownNameError, match="unknown stream group 'expression'"):
        body.bake().forward(
            torch.zeros(3, 7, dtype=torch.float64), {"expression": torch.zeros(10, dtype=torch.float64)}
        )
    with pytest.raises(ValueError):
        bh.families.smpl(weight_path("smpl", "SMPL_NEUTRAL.npz"), num_betas=301)


def test_points_and_landmarks_gather_vertices(body: bh.HumanModel) -> None:
    identity = body.bake(betas=torch.full((10,), 0.2, dtype=torch.float64))
    state = identity.forward(identity.model.q_neutral)
    ids = torch.tensor([0, 411, 3500, 6889])
    torch.testing.assert_close(state.points(ids), state.vertices[..., ids, :], atol=1e-6, rtol=0)
    keypoints = body.spec.landmarks.vertex_ids
    torch.testing.assert_close(state.landmarks(), state.vertices[..., keypoints, :], atol=1e-6, rtol=0)


def test_inertias_are_finite_with_plausible_total_mass(body: bh.HumanModel) -> None:
    inertias = body.bake().model.body_inertias
    assert torch.isfinite(inertias).all()
    masses = inertias[body.part_body_ids, 0]
    assert (masses > 0).all()
    assert 60.0 < float(masses.sum()) < 90.0
