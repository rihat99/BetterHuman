"""FLAME loader on the new core, against a synthetic FLAME-shaped file (no FLAME asset ships)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

import better_human as bh
from better_human.families.classic import FLAMEClassic, flame_from_classic, flame_to_classic

NUM_VERTICES = 50
NUM_JOINTS = 5


@pytest.fixture
def flame_path(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    weights = rng.random((NUM_VERTICES, NUM_JOINTS))
    regressor = rng.random((NUM_JOINTS, NUM_VERTICES))
    path = tmp_path / "generic_model.npz"
    np.savez(
        path,
        v_template=rng.normal(size=(NUM_VERTICES, 3)) * 0.1,
        shapedirs=rng.normal(size=(NUM_VERTICES, 3, 400)) * 1e-3,
        posedirs=rng.normal(size=(NUM_VERTICES, 3, 36)) * 1e-3,
        J_regressor=regressor / regressor.sum(axis=1, keepdims=True),
        weights=weights / weights.sum(axis=1, keepdims=True),
        kintree_table=np.array([[4294967295, 0, 1, 1, 1], [0, 1, 2, 3, 4]], dtype=np.uint32),
        f=rng.integers(0, NUM_VERTICES, size=(80, 3)).astype(np.uint32),
    )
    return path


def _body(path: Path, **options: int) -> bh.Body:
    return bh.Body(bh.families.flame(path, dtype=torch.float64, **options), mass=False)


def test_load(flame_path: Path) -> None:
    body = _body(flame_path)
    assert body.joint_names == ("global", "neck", "jaw", "left_eye", "right_eye")
    assert body.num_vertices == NUM_VERTICES
    assert body.identity_groups == {"betas": 100}
    assert body.stream_groups == {"expression": 50}


def test_widths_slice_and_validate(flame_path: Path) -> None:
    body = _body(flame_path, num_betas=7, num_expression=3)
    assert body.identity_groups == {"betas": 7}
    assert body.stream_groups == {"expression": 3}
    with pytest.raises(ValueError):
        bh.families.flame(flame_path, num_betas=301)
    with pytest.raises(ValueError):
        bh.families.flame(flame_path, num_expression=101)


def test_expression_stream_moves_vertices_and_joints(flame_path: Path) -> None:
    identity = _body(flame_path).bake()
    q, stream = flame_from_classic(identity, FLAMEClassic(expression=torch.full((50,), 2.0, dtype=torch.float64)))
    torch.testing.assert_close(q[3:], identity.q_neutral[3:], atol=0, rtol=0)
    neutral = identity.forward(q)
    expressive = identity.forward(q, stream)
    assert (expressive.vertices - neutral.vertices).abs().max() > 1e-4
    assert (expressive.joints - neutral.joints).abs().max() > 1e-6


def test_classic_round_trip(flame_path: Path) -> None:
    identity = _body(flame_path).bake(betas=torch.randn(100, dtype=torch.float64))
    generator = torch.Generator().manual_seed(0)
    names = ("global_orient", "neck_pose", "jaw_pose", "leye_pose", "reye_pose", "transl")
    values = {name: 0.3 * torch.randn(2, 3, generator=generator, dtype=torch.float64) for name in names}
    classic = FLAMEClassic(expression=torch.randn(2, 50, generator=generator, dtype=torch.float64), **values)
    q, stream = flame_from_classic(identity, classic)
    back = flame_to_classic(identity, q, stream)
    for name in (*names, "expression"):
        torch.testing.assert_close(getattr(back, name), getattr(classic, name), atol=1e-10, rtol=0)
