"""SMPL-X loader on the new core: reduced bodies, hand PCA and expression streams."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import better_human as bh
from better_human.families.classic import SMPLXClassic, smplx_from_classic, smplx_to_classic

MODEL_PATH = Path(__file__).parents[2] / "models" / "smplx" / "SMPLX_NEUTRAL.npz"
F64 = torch.float64


def _body(**options: bool) -> bh.Body:
    if not MODEL_PATH.is_file():
        pytest.skip(f"licensed SMPL-X fixture is absent: {MODEL_PATH}")
    return bh.Body(bh.families.smplx(MODEL_PATH, dtype=F64, **options), mass=False)


@pytest.mark.parametrize(("hands", "face", "count"), [(True, True, 55), (False, True, 25), (True, False, 52)])
def test_reduced_bodies_drop_joints(hands: bool, face: bool, count: int) -> None:
    body = _body(hands=hands, face=face)
    assert len(body.joint_names) == count
    if hands and face:
        return
    pose = {"jaw_pose": torch.ones(3, dtype=F64)} if not face else {"left_hand_pose": torch.ones(45, dtype=F64)}
    with pytest.raises(ValueError):
        smplx_from_classic(body.bake(), SMPLXClassic(**pose))


@pytest.mark.parametrize("flat_hand_mean", [False, True])
def test_pca_hand_round_trip(flat_hand_mean: bool) -> None:
    identity = _body(flat_hand_mean=flat_hand_mean).bake()
    generator = torch.Generator().manual_seed(0)
    left, right = (torch.randn(2, 6, generator=generator, dtype=F64) for _ in range(2))
    classic = SMPLXClassic(left_hand_pose=left, right_hand_pose=right, num_pca_comps=6)
    q, stream = smplx_from_classic(identity, classic)
    back = smplx_to_classic(identity, q, stream, num_pca_comps=6)
    torch.testing.assert_close(back.left_hand_pose, left, atol=1e-8, rtol=0)
    torch.testing.assert_close(back.right_hand_pose, right, atol=1e-8, rtol=0)


def test_zero_hand_pose_uses_mean_unless_flat_hand_mean() -> None:
    mean = _body().bake()
    flat = _body(flat_hand_mean=True).bake()
    q_mean, _ = smplx_from_classic(mean, SMPLXClassic())
    q_flat, _ = smplx_from_classic(flat, SMPLXClassic())
    assert not torch.allclose(q_mean, q_flat)
    torch.testing.assert_close(q_flat[7:], flat.q_neutral[7:], atol=1e-12, rtol=0)


def test_expression_width_is_validated() -> None:
    body = _body()
    identity = body.bake()
    with pytest.raises(ValueError):
        identity.forward(identity.q_neutral, {"expression": torch.zeros(11, dtype=F64)})
    with pytest.raises(ValueError):
        bh.families.smplx(MODEL_PATH, num_expression=1000)
