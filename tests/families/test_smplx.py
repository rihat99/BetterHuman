"""SMPL-X loader on the new core: reduced bodies, the hand mean and expression streams.

Classic round trips (hand PCA, reduced bodies) run through the shared checker, tests/families/test_classic_formats.py.
"""

from __future__ import annotations

import pytest
import torch

import better_human as bh
from better_human.families import SMPLXClassic
from better_human.io import weight_path

pytestmark = pytest.mark.licensed
F64 = torch.float64


def _body(**options: bool) -> bh.HumanModel:
    return bh.HumanModel.from_spec(
        bh.families.smplx(weight_path("smplx", "SMPLX_NEUTRAL.npz"), dtype=F64, **options), density=None
    )


@pytest.mark.parametrize(("hands", "face", "count"), [(True, True, 55), (False, True, 25), (True, False, 52)])
def test_reduced_bodies_drop_joints(hands: bool, face: bool, count: int) -> None:
    body = _body(hands=hands, face=face)
    assert len(body.spec.joint_names) == count
    if hands and face:
        return
    pose = {"jaw_pose": torch.ones(3, dtype=F64)} if not face else {"left_hand_pose": torch.ones(45, dtype=F64)}
    with pytest.raises(ValueError):
        bh.from_classic(body, SMPLXClassic(**pose))


def test_repeated_drop_joints_keeps_file_joint_ids() -> None:
    spec = _body(face=False).spec
    twice = bh.families.drop_joints(spec, [spec.joint_names[-1]])
    torch.testing.assert_close(twice.file_joint_ids, spec.file_joint_ids[:-1], rtol=0, atol=0)


def test_zero_hand_pose_uses_mean_unless_flat_hand_mean() -> None:
    _, q_mean, _ = bh.from_classic(_body(), SMPLXClassic())
    flat, q_flat, _ = bh.from_classic(_body(flat_hand_mean=True), SMPLXClassic())
    assert not torch.allclose(q_mean, q_flat)
    torch.testing.assert_close(q_flat[7:], flat.model.q_neutral[7:], atol=1e-12, rtol=0)


def test_expression_width_is_validated() -> None:
    body = _body()
    identity = body.bake()
    with pytest.raises(ValueError):
        identity.forward(identity.model.q_neutral, {"expression": torch.zeros(11, dtype=F64)})
    with pytest.raises(ValueError):
        bh.families.smplx(weight_path("smplx", "SMPLX_NEUTRAL.npz"), num_expression=1000)
