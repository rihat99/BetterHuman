"""SMPL+H loader on the new core. Classic round trips run through tests/families/test_classic_formats.py."""

from __future__ import annotations

import pytest
import torch

import better_human as bh
from better_human.families import SMPLHClassic
from better_human.io import weight_path

pytestmark = pytest.mark.licensed


@pytest.fixture(scope="module")
def body() -> bh.HumanModel:
    return bh.HumanModel.from_spec(
        bh.families.smplh(weight_path("smplh/smplh/neutral", "model.npz"), dtype=torch.float64), density=None
    )


def test_joint_counts(body: bh.HumanModel) -> None:
    assert len(body.spec.joint_names) == 52
    reduced = bh.HumanModel.from_spec(
        bh.families.smplh(weight_path("smplh/smplh/neutral", "model.npz"), hands=False, dtype=torch.float64),
        density=None,
    )
    assert len(reduced.spec.joint_names) == 22


def test_posed_hand_moves_finger_vertices(body: bh.HumanModel) -> None:
    identity, q, stream = bh.from_classic(body, SMPLHClassic())
    neutral = identity.forward(q, stream)
    left_hand_pose = torch.full((45,), 0.5, dtype=torch.float64)
    identity, q, stream = bh.from_classic(body, SMPLHClassic(left_hand_pose=left_hand_pose))
    posed = identity.forward(q, stream)
    displacement = (posed.vertices - neutral.vertices).norm(dim=-1)
    assert displacement.max() > 1e-2
    assert (displacement > 1e-3).sum() < body.spec.num_vertices // 4
    torch.testing.assert_close(posed.joint_translations[:22], neutral.joint_translations[:22], atol=1e-12, rtol=0)
