"""SMPL+H loader on the new core."""

from __future__ import annotations

import pytest
import torch

import better_human as bh
from better_human.families.classic import SMPLHClassic, smplh_from_classic, smplh_to_classic
from better_human.io import weight_path

pytestmark = pytest.mark.licensed


@pytest.fixture(scope="module")
def body() -> bh.HumanModel:
    return bh.HumanModel.from_spec(
        bh.families.smplh(weight_path("smplh", "smplh/neutral/model.npz"), dtype=torch.float64), density=None
    )


def test_joint_counts(body: bh.HumanModel) -> None:
    assert len(body.spec.joint_names) == 52
    reduced = bh.HumanModel.from_spec(
        bh.families.smplh(weight_path("smplh", "smplh/neutral/model.npz"), hands=False, dtype=torch.float64),
        density=None,
    )
    assert len(reduced.spec.joint_names) == 22


def test_classic_round_trip(body: bh.HumanModel) -> None:
    generator = torch.Generator().manual_seed(0)
    identity = body.bake(betas=torch.randn(16, generator=generator, dtype=torch.float64))
    classic = SMPLHClassic(
        global_orient=0.3 * torch.randn(2, 3, generator=generator, dtype=torch.float64),
        body_pose=0.3 * torch.randn(2, 63, generator=generator, dtype=torch.float64),
        left_hand_pose=0.3 * torch.randn(2, 45, generator=generator, dtype=torch.float64),
        right_hand_pose=0.3 * torch.randn(2, 45, generator=generator, dtype=torch.float64),
        transl=torch.randn(2, 3, generator=generator, dtype=torch.float64),
    )
    q, stream = smplh_from_classic(identity, classic)
    back = smplh_to_classic(identity, q, stream)
    for name in ("global_orient", "body_pose", "left_hand_pose", "right_hand_pose", "transl"):
        torch.testing.assert_close(getattr(back, name), getattr(classic, name), atol=1e-10, rtol=0)


def test_posed_hand_moves_finger_vertices(body: bh.HumanModel) -> None:
    identity = body.bake()
    neutral = identity.forward(*smplh_from_classic(identity, SMPLHClassic()))
    q, stream = smplh_from_classic(identity, SMPLHClassic(left_hand_pose=torch.full((45,), 0.5, dtype=torch.float64)))
    posed = identity.forward(q, stream)
    displacement = (posed.vertices - neutral.vertices).norm(dim=-1)
    assert displacement.max() > 1e-2
    assert (displacement > 1e-3).sum() < body.spec.num_vertices // 4
    torch.testing.assert_close(posed.joint_translations[:22], neutral.joint_translations[:22], atol=1e-12, rtol=0)
