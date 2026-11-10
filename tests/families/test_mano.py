"""MANO loader on the new core. Classic round trips run through tests/families/test_classic_formats.py."""

from __future__ import annotations

import pytest
import torch

import better_human as bh
from better_human.families import MANOClassic
from better_human.io import weight_path

pytestmark = pytest.mark.licensed


def _body(flat_hand_mean: bool = False) -> bh.HumanModel:
    return bh.HumanModel.from_spec(
        bh.families.mano(weight_path("mano", "MANO_RIGHT.npz"), flat_hand_mean=flat_hand_mean, dtype=torch.float64),
        density=None,
    )


def test_joint_count() -> None:
    assert len(_body().spec.joint_names) == 16


def test_flat_hand_mean() -> None:
    mean_body, flat_body = _body(), _body(flat_hand_mean=True)
    flat_identity, flat_q, _ = bh.from_classic(flat_body, MANOClassic())
    torch.testing.assert_close(flat_q[3:], flat_identity.model.q_neutral[3:], atol=1e-12, rtol=0)
    torch.testing.assert_close(flat_q[:3], flat_identity.root_offset, atol=1e-12, rtol=0)
    mean_identity, mean_q, _ = bh.from_classic(mean_body, MANOClassic())
    assert "hand_pose" in mean_body.spec.classic.hand_means and not flat_body.spec.classic.hand_means
    full = bh.to_classic(mean_identity, mean_q)
    torch.testing.assert_close(full.hand_pose, torch.zeros(45, dtype=torch.float64), atol=1e-10, rtol=0)
    assert not torch.allclose(mean_identity.forward(mean_q).vertices, flat_identity.forward(flat_q).vertices)
