"""MANO loader on the new core."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import better_human as bh
from better_human.families.classic import MANOClassic, mano_from_classic, mano_to_classic

MODEL_PATH = Path(__file__).parents[2] / "models" / "mano" / "MANO_RIGHT.npz"


def _body(flat_hand_mean: bool = False) -> bh.Body:
    if not MODEL_PATH.is_file():
        pytest.skip(f"licensed MANO fixture is absent: {MODEL_PATH}")
    return bh.Body(bh.families.mano(MODEL_PATH, flat_hand_mean=flat_hand_mean, dtype=torch.float64), mass=False)


def test_joint_count() -> None:
    assert len(_body().joint_names) == 16


def test_pca_round_trip() -> None:
    body = _body()
    generator = torch.Generator().manual_seed(0)
    identity = body.bake(betas=torch.randn(10, generator=generator, dtype=torch.float64))
    classic = MANOClassic(
        global_orient=0.3 * torch.randn(3, 3, generator=generator, dtype=torch.float64),
        hand_pose=torch.randn(3, 6, generator=generator, dtype=torch.float64),
        transl=torch.randn(3, 3, generator=generator, dtype=torch.float64),
        num_pca_comps=6,
    )
    q, stream = mano_from_classic(identity, classic)
    back = mano_to_classic(identity, q, stream, num_pca_comps=6)
    for name in ("global_orient", "hand_pose", "transl"):
        torch.testing.assert_close(getattr(back, name), getattr(classic, name), atol=1e-9, rtol=0)


def test_flat_hand_mean() -> None:
    mean_body, flat_body = _body(), _body(flat_hand_mean=True)
    mean_identity, flat_identity = mean_body.bake(), flat_body.bake()
    flat_q, _ = mano_from_classic(flat_identity, MANOClassic())
    torch.testing.assert_close(flat_q[3:], flat_identity.q_neutral[3:], atol=1e-12, rtol=0)
    torch.testing.assert_close(flat_q[:3], flat_identity.pelvis, atol=1e-12, rtol=0)
    mean_q, _ = mano_from_classic(mean_identity, MANOClassic())
    hand_mean = mean_body.spec.meta["hand_mean"]["right"]
    full = mano_to_classic(mean_identity, mean_q)
    torch.testing.assert_close(full.hand_pose, torch.zeros_like(hand_mean), atol=1e-10, rtol=0)
    assert not torch.allclose(mean_identity.forward(mean_q).vertices, flat_identity.forward(flat_q).vertices)
