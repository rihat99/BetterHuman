"""Tests for shared SMPL-family classic converter algebra."""

from __future__ import annotations

import torch

from better_human.core.classic import (
    axis_angle_to_quaternion,
    classic_root_to_free_flyer,
    free_flyer_to_classic_root,
    quaternion_to_axis_angle,
)


def test_axis_angle_quaternion_round_trip_is_batched() -> None:
    axis_angle = torch.tensor(
        [[[0.2, -0.1, 0.3], [0.0, 0.0, 0.0]], [[-0.3, 0.4, 0.1], [0.1, 0.2, -0.2]]],
        dtype=torch.float32,
    )
    recovered = quaternion_to_axis_angle(axis_angle_to_quaternion(axis_angle))
    torch.testing.assert_close(recovered, axis_angle, rtol=2e-6, atol=2e-6)


def test_converter_preserves_negative_quaternion_sign_and_large_angle() -> None:
    axis_angle = torch.tensor([4.0, 0.0, 0.0], dtype=torch.float32)
    quaternion = axis_angle_to_quaternion(axis_angle)
    assert quaternion[-1] < 0
    torch.testing.assert_close(quaternion_to_axis_angle(quaternion), axis_angle, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(
        axis_angle_to_quaternion(quaternion_to_axis_angle(-quaternion)),
        -quaternion,
        rtol=2e-6,
        atol=2e-6,
    )


def test_root_wart_round_trip_broadcasts_shape_batch() -> None:
    global_orient = torch.tensor([0.4, -0.2, 0.1], dtype=torch.float32)
    transl = torch.tensor([[1.0, 2.0, 3.0], [-1.0, 0.5, 0.25]], dtype=torch.float32)
    pelvis = torch.tensor([[0.1, -0.2, 0.3], [0.4, 0.2, -0.1]], dtype=torch.float32)
    root = classic_root_to_free_flyer(global_orient, transl, pelvis)
    recovered_orient, recovered_transl = free_flyer_to_classic_root(root, pelvis)
    torch.testing.assert_close(recovered_orient, global_orient.expand(2, 3), rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(recovered_transl, transl)
