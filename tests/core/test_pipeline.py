"""Unit tests for the shared tensor pipeline."""

from __future__ import annotations

import torch
from better_robot.lie import se3

from better_human.core.pipeline import (
    blend_shapes,
    lbs,
    linear_corrective,
    pose_feature_rotmat_minus_identity,
    regress_joints,
    skinning_transforms,
)


def test_blend_shapes_and_linear_corrective_preserve_batch_axes() -> None:
    coefficients = torch.tensor([[1.0, 2.0], [-1.0, 0.5]], dtype=torch.float32)
    basis = torch.arange(18, dtype=torch.float32).reshape(3, 3, 2)
    expected = torch.stack((basis[..., 0] + 2 * basis[..., 1], -basis[..., 0] + 0.5 * basis[..., 1]))
    torch.testing.assert_close(blend_shapes(coefficients, basis), expected)

    feature = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    posedirs = torch.arange(12, dtype=torch.float32).reshape(2, 6)
    torch.testing.assert_close(linear_corrective(feature, posedirs), (feature @ posedirs).reshape(2, 2, 3))


def test_regress_joints_matches_einsum_with_two_batch_axes() -> None:
    dense = torch.tensor(
        [[0.5, 0.5, 0.0, 0.0], [0.0, 0.25, 0.25, 0.5]],
        dtype=torch.float32,
    )
    vertices = torch.arange(2 * 3 * 4 * 3, dtype=torch.float32).reshape(2, 3, 4, 3)
    expected = torch.einsum("jv,...vc->...jc", dense, vertices)
    torch.testing.assert_close(regress_joints(dense, vertices), expected)


def test_pose_feature_is_zero_at_rest_and_excludes_root_by_contract() -> None:
    quaternions = torch.zeros(2, 5, 4, dtype=torch.float32)
    quaternions[..., 3] = 1.0
    feature = pose_feature_rotmat_minus_identity(quaternions)
    assert feature.shape == (2, 45)
    torch.testing.assert_close(feature, torch.zeros_like(feature))


def test_pose_feature_normalizes_quaternions_like_betterrobot_fk() -> None:
    quaternion = torch.tensor([[0.1, -0.2, 0.3, 0.9]], dtype=torch.float32)
    expected = pose_feature_rotmat_minus_identity(quaternion)
    torch.testing.assert_close(pose_feature_rotmat_minus_identity(2.0 * quaternion), expected)


def test_skinning_transforms_and_padded_lbs_use_only_stored_influences() -> None:
    rest_joints = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=torch.float32)
    world_poses = se3.identity(batch_shape=(2,), dtype=torch.float32)
    world_poses = torch.cat((world_poses[..., :3] + torch.tensor([[1.0, 0.0, 0.0], [4.0, 0.0, 0.0]]), world_poses[..., 3:]), dim=-1)
    transforms = skinning_transforms(world_poses, rest_joints)
    vertices = torch.tensor([[0.0, 1.0, 0.0], [2.0, 1.0, 0.0]], dtype=torch.float32)
    weights = torch.tensor([[1.0, 0.0], [0.25, 0.75]], dtype=torch.float32)
    indices = torch.tensor([[0, 0], [0, 1]], dtype=torch.long)
    expected = torch.tensor([[1.0, 1.0, 0.0], [3.75, 1.0, 0.0]], dtype=torch.float32)
    torch.testing.assert_close(lbs(vertices, transforms, weights, indices), expected)
