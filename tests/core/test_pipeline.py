"""Unit tests for the shared tensor pipeline."""

from __future__ import annotations

import pytest
import torch
from better_robot.lie import se3, so3
from torch.nn import functional

from better_human.core.pipeline import (
    blend_shapes,
    dense_skinning_weights,
    dense_weight_from_sparse,
    lbs,
    linear_corrective,
    mlp_corrective,
    pose_feature_6d_from_euler,
    pose_feature_rotmat_minus_identity,
    regress_joints,
    skinning_transforms,
    skinning_transforms_from_inverse,
)


def _reference_euler_matrix(angles: torch.Tensor) -> torch.Tensor:
    cosine = torch.cos(angles)
    sine = torch.sin(angles)
    cx, cy, cz = cosine.unbind(dim=-1)
    sx, sy, sz = sine.unbind(dim=-1)
    return torch.stack(
        (
            cy * cz,
            -cx * sz + sx * sy * cz,
            sx * sz + cx * sy * cz,
            cy * sz,
            cx * cz + sx * sy * sz,
            -sx * cz + cx * sy * sz,
            -sy,
            sx * cy,
            cx * cy,
        ),
        dim=-1,
    ).reshape(*angles.shape[:-1], 3, 3)


def _reference_6d_feature(angles: torch.Tensor) -> torch.Tensor:
    matrix = _reference_euler_matrix(angles)
    feature = torch.cat((matrix[..., :, 0], matrix[..., :, 1]), dim=-1)
    identity = feature.new_tensor((1.0, 0.0, 0.0, 0.0, 1.0, 0.0))
    return (feature - identity).flatten(start_dim=-2)


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


def test_euler_6d_pose_feature_matches_reference_formula_with_leading_axes() -> None:
    generator = torch.Generator().manual_seed(20260721)
    angles = torch.rand((2, 3, 5, 3), generator=generator, dtype=torch.float32) * 2 - 1

    actual = pose_feature_6d_from_euler(angles)

    assert actual.shape == (2, 3, 30)
    assert actual.dtype == angles.dtype
    assert actual.device == angles.device
    torch.testing.assert_close(actual, _reference_6d_feature(angles), rtol=0, atol=torch.finfo(angles.dtype).eps)


def test_dense_sparse_weight_and_mlp_match_reference_without_mutating_input() -> None:
    mask = torch.tensor(
        [[True, False, True, False, False], [False, True, False, True, True], [True, False, False, False, True]],
        dtype=torch.bool,
    )
    indices = mask.nonzero().T
    sparse_values = torch.linspace(-0.4, 0.6, indices.shape[1], dtype=torch.float32)
    first_weight = dense_weight_from_sparse(indices, sparse_values, mask)
    expected_weight = torch.zeros_like(first_weight)
    expected_weight[indices[0], indices[1]] = sparse_values
    assert torch.equal(first_weight, expected_weight)

    feature = torch.linspace(-0.5, 0.7, 2 * 3 * 5, dtype=torch.float32).reshape(2, 3, 5)
    original = feature.clone()
    output_weight = torch.linspace(-0.3, 0.4, 3 * 12, dtype=torch.float32).reshape(3, 12)
    expected = (functional.linear(feature, first_weight).relu() @ output_weight).reshape(2, 3, 4, 3)

    actual = mlp_corrective(feature, first_weight, output_weight)

    assert torch.equal(feature, original)
    assert actual.shape == (2, 3, 4, 3)
    assert actual.dtype == feature.dtype
    assert actual.device == feature.device
    torch.testing.assert_close(actual, expected)


def test_dense_skinning_weights_and_matrix_lbs_accumulate_duplicate_influences() -> None:
    rest_joints = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=torch.float32)
    world_poses = se3.identity(batch_shape=(2,), dtype=torch.float32)
    world_poses = torch.cat(
        (world_poses[..., :3] + torch.tensor([[1.0, 0.0, 0.0], [4.0, 0.0, 0.0]]), world_poses[..., 3:]), dim=-1
    )
    transforms = skinning_transforms(world_poses, rest_joints)
    vertices = torch.tensor([[0.0, 1.0, 0.0], [2.0, 1.0, 0.0]], dtype=torch.float32)
    weights = torch.tensor([[0.4, 0.6], [0.25, 0.75]], dtype=torch.float32)
    indices = torch.tensor([[0, 0], [0, 1]], dtype=torch.long)
    weight_matrix = dense_skinning_weights(weights, indices, 2)
    assert torch.equal(weight_matrix, torch.tensor([[1.0, 0.0], [0.25, 0.75]]))
    assert weight_matrix.dtype == weights.dtype
    assert weight_matrix.device == weights.device
    expected = torch.tensor([[1.0, 1.0, 0.0], [3.75, 1.0, 0.0]], dtype=torch.float32)
    torch.testing.assert_close(lbs(vertices, transforms, weight_matrix), expected)


def test_scaled_lbs_matches_similarity_transform_about_nonidentity_bind_pose() -> None:
    half_angle = torch.tensor(torch.pi / 8, dtype=torch.float32)
    bind_quaternion = torch.tensor((0.0, 0.0, half_angle.sin(), half_angle.cos()), dtype=torch.float32)
    world_quaternion = torch.tensor((half_angle.sin(), 0.0, 0.0, half_angle.cos()), dtype=torch.float32)
    bind_pose = torch.cat((torch.tensor((1.0, -0.5, 0.25)), bind_quaternion)).reshape(1, 7)
    world_pose = torch.cat((torch.tensor((-0.25, 0.75, 1.5)), world_quaternion)).reshape(1, 7)
    vertex = torch.tensor([[1.5, 0.25, -0.75]], dtype=torch.float32)
    scale = torch.tensor((1.6,), dtype=torch.float32)
    transforms = skinning_transforms_from_inverse(world_pose, se3.inverse(bind_pose))
    assert transforms.dtype == world_pose.dtype
    assert transforms.device == world_pose.device

    actual = lbs(
        vertex,
        transforms,
        torch.ones((1, 1), dtype=torch.float32),
        joint_scales=scale,
        bind_joint_positions=bind_pose[..., :3],
    )

    bind_relative = so3.act(so3.inverse(bind_quaternion), vertex[0] - bind_pose[0, :3])
    expected = world_pose[0, :3] + scale[0] * so3.act(world_quaternion, bind_relative)
    assert actual.dtype == vertex.dtype
    assert actual.device == vertex.device
    torch.testing.assert_close(actual[0], expected)


def test_scaled_lbs_broadcasts_arbitrary_leading_axes() -> None:
    translations = torch.arange(2 * 3 * 3, dtype=torch.float32).reshape(2, 3, 1, 3) / 10
    transforms = se3.identity(batch_shape=(2, 3, 1), dtype=torch.float32)
    transforms = torch.cat((translations, transforms[..., 3:]), dim=-1)
    vertices = torch.tensor(
        [[[[0.25, -0.5, 1.0], [1.5, 0.75, -0.25]]], [[[0.5, 0.25, -1.0], [-0.5, 1.0, 0.5]]]],
        dtype=torch.float32,
    )
    scales = torch.tensor([[[1.0], [1.5], [2.0]]], dtype=torch.float32)
    weights = torch.ones((2, 1), dtype=torch.float32)

    actual = lbs(
        vertices,
        transforms,
        weights,
        joint_scales=scales,
        bind_joint_positions=torch.zeros((1, 3), dtype=torch.float32),
    )

    expected = translations + scales[..., None] * vertices
    assert actual.shape == (2, 3, 2, 3)
    torch.testing.assert_close(actual, expected)


def test_scaled_lbs_requires_bind_joint_positions() -> None:
    vertices = torch.zeros((1, 3), dtype=torch.float32)
    transforms = se3.identity(batch_shape=(1,), dtype=torch.float32)
    weights = torch.ones((1, 1), dtype=torch.float32)

    with pytest.raises(ValueError, match="bind_joint_positions are required"):
        lbs(vertices, transforms, weights, joint_scales=torch.ones(1))


def test_float32_gradcheck_6d_pose_feature() -> None:
    angles = torch.tensor(
        [[[0.2, -0.3, 0.4], [-0.4, 0.1, 0.3]]],
        dtype=torch.float32,
        requires_grad=True,
    )
    assert torch.autograd.gradcheck(
        pose_feature_6d_from_euler,
        (angles,),
        eps=1e-3,
        atol=2e-3,
        rtol=2e-2,
        fast_mode=True,
    )


def test_float32_gradcheck_mlp_corrective() -> None:
    feature = torch.tensor([[0.2, -0.1, 0.3]], dtype=torch.float32, requires_grad=True)
    first_weight = torch.tensor(
        [[0.1, -0.2, 0.3], [-0.1, 0.2, 0.1]],
        dtype=torch.float32,
        requires_grad=True,
    )
    output_weight = torch.tensor(
        [[0.2, -0.1, 0.3], [0.1, 0.4, -0.2]],
        dtype=torch.float32,
        requires_grad=True,
    )
    assert torch.autograd.gradcheck(
        mlp_corrective,
        (feature, first_weight, output_weight),
        eps=1e-3,
        atol=1e-2,
        rtol=3e-2,
        fast_mode=True,
    )


def test_float32_gradcheck_inverse_bind_transforms_and_scaled_lbs() -> None:
    world_poses = torch.tensor(
        [[0.2, -0.1, 0.3, 0.05, -0.02, 0.04, 0.9975]],
        dtype=torch.float32,
        requires_grad=True,
    )
    bind_poses = torch.tensor(
        [[-0.1, 0.2, -0.3, -0.03, 0.04, 0.02, 0.9985]],
        dtype=torch.float32,
    )
    inverse_bind_poses = se3.inverse(bind_poses).detach().requires_grad_(True)
    assert torch.autograd.gradcheck(
        skinning_transforms_from_inverse,
        (world_poses, inverse_bind_poses),
        eps=1e-3,
        atol=1e-2,
        rtol=3e-2,
        fast_mode=True,
    )

    transforms = skinning_transforms_from_inverse(
        world_poses.detach(),
        inverse_bind_poses.detach(),
    ).requires_grad_(True)
    vertices = torch.tensor([[0.4, -0.2, 0.1]], dtype=torch.float32, requires_grad=True)
    scales = torch.tensor([1.2], dtype=torch.float32, requires_grad=True)
    bind_positions = bind_poses.detach()[..., :3].requires_grad_(True)
    weights = torch.ones((1, 1), dtype=torch.float32)

    def scaled_skinning(
        points: torch.Tensor,
        skinning: torch.Tensor,
        joint_scales: torch.Tensor,
        joint_positions: torch.Tensor,
    ) -> torch.Tensor:
        return lbs(
            points,
            skinning,
            weights,
            joint_scales=joint_scales,
            bind_joint_positions=joint_positions,
        )

    assert torch.autograd.gradcheck(
        scaled_skinning,
        (vertices, transforms, scales, bind_positions),
        eps=1e-3,
        atol=1e-2,
        rtol=3e-2,
        fast_mode=True,
    )
