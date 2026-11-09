"""The rigs' corrective features and MHR's dense corrective weight, against reference formulas."""

from __future__ import annotations

import torch

from better_human.families.mhr import _dense_weight_from_sparse
from better_human.families.mhr_rig import _pose_feature_6d_from_euler
from better_human.families.regressed_rig import _pose_feature_rotmat_minus_identity
from tests.support.marks import FLOAT32_GRADCHECK


def _reference_6d_feature(angles: torch.Tensor) -> torch.Tensor:
    cosine = torch.cos(angles)
    sine = torch.sin(angles)
    cx, cy, cz = cosine.unbind(dim=-1)
    sx, sy, sz = sine.unbind(dim=-1)
    matrix = torch.stack(
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
    feature = torch.cat((matrix[..., :, 0], matrix[..., :, 1]), dim=-1)
    identity = feature.new_tensor((1.0, 0.0, 0.0, 0.0, 1.0, 0.0))
    return (feature - identity).flatten(start_dim=-2)


def test_rotmat_pose_feature_is_zero_at_rest_and_excludes_root_by_contract() -> None:
    quaternions = torch.zeros(2, 5, 4, dtype=torch.float32)
    quaternions[..., 3] = 1.0
    feature = _pose_feature_rotmat_minus_identity(quaternions)
    assert feature.shape == (2, 45)
    torch.testing.assert_close(feature, torch.zeros_like(feature))


def test_rotmat_pose_feature_normalizes_quaternions_like_betterrobot_fk() -> None:
    quaternion = torch.tensor([[0.1, -0.2, 0.3, 0.9]], dtype=torch.float32)
    expected = _pose_feature_rotmat_minus_identity(quaternion)
    torch.testing.assert_close(_pose_feature_rotmat_minus_identity(2.0 * quaternion), expected)


def test_euler_6d_pose_feature_matches_reference_formula_with_leading_axes() -> None:
    generator = torch.Generator().manual_seed(20260721)
    angles = torch.rand((2, 3, 5, 3), generator=generator, dtype=torch.float32) * 2 - 1

    actual = _pose_feature_6d_from_euler(angles)

    assert actual.shape == (2, 3, 30)
    assert actual.dtype == angles.dtype
    assert actual.device == angles.device
    torch.testing.assert_close(actual, _reference_6d_feature(angles), rtol=0, atol=torch.finfo(angles.dtype).eps)


def test_dense_weight_from_sparse_matches_reference() -> None:
    mask = torch.tensor(
        [[True, False, True, False, False], [False, True, False, True, True], [True, False, False, False, True]],
        dtype=torch.bool,
    )
    indices = mask.nonzero().T
    sparse_values = torch.linspace(-0.4, 0.6, indices.shape[1], dtype=torch.float32)
    expected = torch.zeros(mask.shape, dtype=torch.float32)
    expected[indices[0], indices[1]] = sparse_values
    assert torch.equal(_dense_weight_from_sparse(indices, sparse_values, mask), expected)


@FLOAT32_GRADCHECK
def test_float32_gradcheck_6d_pose_feature() -> None:
    angles = torch.tensor(
        [[[0.2, -0.3, 0.4], [-0.4, 0.1, 0.3]]],
        dtype=torch.float32,
        requires_grad=True,
    )
    assert torch.autograd.gradcheck(
        _pose_feature_6d_from_euler,
        (angles,),
        eps=1e-3,
        atol=2e-3,
        rtol=2e-2,
        fast_mode=True,
    )
