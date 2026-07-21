"""Tests for generic partial-body load-time reduction."""

from __future__ import annotations

import pytest
import torch

from better_human.core.pipeline import linear_corrective, pose_feature_rotmat_minus_identity
from better_human.core.reduction import reduce_joint_data, slice_parameter_basis

NAMES = ("root", "branch", "leaf_a", "leaf_b")
PARENTS = (-1, 0, 1, 1)


def _joint_data() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    regressor = torch.arange(4 * 2, dtype=torch.float32).reshape(4, 2)
    weights = torch.tensor(
        [[0.1, 0.2, 0.3, 0.3], [0.2, 0.1, 0.4, 0.2]],
        dtype=torch.float32,
    )
    posedirs = torch.arange(2 * 3 * 27, dtype=torch.float32).reshape(2, 3, 27)
    return regressor, weights, posedirs


def test_reduction_rejects_root_and_nonterminal_drops() -> None:
    regressor, weights, posedirs = _joint_data()
    with pytest.raises(ValueError, match="root joint cannot"):
        reduce_joint_data(NAMES, PARENTS, regressor, weights, posedirs, [0])
    with pytest.raises(ValueError, match="terminal subtrees"):
        reduce_joint_data(NAMES, PARENTS, regressor, weights, posedirs, [1, 2])


def test_reduction_transfers_weights_renormalizes_and_slices_regressor() -> None:
    regressor, weights, posedirs = _joint_data()
    reduced = reduce_joint_data(
        NAMES,
        PARENTS,
        regressor,
        weights,
        posedirs,
        [2, 3],
    )
    names, parents, reduced_regressor, reduced_weights, reduced_posedirs, survivors = reduced
    assert names == ("root", "branch")
    assert parents == (-1, 0)
    assert survivors == (0, 1)
    torch.testing.assert_close(reduced_regressor, regressor[:2])
    torch.testing.assert_close(reduced_weights.sum(dim=-1), torch.ones(2))
    expected = torch.stack((weights[:, 0], weights[:, 1:].sum(dim=-1)), dim=-1)
    expected = expected / expected.sum(dim=-1, keepdim=True)
    torch.testing.assert_close(reduced_weights, expected)
    torch.testing.assert_close(reduced_posedirs, posedirs[..., :9])


def test_corrective_slice_is_exact_when_dropped_joint_is_at_rest() -> None:
    regressor, weights, posedirs = _joint_data()
    *_, reduced_posedirs, survivors = reduce_joint_data(
        NAMES,
        PARENTS,
        regressor,
        weights,
        posedirs,
        [3],
    )
    assert survivors == (0, 1, 2)
    torch.testing.assert_close(reduced_posedirs, posedirs[..., :18], rtol=0, atol=0)

    quaternions = torch.tensor(
        [[0.1, 0.2, -0.1, 0.9], [-0.2, 0.1, 0.15, 0.95], [0.0, 0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    full_feature = pose_feature_rotmat_minus_identity(quaternions)
    reduced_feature = pose_feature_rotmat_minus_identity(quaternions[:2])
    full_matrix = posedirs.reshape(-1, posedirs.shape[-1]).T
    reduced_matrix = reduced_posedirs.reshape(-1, reduced_posedirs.shape[-1]).T
    torch.testing.assert_close(
        linear_corrective(full_feature, full_matrix),
        linear_corrective(reduced_feature, reduced_matrix),
        rtol=0,
        atol=0,
    )


def test_parameter_basis_selects_named_ranges_in_requested_order() -> None:
    basis = torch.arange(2 * 3 * 8, dtype=torch.float32).reshape(2, 3, 8)
    selected, groups = slice_parameter_basis(
        basis,
        {"shape": slice(0, 5), "expression": slice(5, 8)},
        {"shape": 2, "expression": 1},
    )
    assert groups == {"shape": 2, "expression": 1}
    torch.testing.assert_close(selected, torch.cat((basis[..., :2], basis[..., 5:6]), dim=-1))


def test_parameter_basis_rejects_invalid_shapes_slices_and_widths() -> None:
    basis = torch.zeros(2, 3, 8)
    with pytest.raises(ValueError, match=r"shape \(V, 3, S\)"):
        slice_parameter_basis(torch.zeros(2, 8), {"shape": slice(0, 8)}, {"shape": 1})
    with pytest.raises(ValueError, match="must fit basis width 8"):
        slice_parameter_basis(basis, {"shape": slice(0, 9)}, {"shape": 1})
    with pytest.raises(ValueError, match="width must be"):
        slice_parameter_basis(basis, {"shape": slice(0, 8)}, {"shape": True})


@pytest.mark.parametrize(
    ("regressor", "weights", "posedirs", "message"),
    [
        (torch.zeros(20), torch.ones(5, 4), torch.zeros(5, 3, 27), "joint_regressor"),
        (torch.zeros(4, 5), torch.ones(20), torch.zeros(5, 3, 27), "skinning_weights"),
        (torch.zeros(4, 5), torch.ones(5, 4), torch.zeros(5, 27), "posedirs"),
        (torch.zeros(4, 6), torch.ones(5, 4), torch.zeros(5, 3, 27), "same vertex count"),
        (torch.zeros(4, 5), torch.zeros(5, 4), torch.zeros(5, 3, 27), "positive finite sums"),
    ],
)
def test_reduction_rejects_malformed_joint_data(
    regressor: torch.Tensor,
    weights: torch.Tensor,
    posedirs: torch.Tensor,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        reduce_joint_data(NAMES, PARENTS, regressor, weights, posedirs, [3])
