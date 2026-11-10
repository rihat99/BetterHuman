"""Unit tests for the skinning path and the shared tensor stages."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
import torch
from better_robot.lie import so3
from torch.nn import functional

from better_human.core.skinning import blend_shapes, linear_blend_skinning, skin_affines
from tests.support.marks import FLOAT32_GRADCHECK
from tests.support.synthetic import flame_spec


def test_blend_shapes_preserves_batch_axes() -> None:
    coefficients = torch.tensor([[1.0, 2.0], [-1.0, 0.5]], dtype=torch.float32)
    basis = torch.arange(18, dtype=torch.float32).reshape(3, 3, 2)
    expected = torch.stack((basis[..., 0] + 2 * basis[..., 1], -basis[..., 0] + 0.5 * basis[..., 1]))
    torch.testing.assert_close(blend_shapes(coefficients, basis), expected)


@FLOAT32_GRADCHECK
def test_float32_gradcheck_skin_affines_and_linear_blend_skinning() -> None:
    world_poses = torch.tensor([[0.2, -0.1, 0.3, 0.05, -0.02, 0.04, 0.9975]], dtype=torch.float32)
    bind_poses = torch.tensor([[-0.1, 0.2, -0.3, -0.03, 0.04, 0.02, 0.9985]], dtype=torch.float32)
    transforms = _affine(world_poses).requires_grad_(True)
    inverse_binds = (1.2 * _affine(bind_poses)).requires_grad_(True)  # a scaled bind, as MHR folds its scales
    vertices = torch.tensor([[0.4, -0.2, 0.1]], dtype=torch.float32, requires_grad=True)
    weights = torch.ones((1, 1), dtype=torch.float32)

    def skinning(points: torch.Tensor, joint_transforms: torch.Tensor, binds: torch.Tensor) -> torch.Tensor:
        return linear_blend_skinning(points, skin_affines(joint_transforms, binds), weights=weights)

    assert torch.autograd.gradcheck(
        skinning,
        (vertices, transforms, inverse_binds),
        eps=1e-3,
        atol=1e-2,
        rtol=3e-2,
        fast_mode=True,
    )


def test_duplicate_skin_influences_accumulate(tmp_path: Path) -> None:
    spec = flame_spec(tmp_path)
    num_vertices = spec.num_vertices
    weights = torch.full((num_vertices, 2), 0.5, dtype=spec.dtype)
    weights[1] = torch.tensor([0.25, 0.75], dtype=spec.dtype)
    indices = torch.ones((num_vertices, 2), dtype=torch.long)
    indices[0], indices[1] = torch.tensor([0, 0]), torch.tensor([0, 2])  # vertex 0 lists joint 0 twice
    matrix = dataclasses.replace(spec, skin_weights=weights, skin_indices=indices).skin_matrix()
    assert torch.equal(matrix[0], torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0], dtype=spec.dtype))
    assert torch.equal(matrix[1], torch.tensor([0.25, 0.0, 0.75, 0.0, 0.0], dtype=spec.dtype))

    affines = _affine(_random_world_poses(1, 5, spec.dtype))[0]
    rest = spec.template_vertices
    skinned = linear_blend_skinning(rest, affines, weights=matrix)
    expected = sum(weights[:, slot, None] * _apply(affines[indices[:, slot]], rest) for slot in range(2))
    torch.testing.assert_close(skinned, expected, rtol=0, atol=1e-14)


def _random_world_poses(batch: int, joints: int, dtype: torch.dtype) -> torch.Tensor:
    generator = torch.Generator().manual_seed(3)
    quaternions = functional.normalize(torch.randn(batch, joints, 4, generator=generator, dtype=dtype), dim=-1)
    translations = torch.randn(batch, joints, 3, generator=generator, dtype=dtype)
    return torch.cat((translations, quaternions), dim=-1)


def _affine(poses: torch.Tensor) -> torch.Tensor:
    return torch.cat((so3.to_matrix(poses[..., 3:]), poses[..., :3, None]), dim=-1)


def _apply(affines: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    return (affines[..., :3] @ points[..., None]).squeeze(-1) + affines[..., 3]


def test_skin_affines_with_rest_joint_binds_skin_about_the_rest_joints() -> None:
    generator = torch.Generator().manual_seed(4)
    world = _affine(_random_world_poses(2, 5, torch.float64))
    rest_joints = torch.randn(5, 3, generator=generator, dtype=torch.float64)
    rest_vertices = torch.randn(2, 40, 3, generator=generator, dtype=torch.float64)
    weights = torch.rand(40, 5, generator=generator, dtype=torch.float64)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    inverse_binds = torch.cat((torch.eye(3, dtype=torch.float64).expand(5, 3, 3), -rest_joints[..., None]), dim=-1)
    affines = skin_affines(world, inverse_binds)
    # The SMPL family's [I | -rest] bind is bitwise the rest-joint form (decision record 0003).
    assert torch.equal(affines[..., :3], world[..., :3])
    assert torch.equal(affines[..., 3], world[..., 3] - (world[..., :3] @ rest_joints[..., None]).squeeze(-1))
    expected = torch.einsum("vj,bvjc->bvc", weights, _apply(world[:, None], rest_vertices[:, :, None] - rest_joints))
    torch.testing.assert_close(
        linear_blend_skinning(rest_vertices, affines, weights=weights), expected, atol=1e-6, rtol=0
    )


@pytest.mark.parametrize("scaled", [False, True])
def test_skin_affines_with_scaled_inverse_binds_scale_about_the_bind_positions(scaled: bool) -> None:
    generator = torch.Generator().manual_seed(5)
    world = _affine(_random_world_poses(3, 4, torch.float64))
    inverse_binds = _affine(_random_world_poses(1, 4, torch.float64)[0])
    rest_vertices = torch.randn(30, 3, generator=generator, dtype=torch.float64)
    weights = torch.rand(30, 4, generator=generator, dtype=torch.float64)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    ones = torch.ones(3, 4, dtype=torch.float64)
    scales = ones + torch.rand(3, 4, generator=generator, dtype=torch.float64) - 0.5 if scaled else ones
    pivots = torch.randn(4, 3, generator=generator, dtype=torch.float64)
    # MHR's fold [s R_b | t_b - (s - 1) R_b c] (decision record 0003).
    rotations, translations = inverse_binds[..., :3], inverse_binds[..., 3]
    folded_translations = translations - (scales - 1.0)[..., None] * (rotations @ pivots[..., None]).squeeze(-1)
    folded = torch.cat((scales[..., None, None] * rotations, folded_translations[..., None]), dim=-1)
    # Reference: scale each vertex about the joint's bind position, then apply the inverse bind and the world transform.
    scaled_vertices = pivots + scales[:, None, :, None] * (rest_vertices[:, None] - pivots)
    expected = torch.einsum("vj,bvjc->bvc", weights, _apply(world[:, None], _apply(inverse_binds, scaled_vertices)))
    affines = skin_affines(world, folded)
    torch.testing.assert_close(
        linear_blend_skinning(rest_vertices, affines, weights=weights), expected, atol=1e-6, rtol=0
    )
