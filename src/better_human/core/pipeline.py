"""Shared tensor stages for linear parametric bodies."""

from __future__ import annotations

import torch
from better_robot.lie import se3, so3


def blend_shapes(coefficients: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Blend linear basis columns.

    Parameters
    ----------
    coefficients
        Shape ``(..., S)``.
    basis
        Shape ``(V, 3, S)``.

    Returns
    -------
    torch.Tensor
        Displacements with shape ``(..., V, 3)``.
    """
    return torch.einsum("...s,vcs->...vc", coefficients, basis)


def regress_joints(regressor: torch.Tensor, vertices: torch.Tensor) -> torch.Tensor:
    """Regress joint centers with a dense linear map.

    ``regressor`` has shape ``(J, V)`` and ``vertices`` has shape
    ``(..., V, 3)``. The result has shape ``(..., J, 3)``.
    """
    return torch.einsum("jv,...vc->...jc", regressor, vertices)


def pose_feature_rotmat_minus_identity(joint_quaternions: torch.Tensor) -> torch.Tensor:
    """Return flattened non-root ``(R - I)`` pose features.

    ``joint_quaternions`` contains only non-root joints and has shape
    ``(..., J - 1, 4)`` in scalar-last order. The result has shape
    ``(..., (J - 1) * 9)``.
    """
    rotation_matrices = so3.to_matrix(so3.normalize(joint_quaternions))
    identity = torch.eye(3, dtype=joint_quaternions.dtype, device=joint_quaternions.device)
    return (rotation_matrices - identity).flatten(start_dim=-3)


def linear_corrective(feature: torch.Tensor, posedirs: torch.Tensor) -> torch.Tensor:
    """Apply a linear corrective basis.

    ``feature`` has shape ``(..., P)`` and ``posedirs`` has shape
    ``(P, V * 3)``. The result has shape ``(..., V, 3)``.
    """
    return (feature @ posedirs).reshape(*feature.shape[:-1], -1, 3)


def skinning_transforms(
    world_joint_poses: torch.Tensor,
    rest_joints: torch.Tensor,
) -> torch.Tensor:
    """Build ``T_world @ inverse(T_rest)`` for every native joint.

    ``world_joint_poses`` has shape ``(..., J, 7)`` and ``rest_joints`` has
    shape ``(..., J, 3)``. Rest-joint orientations are identity.
    """
    identity_quaternions = rest_joints.new_zeros((*rest_joints.shape[:-1], 4))
    identity_quaternions[..., 3] = 1.0
    rest_poses = torch.cat((rest_joints, identity_quaternions), dim=-1)
    return se3.compose(world_joint_poses, se3.inverse(rest_poses))


def lbs(
    rest_vertices: torch.Tensor,
    transforms: torch.Tensor,
    weights_vk: torch.Tensor,
    indices_vk: torch.Tensor,
) -> torch.Tensor:
    """Apply padded sparse linear blend skinning.

    Parameters use shapes ``rest_vertices (..., V, 3)``, ``transforms
    (..., J, 7)``, and ``weights_vk`` / ``indices_vk (V, K)``. Only the
    ``K`` stored influences are gathered for each vertex.
    """
    selected_transforms = transforms[..., indices_vk, :]
    transformed_vertices = se3.act(selected_transforms, rest_vertices[..., :, None, :])
    return (transformed_vertices * weights_vk[..., None]).sum(dim=-2)


__all__ = [
    "blend_shapes",
    "lbs",
    "linear_corrective",
    "pose_feature_rotmat_minus_identity",
    "regress_joints",
    "skinning_transforms",
]
