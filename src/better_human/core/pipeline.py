"""Shared tensor stages for parametric-body deformation."""

from __future__ import annotations

import torch
from better_robot.lie import se3, so3
from torch.nn import functional


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


def pose_feature_6d_from_euler(joint_euler_angles: torch.Tensor) -> torch.Tensor:
    """Return MHR-style 6D features from local XYZ Euler channels.

    ``joint_euler_angles`` has shape ``(..., J, 3)`` and stores ``rx, ry,
    rz``. The local rotation is ``Rz(rz) @ Ry(ry) @ Rx(rx)``. The first two
    matrix columns, minus their identity values, are flattened to
    ``(..., J * 6)``.
    """
    cosine = torch.cos(joint_euler_angles)
    sine = torch.sin(joint_euler_angles)
    cx, cy, cz = cosine.unbind(dim=-1)
    sx, sy, sz = sine.unbind(dim=-1)
    feature = torch.stack(
        (
            cy * cz,
            cy * sz,
            -sy,
            -cx * sz + sx * sy * cz,
            cx * cz + sx * sy * sz,
            sx * cy,
        ),
        dim=-1,
    )
    identity = feature.new_tensor((1.0, 0.0, 0.0, 0.0, 1.0, 0.0))
    return (feature - identity).flatten(start_dim=-2)


def linear_corrective(feature: torch.Tensor, posedirs: torch.Tensor) -> torch.Tensor:
    """Apply a linear corrective basis.

    ``feature`` has shape ``(..., P)`` and ``posedirs`` has shape
    ``(P, V * 3)``. The result has shape ``(..., V, 3)``.
    """
    return (feature @ posedirs).reshape(*feature.shape[:-1], -1, 3)


def dense_weight_from_sparse(
    sparse_indices: torch.Tensor,
    sparse_values: torch.Tensor,
    sparse_mask: torch.Tensor,
) -> torch.Tensor:
    """Bake coordinate-list values into one immutable dense weight matrix."""
    if sparse_indices.dtype != torch.long or sparse_indices.ndim != 2 or sparse_indices.shape[0] != 2:
        raise ValueError("sparse_indices must be an int64 tensor with shape (2, K)")
    if sparse_values.ndim != 1 or sparse_values.shape[0] != sparse_indices.shape[1]:
        raise ValueError("sparse_values must have shape (K,) matching sparse_indices")
    if sparse_mask.dtype != torch.bool or sparse_mask.ndim != 2:
        raise ValueError("sparse_mask must be a bool tensor with shape (H, P)")
    if not torch.equal(sparse_indices, sparse_mask.nonzero().T):
        raise ValueError("sparse_indices must exactly match sparse_mask.nonzero().T")
    dense_weight = sparse_values.new_zeros(sparse_mask.shape)
    return dense_weight.index_put((sparse_indices[0], sparse_indices[1]), sparse_values)


def dense_skinning_weights(
    weights_vk: torch.Tensor,
    indices_vk: torch.Tensor,
    joint_count: int,
) -> torch.Tensor:
    """Expand padded ``(V, K)`` influences into a dense ``(V, J)`` matrix."""
    weights_vj = weights_vk.new_zeros((weights_vk.shape[0], joint_count))
    return weights_vj.scatter_add_(1, indices_vk, weights_vk)


def mlp_corrective(
    feature: torch.Tensor,
    first_layer_weight: torch.Tensor,
    output_weight: torch.Tensor,
) -> torch.Tensor:
    """Apply a baked-dense first layer, ReLU, and dense corrective output.

    ``feature`` has shape ``(..., P)``, ``first_layer_weight`` is ``(H, P)``,
    and ``output_weight`` is stored as ``(H, V * 3)``. The returned offsets
    have shape ``(..., V, 3)``.
    """
    hidden = functional.linear(feature, first_layer_weight).relu()
    return (hidden @ output_weight).reshape(*feature.shape[:-1], -1, 3)


def skinning_transforms(
    world_joint_poses: torch.Tensor,
    bind_poses: torch.Tensor,
) -> torch.Tensor:
    """Build ``T_world @ inverse(T_rest)`` for identity-oriented rest joints.

    ``world_joint_poses`` has shape ``(..., J, 7)`` and ``bind_poses`` has
    shape ``(..., J, 3)``.
    """
    if bind_poses.shape[-1:] != (3,):
        raise ValueError(f"bind_poses must have shape (..., J, 3); got {tuple(bind_poses.shape)}")
    identity_quaternions = bind_poses.new_zeros((*bind_poses.shape[:-1], 4))
    identity_quaternions[..., 3] = 1.0
    bind_poses = torch.cat((bind_poses, identity_quaternions), dim=-1)
    return se3.compose(world_joint_poses, se3.inverse(bind_poses))


def skinning_transforms_from_inverse(
    world_joint_poses: torch.Tensor,
    inverse_bind_poses: torch.Tensor,
) -> torch.Tensor:
    """Build skinning transforms from authored inverse bind poses.

    Both inputs use scalar-last 7D SE(3) poses with trailing shape ``(J, 7)``.
    This path preserves an authoritative stored inverse instead of numerically
    reinverting its companion bind matrix.
    """
    if inverse_bind_poses.shape[-1:] != (7,):
        raise ValueError(f"inverse_bind_poses must have shape (..., J, 7); got {tuple(inverse_bind_poses.shape)}")
    return se3.compose(world_joint_poses, inverse_bind_poses)


def lbs(
    rest_vertices: torch.Tensor,
    transforms: torch.Tensor,
    weights_vj: torch.Tensor,
    *,
    joint_scales: torch.Tensor | None = None,
    bind_joint_positions: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply dense matrix-blend linear blend skinning.

    Parameters use shapes ``rest_vertices (..., V, 3)``, ``transforms
    (..., J, 7)``, and ``weights_vj (V, J)``. Joint transforms are converted
    to affines and blended once before being applied to each vertex. Optional
    accumulated ``joint_scales (..., J)`` apply a similarity transform about
    ``bind_joint_positions (..., J, 3)``.
    """
    rotations = so3.to_matrix(transforms[..., 3:7])
    translations = transforms[..., :3]
    if joint_scales is not None:
        if bind_joint_positions is None:
            raise ValueError("bind_joint_positions are required when joint_scales are provided")
        rotated_bind_positions = (rotations @ bind_joint_positions.unsqueeze(-1)).squeeze(-1)
        scale_offsets = joint_scales - 1.0
        rotations = rotations * joint_scales[..., :, None, None]
        translations = translations - scale_offsets[..., :, None] * rotated_bind_positions

    batch_shape = torch.broadcast_shapes(rotations.shape[:-3], translations.shape[:-2])
    rotations = rotations.expand(*batch_shape, *rotations.shape[-3:])
    translations = translations.expand(*batch_shape, *translations.shape[-2:])
    joint_affines = torch.cat((rotations, translations.unsqueeze(-1)), dim=-1).flatten(start_dim=-2)
    blended_affines = torch.einsum("vj,...jx->...vx", weights_vj, joint_affines).unflatten(-1, (3, 4))
    return (blended_affines[..., :3] @ rest_vertices.unsqueeze(-1)).squeeze(-1) + blended_affines[..., 3]


__all__ = [
    "blend_shapes",
    "dense_skinning_weights",
    "dense_weight_from_sparse",
    "lbs",
    "linear_corrective",
    "mlp_corrective",
    "pose_feature_6d_from_euler",
    "pose_feature_rotmat_minus_identity",
    "regress_joints",
    "skinning_transforms",
    "skinning_transforms_from_inverse",
]
