"""Pure converters between classic SMPL-family and BetterRobot poses."""

from __future__ import annotations

import torch
from better_robot.lie import so3


def axis_angle_to_quaternion(axis_angle: torch.Tensor) -> torch.Tensor:
    """Convert ``(..., 3)`` axis-angle vectors to scalar-last quaternions."""
    if axis_angle.shape[-1:] != (3,):
        raise ValueError(f"axis_angle must have shape (..., 3); got {tuple(axis_angle.shape)}")
    return so3.exp(axis_angle)


def quaternion_to_axis_angle(quaternion: torch.Tensor) -> torch.Tensor:
    """Convert scalar-last quaternions to sign-preserving axis angles.

    Unlike a principal SO(3) logarithm, this converter preserves both halves
    of the unit-quaternion sphere: negative-scalar quaternions map to angles
    in ``(pi, 2*pi]`` so converting back reproduces the input quaternion,
    not merely the same rotation.
    """
    if quaternion.shape[-1:] != (4,):
        raise ValueError(f"quaternion must have shape (..., 4); got {tuple(quaternion.shape)}")
    quaternion = so3.normalize(quaternion)
    vector = quaternion[..., :3]
    scalar = quaternion[..., 3:4]
    sin_half = vector.norm(dim=-1, keepdim=True)
    positive_singularity = (sin_half < 1e-7) & (scalar >= 0)
    safe_sin_half = torch.where(positive_singularity, torch.ones_like(sin_half), sin_half)
    angle = 2.0 * torch.atan2(safe_sin_half, scalar)
    full_factor = angle / safe_sin_half.clamp(min=1e-30)
    taylor_factor = 2.0 + (2.0 / 3.0) * sin_half.square()
    axis_angle = torch.where(positive_singularity, taylor_factor, full_factor) * vector

    negative_singularity = (sin_half < 1e-7) & (scalar < 0)
    fallback = torch.zeros_like(axis_angle)
    fallback[..., 0] = 2.0 * torch.pi
    return torch.where(negative_singularity.expand_as(axis_angle), fallback, axis_angle)


def classic_root_to_free_flyer(
    global_orient: torch.Tensor,
    transl: torch.Tensor,
    pelvis_offset: torch.Tensor,
) -> torch.Tensor:
    """Convert the SMPL pelvis-pivoted root to a free-flyer pose.

    All inputs have trailing shape ``(3,)``. The result has shape
    ``(..., 7)`` and stores the pelvis world position followed by a
    scalar-last quaternion.
    """
    if any(value.shape[-1:] != (3,) for value in (global_orient, transl, pelvis_offset)):
        raise ValueError("global_orient, transl, and pelvis_offset must each have shape (..., 3)")
    global_orient, transl, pelvis_offset = torch.broadcast_tensors(global_orient, transl, pelvis_offset)
    return torch.cat((transl + pelvis_offset, axis_angle_to_quaternion(global_orient)), dim=-1)


def free_flyer_to_classic_root(
    free_flyer: torch.Tensor,
    pelvis_offset: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert a free-flyer pelvis pose to SMPL ``global_orient, transl``."""
    if free_flyer.shape[-1:] != (7,):
        raise ValueError(f"free_flyer must have shape (..., 7); got {tuple(free_flyer.shape)}")
    if pelvis_offset.shape[-1:] != (3,):
        raise ValueError(f"pelvis_offset must have shape (..., 3); got {tuple(pelvis_offset.shape)}")
    batch_shape = torch.broadcast_shapes(free_flyer.shape[:-1], pelvis_offset.shape[:-1])
    free_flyer = free_flyer.expand(*batch_shape, 7)
    pelvis_offset = pelvis_offset.expand(*batch_shape, 3)
    global_orient = quaternion_to_axis_angle(free_flyer[..., 3:7])
    transl = free_flyer[..., :3] - pelvis_offset
    return global_orient, transl


__all__ = [
    "axis_angle_to_quaternion",
    "classic_root_to_free_flyer",
    "free_flyer_to_classic_root",
    "quaternion_to_axis_angle",
]
