"""Layer protocol and shared color and geometry helpers for studio layers.

A layer owns one group of scene nodes and reacts to session change kinds. The
front end builds every layer once, toggles visibility, and forwards change
notifications; the layer decides whether a change needs an in-place vertex
rewrite or a full node rebuild.
"""

from __future__ import annotations

import colorsys
import math
from typing import Protocol, runtime_checkable

import numpy as np
import torch
import viser

from ..state import StudioSession

#: Light blue used for the solid skinned mesh.
MESH_COLOR: tuple[int, int, int] = (166, 199, 242)
#: Red used for skeleton joint spheres.
JOINT_COLOR: tuple[int, int, int] = (214, 64, 64)
#: Gold used for the currently selected joint sphere.
SELECTED_JOINT_COLOR: tuple[int, int, int] = (240, 200, 64)
#: White-gray used for skeleton bone cylinders.
BONE_COLOR: tuple[int, int, int] = (222, 222, 226)
#: Green used for frame spheres.
FRAME_COLOR: tuple[int, int, int] = (86, 196, 118)
#: Orange used for the active (just-added / selected) frame sphere.
FRAME_ACTIVE_COLOR: tuple[int, int, int] = (240, 148, 48)
#: Light gray-green used for frame-to-joint connector cylinders.
CONNECTOR_COLOR: tuple[int, int, int] = (150, 190, 160)
#: Dark gray used for unassigned segmentation faces.
UNASSIGNED_COLOR: tuple[int, int, int] = (70, 70, 74)
#: Neutral gray used for the edit-mode all-vertices point cloud.
EDIT_POINT_COLOR: tuple[int, int, int] = (170, 170, 176)
#: Orange used for the edit-mode selection overlay.
SELECTION_COLOR: tuple[int, int, int] = (240, 148, 48)

_GOLDEN_RATIO_CONJUGATE = 0.618033988749895


@runtime_checkable
class Layer(Protocol):
    """One toggleable group of scene nodes bound to the shared session."""

    name: str

    def build(self, server: viser.ViserServer, session: StudioSession) -> None:
        """Create this layer's scene nodes for the current session state."""

    def update(self, session: StudioSession, kind: str) -> None:
        """React to a session change of the given kind."""

    def set_visible(self, visible: bool) -> None:
        """Show or hide this layer."""


def part_palette(num_parts: int) -> np.ndarray:
    """Return ``num_parts`` distinct RGB colors from a golden-ratio HSV cycle."""
    colors = np.empty((num_parts, 3), dtype=np.uint8)
    for index in range(num_parts):
        hue = (index * _GOLDEN_RATIO_CONJUGATE) % 1.0
        red, green, blue = colorsys.hsv_to_rgb(hue, 0.65, 0.95)
        colors[index] = (int(red * 255), int(green * 255), int(blue * 255))
    return colors


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Detach a tensor to a contiguous CPU ``float32`` numpy array."""
    return tensor.detach().cpu().to(torch.float32).numpy()


def wxyz(quaternion_scalar_last: torch.Tensor) -> np.ndarray:
    """Convert a scalar-last quaternion ``(4,)`` to viser scalar-first order."""
    x, y, z, w = quaternion_scalar_last.detach().cpu().tolist()
    return np.array([w, x, y, z], dtype=np.float64)


def bone_transform(parent: torch.Tensor, child: torch.Tensor) -> tuple[np.ndarray, np.ndarray, float]:
    """Return ``(position, wxyz, height)`` placing a +Z cylinder on a bone.

    The cylinder spans from the ``parent`` to the ``child`` world position: its
    center sits at the midpoint, its local ``+Z`` axis aligns to the bone
    direction, and its height equals the bone length.
    """
    vector = (child - parent).detach().cpu()
    height = float(vector.norm())
    midpoint = ((parent + child) * 0.5).detach().cpu().numpy().astype(np.float64)
    if height < 1e-9:
        return midpoint, np.array([1.0, 0.0, 0.0, 0.0]), 0.0
    direction = vector / height
    axis_z = torch.tensor([0.0, 0.0, 1.0])
    cross = torch.linalg.cross(axis_z, direction)
    cross_norm = float(cross.norm())
    dot = float(torch.dot(axis_z, direction))
    if cross_norm < 1e-9:
        quaternion = np.array([1.0, 0.0, 0.0, 0.0]) if dot > 0 else np.array([0.0, 1.0, 0.0, 0.0])
        return midpoint, quaternion, height
    axis = (cross / cross_norm).tolist()
    angle = math.acos(max(-1.0, min(1.0, dot)))
    half = angle / 2.0
    sin_half = math.sin(half)
    quaternion = np.array(
        [math.cos(half), axis[0] * sin_half, axis[1] * sin_half, axis[2] * sin_half],
        dtype=np.float64,
    )
    return midpoint, quaternion, height


__all__ = [
    "BONE_COLOR",
    "CONNECTOR_COLOR",
    "EDIT_POINT_COLOR",
    "FRAME_ACTIVE_COLOR",
    "FRAME_COLOR",
    "JOINT_COLOR",
    "Layer",
    "MESH_COLOR",
    "SELECTED_JOINT_COLOR",
    "SELECTION_COLOR",
    "UNASSIGNED_COLOR",
    "bone_transform",
    "part_palette",
    "to_numpy",
    "wxyz",
]
