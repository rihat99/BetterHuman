"""Ray, sphere-brush and rectangle-marquee selection math for HumanStudio.

Pure torch; no viser import.  :func:`vertices_in_rect` reproduces the screen
mapping that viser 1.0.26 reports for ``rect-select`` pointer events, so a
marquee drawn in the browser selects the same vertices here.

viser conventions (read-only, from the installed package):

* ``ScenePointerEvent.screen_pos`` are OpenCV image coordinates in ``[0, 1]``
  with ``(0, 0)`` at the top-left and ``(1, 1)`` at the bottom-right
  (``viser/_scene_handles.py:49-52``); a rect event carries the min- and
  max-corners.
* ``CameraHandle.wxyz`` is a scalar-first quaternion with
  ``R_world_camera = SO3(wxyz)`` (``viser/_viser.py:186-197``).  The camera
  frame follows OpenCV: look ``+Z`` (column 2), up ``-Y`` (column 1), right
  ``+X`` (column 0) (``viser/_viser.py:200-204``).
* ``CameraHandle.fov`` is the vertical field of view in radians and
  ``CameraHandle.aspect`` is width divided by height
  (``viser/_viser.py:270-324``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CameraFrustum:
    """Everything needed to project world points exactly as viser does."""

    #: Camera position in world coordinates ``(3,)``.
    position: torch.Tensor
    #: Scalar-first orientation quaternion ``(4,)`` as viser reports it.
    wxyz: torch.Tensor
    #: Vertical field of view in radians.
    fov: float
    #: Aspect ratio, width divided by height.
    aspect: float


def ray_hit(
    origin: torch.Tensor,
    direction: torch.Tensor,
    vertices: torch.Tensor,
    faces: torch.Tensor,
) -> tuple[torch.Tensor, int] | None:
    """Return the nearest mesh intersection ahead of ``origin`` or ``None``.

    A batched, double-sided Möller-Trumbore test over every triangle keeps the
    hit with the smallest positive ray parameter.  Triangles parallel to the
    ray, off the triangle, or behind the origin never win.

    :param origin: Ray origin ``(3,)``.
    :param direction: Ray direction ``(3,)`` (need not be normalized).
    :param vertices: Mesh vertices ``(V, 3)``.
    :param faces: Triangle vertex indices ``(F, 3)``.
    :returns: ``(hit_point (3,), face_index)`` or ``None`` on a miss.
    """
    v0 = vertices[faces[:, 0]]
    edge1 = vertices[faces[:, 1]] - v0
    edge2 = vertices[faces[:, 2]] - v0
    ray = direction.expand_as(edge2)
    pvec = torch.linalg.cross(ray, edge2, dim=-1)
    det = (edge1 * pvec).sum(-1)

    parallel = det.abs() < 1e-8
    inv_det = 1.0 / torch.where(parallel, torch.ones_like(det), det)
    tvec = origin - v0
    u = (tvec * pvec).sum(-1) * inv_det
    qvec = torch.linalg.cross(tvec, edge1, dim=-1)
    v = (ray * qvec).sum(-1) * inv_det
    t = (edge2 * qvec).sum(-1) * inv_det

    tol = 1e-6
    valid = ~parallel & (u >= -tol) & (v >= -tol) & (u + v <= 1.0 + tol) & (t > 1e-8)
    if not bool(valid.any()):
        return None
    t_valid = torch.where(valid, t, t.new_full((), float("inf")))
    face_index = int(torch.argmin(t_valid))
    hit_point = origin + t[face_index] * direction
    return hit_point, face_index


def vertices_in_sphere(vertices: torch.Tensor, center: torch.Tensor, radius: float) -> torch.Tensor:
    """Return indices of vertices within ``radius`` of ``center`` ``(3,)``."""
    distance_squared = ((vertices - center) ** 2).sum(-1)
    return torch.nonzero(distance_squared <= radius * radius, as_tuple=False).flatten()


def _project(vertices: torch.Tensor, camera: CameraFrustum) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(screen (V, 2), depth (V,))`` for viser's screen convention."""
    rotation = _rotation_from_wxyz(camera.wxyz)
    relative = vertices - camera.position
    camera_space = relative @ rotation  # rotation transpose applied per row.
    depth = camera_space[..., 2]
    safe_depth = torch.where(depth > 0, depth, torch.ones_like(depth))
    tan_half = math.tan(camera.fov / 2.0)
    ndc_x = camera_space[..., 0] / safe_depth / (tan_half * camera.aspect)
    ndc_y = camera_space[..., 1] / safe_depth / tan_half
    screen = torch.stack((0.5 + 0.5 * ndc_x, 0.5 + 0.5 * ndc_y), dim=-1)
    return screen, depth


def vertices_in_rect(
    vertices: torch.Tensor,
    camera: CameraFrustum,
    rect_min: torch.Tensor,
    rect_max: torch.Tensor,
) -> torch.Tensor:
    """Return indices of vertices inside the screen-space marquee.

    ``rect_min`` / ``rect_max`` are viser ``screen_pos`` corners in ``[0, 1]``
    with a top-left origin.  Vertices behind the camera are excluded.
    """
    lower = torch.minimum(rect_min, rect_max)
    upper = torch.maximum(rect_min, rect_max)
    screen, depth = _project(vertices, camera)
    inside = (
        (depth > 0)
        & (screen[..., 0] >= lower[0])
        & (screen[..., 0] <= upper[0])
        & (screen[..., 1] >= lower[1])
        & (screen[..., 1] <= upper[1])
    )
    return torch.nonzero(inside, as_tuple=False).flatten()


def _rotation_from_wxyz(wxyz: torch.Tensor) -> torch.Tensor:
    """Convert a scalar-first unit quaternion ``(4,)`` to ``R_world_camera``."""
    quaternion = wxyz / wxyz.norm().clamp(min=1e-8)
    w, x, y, z = quaternion.unbind(-1)
    return torch.stack(
        (
            torch.stack((1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y))),
            torch.stack((2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x))),
            torch.stack((2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y))),
        )
    )


__all__ = ["CameraFrustum", "ray_hit", "vertices_in_rect", "vertices_in_sphere"]
