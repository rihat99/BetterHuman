"""Ray, sphere-brush and rectangle-marquee selection math for HumanStudio."""

from __future__ import annotations

import math

import torch

from better_human.viz.studio import picking

# Unit triangle in the z = 0 plane.
TRIANGLE = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
FACES = torch.tensor([[0, 1, 2]])


def _centered_camera() -> picking.CameraFrustum:
    """Camera at +5z looking toward the origin (OpenCV frame, +Y-up world).

    ``wxyz = (0, 1, 0, 0)`` is a 180 degree turn about x, giving
    ``R_world_camera = diag(1, -1, -1)``: look ``-z``, up ``+y``, right ``+x``.
    """
    return picking.CameraFrustum(
        position=torch.tensor([0.0, 0.0, 5.0]),
        wxyz=torch.tensor([0.0, 1.0, 0.0, 0.0]),
        fov=math.pi / 2.0,
        aspect=1.0,
    )


def test_ray_hit_unit_triangle() -> None:
    hit = picking.ray_hit(
        torch.tensor([0.25, 0.25, 1.0]),
        torch.tensor([0.0, 0.0, -1.0]),
        TRIANGLE,
        FACES,
    )
    assert hit is not None
    point, face_index = hit
    assert face_index == 0
    torch.testing.assert_close(point, torch.tensor([0.25, 0.25, 0.0]), atol=1e-6, rtol=0)


def test_ray_hit_miss_off_triangle() -> None:
    assert picking.ray_hit(
        torch.tensor([5.0, 5.0, 1.0]),
        torch.tensor([0.0, 0.0, -1.0]),
        TRIANGLE,
        FACES,
    ) is None


def test_ray_hit_behind_origin_is_miss() -> None:
    # The triangle sits behind the ray, which travels toward -z from z = -1.
    assert picking.ray_hit(
        torch.tensor([0.25, 0.25, -1.0]),
        torch.tensor([0.0, 0.0, -1.0]),
        TRIANGLE,
        FACES,
    ) is None


def test_ray_hit_is_double_sided() -> None:
    # Approaching the back face still registers a hit.
    hit = picking.ray_hit(
        torch.tensor([0.25, 0.25, -1.0]),
        torch.tensor([0.0, 0.0, 1.0]),
        TRIANGLE,
        FACES,
    )
    assert hit is not None
    torch.testing.assert_close(hit[0], torch.tensor([0.25, 0.25, 0.0]), atol=1e-6, rtol=0)


def test_ray_hit_returns_nearest_positive_t() -> None:
    vertices = torch.cat((TRIANGLE, TRIANGLE + torch.tensor([0.0, 0.0, -0.5])))
    faces = torch.tensor([[0, 1, 2], [3, 4, 5]])
    hit = picking.ray_hit(
        torch.tensor([0.25, 0.25, 1.0]),
        torch.tensor([0.0, 0.0, -1.0]),
        vertices,
        faces,
    )
    assert hit is not None
    # The nearer triangle at z = 0 wins over the one at z = -0.5.
    assert hit[1] == 0
    torch.testing.assert_close(hit[0], torch.tensor([0.25, 0.25, 0.0]), atol=1e-6, rtol=0)


def test_vertices_in_sphere() -> None:
    vertices = torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    selected = picking.vertices_in_sphere(vertices, torch.tensor([0.0, 0.0, 0.0]), 1.0)
    assert selected.tolist() == [0, 1, 3]  # index 3 sits exactly on the radius.
    assert picking.vertices_in_sphere(vertices, torch.tensor([9.0, 9.0, 9.0]), 0.1).numel() == 0


def test_vertices_in_rect_centered_selects_origin() -> None:
    camera = _centered_camera()
    vertices = torch.tensor(
        [
            [0.0, 0.0, 0.0],  # projects to screen center.
            [10.0, 0.0, 0.0],  # far to the right, off the marquee.
            [0.0, 10.0, 0.0],  # far up, off the marquee.
        ]
    )
    selected = picking.vertices_in_rect(
        vertices,
        camera,
        torch.tensor([0.4, 0.4]),
        torch.tensor([0.6, 0.6]),
    )
    assert selected.tolist() == [0]


def test_vertices_in_rect_excludes_behind_camera() -> None:
    camera = _centered_camera()
    # (0, 0, 10) is behind the camera at (0, 0, 5); it projects to the center
    # but has negative depth and must be excluded even by a full-screen rect.
    vertices = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 10.0]])
    selected = picking.vertices_in_rect(
        vertices,
        camera,
        torch.tensor([0.0, 0.0]),
        torch.tensor([1.0, 1.0]),
    )
    assert selected.tolist() == [0]


def test_vertices_in_rect_reordered_corners() -> None:
    camera = _centered_camera()
    vertices = torch.tensor([[0.0, 0.0, 0.0]])
    # Corners passed max-first still describe the same marquee.
    selected = picking.vertices_in_rect(
        vertices,
        camera,
        torch.tensor([0.6, 0.6]),
        torch.tensor([0.4, 0.4]),
    )
    assert selected.tolist() == [0]
