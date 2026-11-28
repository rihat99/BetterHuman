"""``core.vertex_normals`` against normals worked by hand (a tetrahedron) and a sphere's analytic normals (the
icosahedron, whose vertex normals point radially by symmetry); float64 to 1e-12, float32 to 1e-6."""

from __future__ import annotations

import math

import pytest
import torch

from better_human.core import vertex_normals

#: The unit-corner tetrahedron, faces wound counter-clockwise seen from outside: three axis faces of area 1/2 with
#: normals -x, -y, -z and the slanted face of area sqrt(3)/2 with normal (1, 1, 1)/sqrt(3).
TETRAHEDRON = (
    ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    ((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)),
)


def _icosahedron(dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """The 12 vertices ``(0, ±1, ±phi)`` and their cyclic permutations, and the 20 faces wound outward."""
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    vertices = [(-1, phi, 0), (1, phi, 0), (-1, -phi, 0), (1, -phi, 0), (0, -1, phi), (0, 1, phi)]
    vertices += [(0, -1, -phi), (0, 1, -phi), (phi, 0, -1), (phi, 0, 1), (-phi, 0, -1), (-phi, 0, 1)]
    faces = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2)]
    faces += [(10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11)]
    faces += [(6, 2, 10), (8, 6, 7), (9, 8, 1)]
    return torch.tensor(vertices, dtype=dtype), torch.tensor(faces)


@pytest.mark.parametrize(("dtype", "atol"), [(torch.float64, 1e-12), (torch.float32, 1e-6)])
def test_tetrahedron_normals_are_the_area_weighted_face_normals(dtype: torch.dtype, atol: float) -> None:
    """At the origin the three axis faces give -(1, 1, 1)/sqrt(3); at (1, 0, 0) the -y and -z faces cancel the slanted
    face's y and z, leaving +x (equal face weights would tilt it), and the same for +y and +z."""
    vertices, faces = torch.tensor(TETRAHEDRON[0], dtype=dtype), torch.tensor(TETRAHEDRON[1])
    corner = -1.0 / math.sqrt(3.0)
    expected = torch.tensor([[corner] * 3, [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=dtype)
    torch.testing.assert_close(vertex_normals(vertices, faces), expected, rtol=0.0, atol=atol)


@pytest.mark.parametrize(("dtype", "atol"), [(torch.float64, 1e-12), (torch.float32, 1e-6)])
def test_sphere_normals_point_away_from_the_centre_in_every_batch_entry(dtype: torch.dtype, atol: float) -> None:
    """The icosahedron moved and scaled per batch entry ``(2, 3, 12, 3)``: each normal is its vertex's direction from
    that entry's centre, and a vertex no face uses gets zero."""
    vertices, faces = _icosahedron(dtype)
    generator = torch.Generator().manual_seed(0)
    centres = torch.randn(2, 3, 1, 3, generator=generator, dtype=dtype)
    scales = 0.5 + torch.rand(2, 3, 1, 1, generator=generator, dtype=dtype)
    unused = torch.zeros(2, 3, 1, 3, dtype=dtype)  # a 13th vertex outside every face
    mesh = torch.cat((centres + scales * vertices, unused), dim=-2)
    expected = torch.cat((torch.nn.functional.normalize(vertices, dim=-1).expand(2, 3, 12, 3), unused), dim=-2)
    torch.testing.assert_close(vertex_normals(mesh, faces), expected, rtol=0.0, atol=atol)
