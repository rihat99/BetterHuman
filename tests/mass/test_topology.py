"""Part topologies: closed parts from authored faces or from a vertex labelling, checked when they are built."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
from better_robot.exceptions import ArgumentError
from better_robot.spatial import inertia_of_mesh

from better_human.mass import PartTopology
from tests.support.synthetic import flame_spec


def _uv_sphere(rings: int = 8, segments: int = 12) -> tuple[torch.Tensor, torch.Tensor]:
    """A closed, outward-wound UV sphere: vertices ``(V, 3)`` float64, faces ``(F, 3)`` long."""
    points = [(0.0, 0.0, 1.0)]
    for ring in range(1, rings):
        polar = math.pi * ring / rings
        for segment in range(segments):
            azimuth = 2 * math.pi * segment / segments
            points.append((math.sin(polar) * math.cos(azimuth), math.sin(polar) * math.sin(azimuth), math.cos(polar)))
    points.append((0.0, 0.0, -1.0))
    south = len(points) - 1

    def ring_vertex(ring: int, segment: int) -> int:
        return 1 + (ring - 1) * segments + segment % segments

    faces = [(0, ring_vertex(1, s), ring_vertex(1, s + 1)) for s in range(segments)]
    for ring in range(1, rings - 1):
        for s in range(segments):
            a, b = ring_vertex(ring, s), ring_vertex(ring, s + 1)
            c, d = ring_vertex(ring + 1, s), ring_vertex(ring + 1, s + 1)
            faces += [(a, c, d), (a, d, b)]
    faces += [(south, ring_vertex(rings - 1, s + 1), ring_vertex(rings - 1, s)) for s in range(segments)]
    return torch.tensor(points, dtype=torch.float64), torch.tensor(faces)


def _volume(vertices: torch.Tensor, faces: torch.Tensor) -> float:
    return float(inertia_of_mesh(vertices, faces, density=1.0)[0])


def test_parts_meeting_at_junctions_close_with_shared_caps() -> None:
    vertices, faces = _uv_sphere()
    # A top cap bordering two halves: each loop meets two neighbours, three parts meet at two junctions.
    part_ids = torch.where(vertices[:, 2] > 0.4, 0, torch.where(vertices[:, 0] >= 0, 1, 2))
    topology = PartTopology.from_part_ids(faces, part_ids, num_parts=3)
    parts = [_volume(part_vertices, part_faces) for part_vertices, part_faces in topology.meshes(vertices)]
    assert min(parts) > 0
    assert sum(parts) == pytest.approx(_volume(vertices, faces), rel=1e-12)


@pytest.mark.parametrize(
    ("faces", "match"),
    [
        ([[0, 1, 2], [0, 1, 3]], "not a consistently wound manifold surface"),
        ([[0, 0, 1], [0, 1, 2]], "not a consistently wound manifold surface"),
    ],
    ids=["edge twice in one direction", "degenerate face"],
)
def test_a_part_that_cannot_be_closed_is_an_argument_error(faces: list[list[int]], match: str) -> None:
    with pytest.raises(ArgumentError, match=rf"part 0's faces are {match}"):
        PartTopology.from_part_ids(torch.tensor(faces), torch.zeros(4, dtype=torch.long), num_parts=1)


def test_a_random_mesh_raises_instead_of_walking_forever(tmp_path: Path) -> None:
    spec = flame_spec(tmp_path)  # random faces: no consistently wound surface
    with pytest.raises(ArgumentError, match=r"part \d+"):
        PartTopology.from_part_ids(spec.faces, spec.parts.ids, num_parts=spec.parts.count)


def test_an_authored_part_indexing_past_the_joints_is_an_argument_error() -> None:
    vertices, joints = torch.eye(4, 3, dtype=torch.float64), torch.zeros(1, 3, dtype=torch.float64)
    faces = {0: torch.tensor([[0, 1, 2], [0, 2, 4], [0, 4, 1], [1, 4, 2]]), 1: torch.tensor([[0, 1, 5]])}
    with pytest.raises(ArgumentError, match=r"closed part 1 indexes outside \[0, 5\)"):
        PartTopology.from_closed_faces(faces, reference_vertices=vertices, reference_joints=joints)


def test_an_authored_part_that_is_open_is_an_argument_error() -> None:
    vertices, joints = torch.eye(4, 3, dtype=torch.float64), torch.zeros(1, 3, dtype=torch.float64)
    tetrahedron = torch.tensor([[0, 1, 2], [0, 2, 4], [0, 4, 1], [1, 4, 2]])
    with pytest.raises(ArgumentError, match="closed part 1 is not a closed, consistently wound mesh"):
        PartTopology.from_closed_faces(
            {0: tetrahedron, 1: tetrahedron[:3]}, reference_vertices=vertices, reference_joints=joints
        )
