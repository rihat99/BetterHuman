"""Closed part meshes: which mesh vertices and caps make each part, and their triangles.

Two segmentations close the parts into one :class:`PartTopology`:

- :meth:`PartTopology.from_closed_faces` reads authored per-part faces whose caps are rest joints (the SMPL family's
  shipped segmentation).
- :meth:`PartTopology.from_part_ids` closes each part of a vertex labelling with reverse-wound centroid caps that
  neighbouring parts share (every family without an authored segmentation).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import final

import torch
from better_robot.exceptions import ArgumentError
from better_robot.spatial import orient_faces_by_component


@final
@dataclass(frozen=True)
class PartTopology:
    """Static topology of one closed mesh per part; built once per ``HumanModel``, applied to every shape.

    Part ``p`` gathers the mesh vertices ``vertex_ids[p]``, then closes with either the joints ``cap_joint_ids[p]``
    (authored) or the centroid of each vertex set in ``cap_centroid_ids[p]`` (from part ids), and applies ``faces[p]``,
    which index that gathered list. A part uses one capping source; the other is empty for it.
    """

    vertex_ids: tuple[torch.Tensor, ...]
    cap_joint_ids: tuple[torch.Tensor, ...]
    cap_centroid_ids: tuple[tuple[torch.Tensor, ...], ...]
    faces: tuple[torch.Tensor, ...]

    @property
    def num_parts(self) -> int:
        return len(self.faces)

    @classmethod
    def from_closed_faces(
        cls,
        closed_faces: Mapping[int, torch.Tensor],
        *,
        reference_vertices: torch.Tensor,
        reference_joints: torch.Tensor,
    ) -> PartTopology:
        """The topology of authored closed parts: ``closed_faces[p] (F_p, 3)`` indexes ``[vertices; joints]``.

        An index ``>= V`` is a joint capping the part. The winding is fixed once on the reference geometry with
        :func:`~better_robot.spatial.orient_faces_by_component`, which tolerates the open, non-manifold shoulder seam,
        so the face tables are static and the integration downstream stays differentiable.

        :param closed_faces: part ``0..P-1`` -> its ``(F_p, 3)`` long faces.
        :param reference_vertices: ``(V, 3)`` rest mesh used to orient the faces.
        :param reference_joints: ``(J, 3)`` rest joints, the caps.
        :raises ~better_robot.exceptions.ArgumentError: for a part with an index outside ``[0, V + J)``.
        """
        num_vertices, num_joints = reference_vertices.shape[0], reference_joints.shape[0]
        for part, part_faces in closed_faces.items():
            if (
                part_faces.numel()
                and not 0 <= int(part_faces.min()) <= int(part_faces.max()) < num_vertices + num_joints
            ):
                raise ArgumentError(
                    f"closed part {part} indexes outside [0, {num_vertices + num_joints}) = [vertices; joints]; "
                    "pass faces over this family's rest vertices and joints"
                )
        vertex_tables: list[torch.Tensor] = []
        cap_tables: list[torch.Tensor] = []
        face_tables: list[torch.Tensor] = []
        for part in range(len(closed_faces)):
            global_faces = closed_faces[part]
            references = torch.unique(global_faces)
            mesh_ids = references[references < num_vertices]
            cap_ids = references[references >= num_vertices] - num_vertices
            local = global_faces.new_full((num_vertices + num_joints,), -1)
            local[mesh_ids] = torch.arange(mesh_ids.numel(), device=global_faces.device)
            local[num_vertices + cap_ids] = mesh_ids.numel() + torch.arange(cap_ids.numel(), device=global_faces.device)
            reference_mesh = torch.cat((reference_vertices[mesh_ids], reference_joints[cap_ids]), dim=0)
            vertex_tables.append(mesh_ids)
            cap_tables.append(cap_ids)
            face_tables.append(orient_faces_by_component(reference_mesh, local[global_faces]))
        no_centroids: tuple[tuple[torch.Tensor, ...], ...] = ((),) * len(face_tables)
        return cls(tuple(vertex_tables), tuple(cap_tables), no_centroids, tuple(face_tables))

    @classmethod
    def from_part_ids(cls, faces: torch.Tensor, part_ids: torch.Tensor, *, num_parts: int) -> PartTopology:
        """The topology of a vertex labelling ``part_ids (V,)`` on the mesh ``faces (F, 3)``.

        A face joins the part held by at least two of its vertices, else its first vertex's. A part's open boundary
        splits into simple loops, and each loop into runs, one per neighbouring part (or mesh hole) it borders. A loop
        with one run is capped by a reverse-wound fan about its centroid. Otherwise each run is capped by a fan about
        its own centroid over the run and the chord between its end junctions, and three or more junctions by a fan
        about their centroid. A neighbour computes the same centroids from the same vertices, so the caps two parts
        share coincide with opposite winding and cancel in the sum of the part volumes; only a junction polygon that
        no neighbour shares leaves a residual.

        Raises :class:`~better_robot.exceptions.ArgumentError` for a part that owns no face, whose faces are not a
        consistently wound manifold surface (then its boundary cannot be walked), or that does not close into a
        consistently wound mesh.
        """
        face_parts = _majority_face_parts(faces, part_ids)
        faces_cpu = faces.detach().cpu().tolist()
        face_parts_cpu = face_parts.detach().cpu().tolist()
        edge_parts = {
            edge: owner for face, owner in zip(faces_cpu, face_parts_cpu, strict=True) for edge in _edges(face)
        }
        vertex_tables: list[torch.Tensor] = []
        centroid_tables: list[tuple[torch.Tensor, ...]] = []
        face_tables: list[torch.Tensor] = []
        for part in range(num_parts):
            surface = [tuple(face) for face, owner in zip(faces_cpu, face_parts_cpu, strict=True) if owner == part]
            if not surface:
                raise ArgumentError(f"part {part} owns no surface faces; every part needs at least one face")
            edge_counts = Counter(edge for face in surface for edge in _edges(face))
            if any(len(set(face)) < 3 for face in surface) or max(edge_counts.values()) > 1:
                raise ArgumentError(
                    f"part {part}'s faces are not a consistently wound manifold surface (a degenerate face, or an edge "
                    "used twice in one direction), so its boundary cannot be closed; pass a consistently wound mesh"
                )
            used_vertices = tuple(sorted({vertex for face in surface for vertex in face}))
            local_index = {vertex: index for index, vertex in enumerate(used_vertices)}
            local_faces = [[local_index[vertex] for vertex in face] for face in surface]
            centroids: list[tuple[int, ...]] = []

            def cap(polygon: Sequence[int], centroid_of: Sequence[int]) -> None:
                # Sorted ids: a neighbour averaging the same vertices in another order gets the identical point.
                centroids.append(tuple(sorted(set(centroid_of))))
                center = len(used_vertices) + len(centroids) - 1
                for start, end in zip(polygon, (*polygon[1:], polygon[0]), strict=True):
                    local_faces.append([local_index[end], local_index[start], center])

            for loop in _boundary_loops(surface):
                neighbours = [edge_parts.get((end, start)) for start, end in zip(loop, (*loop[1:], loop[0]))]
                if len(set(neighbours)) == 1:
                    cap(loop, loop)
                    continue
                junctions = [index for index in range(len(loop)) if neighbours[index] != neighbours[index - 1]]
                ring = loop + loop
                for start, end in zip(junctions, (*junctions[1:], junctions[0] + len(loop)), strict=True):
                    run = ring[start : end + 1]
                    if len(run) > 2:  # a one-edge run is its own chord: the junction cap closes it
                        cap(run, run)  # the run and its chord back from the end junction to the start junction
                if len(junctions) > 2:
                    corners = tuple(loop[index] for index in junctions)
                    cap(corners, corners)
            closed_counts = Counter(edge for face in local_faces for edge in _edges(face))
            if any(count != 1 or closed_counts[(end, start)] != 1 for (start, end), count in closed_counts.items()):
                raise ArgumentError(f"part {part} does not close into a consistently wound mesh")
            vertex_tables.append(torch.tensor(used_vertices, dtype=torch.long, device=faces.device))
            centroid_tables.append(
                tuple(torch.tensor(vertices, dtype=torch.long, device=faces.device) for vertices in centroids)
            )
            face_tables.append(torch.tensor(local_faces, dtype=torch.long, device=faces.device))
        no_caps = tuple(torch.empty(0, dtype=torch.long, device=faces.device) for _ in range(num_parts))
        return cls(tuple(vertex_tables), no_caps, tuple(centroid_tables), tuple(face_tables))

    def meshes(
        self, vertices: torch.Tensor, joints: torch.Tensor | None = None
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        """``(vertices (..., n_p, 3), faces (f_p, 3))`` of every closed part of the mesh ``vertices (..., V, 3)``.

        Authored parts append their caps from ``joints (..., J, 3)``; parts from part ids append the differentiable
        centroid of each of their cap vertex sets. Raises :class:`~better_robot.exceptions.ArgumentError` when authored caps
        need ``joints`` and none are given.
        """
        meshes: list[tuple[torch.Tensor, torch.Tensor]] = []
        for vertex_ids, cap_ids, centroid_ids, faces in zip(
            self.vertex_ids, self.cap_joint_ids, self.cap_centroid_ids, self.faces, strict=True
        ):
            part = vertices.index_select(-2, vertex_ids.to(device=vertices.device))
            if cap_ids.numel():
                if joints is None:
                    raise ArgumentError("PartTopology.meshes: authored joint caps need joints=(..., J, 3)")
                part = torch.cat((part, joints.index_select(-2, cap_ids.to(device=vertices.device))), dim=-2)
            if centroid_ids:
                centroids = torch.stack(
                    tuple(
                        vertices.index_select(-2, ids.to(device=vertices.device)).mean(dim=-2) for ids in centroid_ids
                    ),
                    dim=-2,
                )
                part = torch.cat((part, centroids), dim=-2)
            meshes.append((part, faces.to(device=vertices.device)))
        return tuple(meshes)


def _majority_face_parts(faces: torch.Tensor, part_ids: torch.Tensor) -> torch.Tensor:
    """Label each face with the part held by at least two of its vertices."""
    labels = part_ids[faces]
    first, second, third = labels.unbind(dim=-1)
    return torch.where(
        (first == second) | (first == third),
        first,
        torch.where(second == third, second, first),
    )


def _edges(face: Sequence[int]) -> tuple[tuple[int, int], ...]:
    """The three directed edges of a triangle, in its winding."""
    a, b, c = face
    return (a, b), (b, c), (c, a)


def _boundary_loops(part_faces: list[tuple[int, int, int]]) -> tuple[tuple[int, ...], ...]:
    """A part's boundary as simple oriented vertex loops (edge ``loop[i] -> loop[i + 1]`` is a boundary half-edge).

    A boundary half-edge lies on an undirected edge incident to one face. Its successor is found by walking the
    interior fan around the shared vertex to the next boundary edge. On a consistently wound manifold surface (the
    caller checks it) each such fan is a path that starts at the boundary, so the walk ends; a cycle that passes a
    pinch vertex twice is then split there into simple loops, each capped on its own.
    """
    edge_occurrences: dict[tuple[int, int], list[tuple[int, int, int, int]]] = {}
    for face_index, face in enumerate(part_faces):
        for edge_index, (start, end) in enumerate(_edges(face)):
            key = (min(start, end), max(start, end))
            edge_occurrences.setdefault(key, []).append((face_index, edge_index, start, end))

    boundary_edges = {
        (occurrences[0][0], occurrences[0][1]) for occurrences in edge_occurrences.values() if len(occurrences) == 1
    }
    if not boundary_edges:
        return ()

    def next_boundary(face_index: int, edge_index: int) -> tuple[int, int]:
        current = (face_index, (edge_index + 1) % 3)
        while current not in boundary_edges:
            current_face, current_edge = current
            face = part_faces[current_face]
            start, end = face[current_edge], face[(current_edge + 1) % 3]
            occurrences = edge_occurrences[(min(start, end), max(start, end))]
            other = next(occurrence for occurrence in occurrences if (occurrence[0], occurrence[1]) != current)
            current = (other[0], (other[1] + 1) % 3)
        return current

    successors = {edge: next_boundary(*edge) for edge in boundary_edges}
    remaining = set(boundary_edges)
    loops: list[tuple[int, ...]] = []
    while remaining:
        edge = min(remaining)
        cycle: list[int] = []
        while edge in remaining:
            remaining.discard(edge)
            face_index, edge_index = edge
            cycle.append(part_faces[face_index][edge_index])
            edge = successors[edge]
        loops += _simple_loops(cycle)
    return tuple(loops)


def _simple_loops(cycle: list[int]) -> list[tuple[int, ...]]:
    """Split a closed vertex cycle at every repeated vertex into loops that visit each vertex once."""
    loops: list[tuple[int, ...]] = []
    path: list[int] = []
    position: dict[int, int] = {}
    for vertex in cycle:
        if vertex in position:
            start = position[vertex]
            loops.append(tuple(path[start:]))
            for removed in path[start:]:
                del position[removed]
            del path[start:]
        position[vertex] = len(path)
        path.append(vertex)
    loops.append(tuple(path))
    return loops
