"""Closed part meshes: which mesh vertices and caps make each part, and their triangles.

Two segmentations close the parts into one :class:`PartTopology`:

- :meth:`PartTopology.from_closed_faces` reads authored per-part faces whose caps are rest joints (the SMPL family's
  shipped segmentation).
- :meth:`PartTopology.from_part_ids` closes each part of a vertex labelling with reverse-wound centroid caps (every
  family without an authored segmentation).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from better_robot.exceptions import ArgumentError
from better_robot.spatial import orient_faces_by_component


@dataclass(frozen=True)
class PartTopology:
    """Static topology of one closed mesh per part; built once per ``HumanModel``, applied to every shape.

    Part ``p`` gathers the mesh vertices ``vertex_ids[p]``, then closes with either the joints ``cap_joint_ids[p]``
    (authored) or the centroid of each loop in ``boundary_loops[p]`` (from part ids), and applies ``faces[p]``, which
    index that gathered list. A part uses one capping source; the other is empty for it.
    """

    vertex_ids: tuple[torch.Tensor, ...]
    cap_joint_ids: tuple[torch.Tensor, ...]
    boundary_loops: tuple[tuple[torch.Tensor, ...], ...]
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
        """
        num_vertices, num_joints = reference_vertices.shape[0], reference_joints.shape[0]
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
        no_loops: tuple[tuple[torch.Tensor, ...], ...] = ((),) * len(face_tables)
        return cls(tuple(vertex_tables), tuple(cap_tables), no_loops, tuple(face_tables))

    @classmethod
    def from_part_ids(cls, faces: torch.Tensor, part_ids: torch.Tensor, *, num_parts: int) -> PartTopology:
        """The topology of a vertex labelling ``part_ids (V,)`` on the mesh ``faces (F, 3)``.

        A face joins the part held by at least two of its vertices, else its first vertex's. Every open boundary
        loop is capped by a reverse-wound triangle fan about its centroid. Raises
        :class:`~better_robot.exceptions.ArgumentError` for a part that owns no face.
        """
        face_parts = _majority_face_parts(faces, part_ids)
        faces_cpu = faces.detach().cpu().tolist()
        face_parts_cpu = face_parts.detach().cpu().tolist()
        vertex_tables: list[torch.Tensor] = []
        loop_tables: list[tuple[torch.Tensor, ...]] = []
        face_tables: list[torch.Tensor] = []
        for part in range(num_parts):
            surface = [tuple(face) for face, owner in zip(faces_cpu, face_parts_cpu, strict=True) if owner == part]
            if not surface:
                raise ArgumentError(f"part {part} owns no surface faces; every part needs at least one face")
            loops = _boundary_loops(surface)
            used_vertices = tuple(sorted({vertex for face in surface for vertex in face}))
            local_index = {vertex: index for index, vertex in enumerate(used_vertices)}
            local_faces = [[local_index[vertex] for vertex in face] for face in surface]
            for loop_index, loop in enumerate(loops):
                center = len(used_vertices) + loop_index
                for start, end in zip(loop, (*loop[1:], loop[0]), strict=True):
                    local_faces.append([local_index[end], local_index[start], center])
            vertex_tables.append(torch.tensor(used_vertices, dtype=torch.long, device=faces.device))
            loop_tables.append(tuple(torch.tensor(loop, dtype=torch.long, device=faces.device) for loop in loops))
            face_tables.append(torch.tensor(local_faces, dtype=torch.long, device=faces.device))
        no_caps = tuple(torch.empty(0, dtype=torch.long, device=faces.device) for _ in range(num_parts))
        return cls(tuple(vertex_tables), no_caps, tuple(loop_tables), tuple(face_tables))

    def meshes(
        self, vertices: torch.Tensor, joints: torch.Tensor | None = None
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        """``(vertices (..., n_p, 3), faces (f_p, 3))`` of every closed part of the mesh ``vertices (..., V, 3)``.

        Authored parts append their caps from ``joints (..., J, 3)``; parts from part ids append the differentiable
        centroid of each boundary loop. Raises :class:`~better_robot.exceptions.ArgumentError` when authored caps
        need ``joints`` and none are given.
        """
        meshes: list[tuple[torch.Tensor, torch.Tensor]] = []
        for vertex_ids, cap_ids, loops, faces in zip(
            self.vertex_ids, self.cap_joint_ids, self.boundary_loops, self.faces, strict=True
        ):
            part = vertices.index_select(-2, vertex_ids.to(device=vertices.device))
            if cap_ids.numel():
                if joints is None:
                    raise ArgumentError("PartTopology.meshes: authored joint caps need joints=(..., J, 3)")
                part = torch.cat((part, joints.index_select(-2, cap_ids.to(device=vertices.device))), dim=-2)
            if loops:
                centroids = torch.stack(
                    tuple(vertices.index_select(-2, loop.to(device=vertices.device)).mean(dim=-2) for loop in loops),
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


def _boundary_loops(part_faces: list[tuple[int, int, int]]) -> tuple[tuple[int, ...], ...]:
    """Order a part's boundary half-edges into oriented loops.

    A boundary half-edge lies on an undirected edge incident to one face. Its
    successor is found by walking the interior fan around the shared vertex to
    the next boundary edge, which disambiguates pinch vertices where two loops
    meet. The native meshes are consistently wound and vetted, so the walk lands
    on a boundary edge and the successors partition into simple cycles; no
    per-spawn manifold re-validation is done.
    """
    edge_occurrences: dict[tuple[int, int], list[tuple[int, int, int, int]]] = {}
    for face_index, (a, b, c) in enumerate(part_faces):
        for edge_index, (start, end) in enumerate(((a, b), (b, c), (c, a))):
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
        loop: list[int] = []
        while edge in remaining:
            remaining.discard(edge)
            face_index, edge_index = edge
            loop.append(part_faces[face_index][edge_index])
            edge = successors[edge]
        loops.append(tuple(loop))
    return tuple(loops)
