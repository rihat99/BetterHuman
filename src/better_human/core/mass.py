"""Shape-dependent human mesh segmentation and rigid-body inertias.

Two segmentation sources feed one :class:`PartMeshTopology` and the shared,
differentiable :meth:`better_robot.spatial.Inertia.from_mesh` integrator:

- :func:`build_authored_topology` closes each part with authored joint caps
  from a vetted closed-3D asset (default for SMPL-family models that ship one).
- :func:`build_part_topology` closes each argmax-segmented part with
  reverse-wound centroid caps (fallback for models without an asset).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from better_robot.lie import se3
from better_robot.spatial import Inertia
from better_robot.spatial.mesh import orient_faces_by_component


@dataclass(frozen=True)
class PartMeshTopology:
    """Static compact topology for one closed mesh per body part.

    Each part gathers ``vertex_indices`` from the shaped mesh and closes with
    either authored ``cap_joint_indices`` gathered from the rest joints, or the
    reverse-wound centroids of ``boundary_loops``, then applies its local
    ``faces`` table. A given part uses a single capping source; the other tuple
    is empty for that part.
    """

    #: Per-part shaped-mesh vertex indices.
    vertex_indices: tuple[torch.Tensor, ...]
    #: Per-part rest-joint indices used as authored cap vertices.
    cap_joint_indices: tuple[torch.Tensor, ...]
    #: Per-part ordered boundary loops whose centroids cap argmax parts.
    boundary_loops: tuple[tuple[torch.Tensor, ...], ...]
    #: Per-part local triangle tables.
    faces: tuple[torch.Tensor, ...]

    @property
    def num_parts(self) -> int:
        """Return the number of articulated body parts."""
        return len(self.faces)


def segment_vertices(weights_vk: torch.Tensor, indices_vk: torch.Tensor) -> torch.Tensor:
    """Assign each vertex to its strongest-weighted padded joint influence.

    :param weights_vk: nonnegative padded skinning weights ``(V, K)``.
    :param indices_vk: padded native joint indices ``(V, K)``.
    :returns: winning native joint index per vertex ``(V,)`` as ``long``.
    """
    strongest = weights_vk.argmax(dim=-1, keepdim=True)
    return indices_vk.gather(-1, strongest).squeeze(-1).to(dtype=torch.long)


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


def build_part_topology(
    faces: torch.Tensor,
    part_ids: torch.Tensor,
    num_parts: int,
) -> PartMeshTopology:
    """Build compact closed-part topology by majority face assignment.

    A mixed face joins the label shared by at least two of its vertices, else
    its first vertex. Every open boundary loop is capped by a reverse-wound
    triangle fan about its centroid.
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
            raise ValueError(f"part {part} owns no surface faces")
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
    empty_caps = tuple(torch.empty(0, dtype=torch.long, device=faces.device) for _ in range(num_parts))
    return PartMeshTopology(tuple(vertex_tables), empty_caps, tuple(loop_tables), tuple(face_tables))


def build_authored_topology(
    closed_faces: Mapping[int, torch.Tensor],
    num_vertices: int,
    reference_vertices: torch.Tensor,
    reference_joints: torch.Tensor,
) -> PartMeshTopology:
    """Build compact closed-part topology from authored per-part face tables.

    ``closed_faces[part]`` is an ``(F, 3)`` table indexing the concatenation
    ``[shaped_vertices; rest_joints]``, the referenced joints being the caps
    that close each part. Winding is fixed once from the reference (rest)
    geometry with :func:`orient_faces_by_component`, tolerating the shoulder
    seam's open, non-manifold parts, so the returned face tables are static and
    downstream vertex integration stays differentiable.

    :param closed_faces: native joint index to global ``(F, 3)`` face table.
    :param num_vertices: shaped-mesh vertex count ``V`` splitting mesh and caps.
    :param reference_vertices: neutral shaped vertices ``(V, 3)`` for orientation.
    :param reference_joints: neutral rest joints ``(J, 3)`` supplying the caps.
    """
    num_parts = len(closed_faces)
    vertex_tables: list[torch.Tensor] = []
    cap_tables: list[torch.Tensor] = []
    face_tables: list[torch.Tensor] = []
    for part in range(num_parts):
        global_faces = closed_faces[part]
        references = torch.unique(global_faces)
        mesh_indices = references[references < num_vertices]
        cap_indices = references[references >= num_vertices] - num_vertices
        local = global_faces.new_full((num_vertices + num_parts,), -1)
        local[mesh_indices] = torch.arange(mesh_indices.numel(), device=global_faces.device)
        local[num_vertices + cap_indices] = mesh_indices.numel() + torch.arange(
            cap_indices.numel(), device=global_faces.device
        )
        reference_mesh = torch.cat((reference_vertices[mesh_indices], reference_joints[cap_indices]), dim=0)
        vertex_tables.append(mesh_indices)
        cap_tables.append(cap_indices)
        face_tables.append(orient_faces_by_component(reference_mesh, local[global_faces]))
    empty_loops: tuple[tuple[torch.Tensor, ...], ...] = ((),) * num_parts
    return PartMeshTopology(tuple(vertex_tables), tuple(cap_tables), empty_loops, tuple(face_tables))


def meshes_from_topology(
    v_shaped: torch.Tensor,
    topology: PartMeshTopology,
    rest_joints: torch.Tensor | None = None,
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    """Materialize differentiable compact vertices for a static part topology.

    Authored parts append their joint caps from ``rest_joints``; argmax parts
    append the differentiable centroid of each boundary loop.
    """
    meshes: list[tuple[torch.Tensor, torch.Tensor]] = []
    for vertex_indices, cap_indices, loops, faces in zip(
        topology.vertex_indices,
        topology.cap_joint_indices,
        topology.boundary_loops,
        topology.faces,
        strict=True,
    ):
        vertices = v_shaped.index_select(-2, vertex_indices.to(device=v_shaped.device))
        if cap_indices.numel():
            if rest_joints is None:
                raise ValueError("authored joint caps require rest_joints")
            vertices = torch.cat(
                (vertices, rest_joints.index_select(-2, cap_indices.to(device=v_shaped.device))), dim=-2
            )
        if loops:
            centroids = torch.stack(
                tuple(v_shaped.index_select(-2, loop.to(device=v_shaped.device)).mean(dim=-2) for loop in loops),
                dim=-2,
            )
            vertices = torch.cat((vertices, centroids), dim=-2)
        meshes.append((vertices, faces.to(device=v_shaped.device)))
    return tuple(meshes)


def body_inertias_from_shape(
    v_shaped: torch.Tensor,
    rest_joints: torch.Tensor,
    topology: PartMeshTopology,
    density: torch.Tensor,
) -> torch.Tensor:
    """Integrate closed part meshes into BetterRobot packed inertias.

    Returns ``(..., num_parts + 1, 10)``: row zero is the massless BetterRobot
    universe and native part ``j`` occupies row ``j + 1``, each physical inertia
    expressed in its rest-joint frame. Compact per-part vertices center each
    integration at the part mean, which is exact for the open seam parts too.
    ``density`` is a scalar or ``(num_parts,)`` tensor already validated by
    ``Body``.
    """
    per_part = density.broadcast_to((topology.num_parts,))

    if rest_joints.shape[-1] not in (3, 7):
        raise ValueError(f"rest_joints must have shape (..., J, 3) or (..., J, 7); got {tuple(rest_joints.shape)}")
    cap_positions = rest_joints[..., :3] if rest_joints.shape[-1] == 7 else rest_joints
    packed_parts: list[torch.Tensor] = []
    for part, (vertices, faces) in enumerate(meshes_from_topology(v_shaped, topology, cap_positions)):
        rest_joint = rest_joints[..., part, :]
        if rest_joint.shape[-1] == 3:
            local_vertices = vertices - rest_joint[..., None, :]
        else:
            local_vertices = se3.act(se3.inverse(rest_joint)[..., None, :], vertices)
        packed_parts.append(Inertia.from_mesh(local_vertices, faces, density=per_part[part]).data)
    native = torch.stack(packed_parts, dim=-2)
    universe = native.new_zeros((*native.shape[:-2], 1, 10))
    return torch.cat((universe, native), dim=-2)


__all__ = [
    "PartMeshTopology",
    "body_inertias_from_shape",
    "build_authored_topology",
    "build_part_topology",
    "meshes_from_topology",
    "segment_vertices",
]
