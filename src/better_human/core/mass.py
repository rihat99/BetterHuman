"""Shape-dependent human mesh segmentation and rigid-body inertias."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import torch
from better_robot.spatial import Inertia


@dataclass(frozen=True)
class PartMeshTopology:
    """Static compact topology for closed body-part meshes.

    Each part gathers ``vertex_indices`` from the shaped mesh, appends one
    differentiable centroid per ``boundary_loops`` entry, then applies its
    local ``faces`` table.
    """

    vertex_indices: tuple[torch.Tensor, ...]
    boundary_loops: tuple[tuple[torch.Tensor, ...], ...]
    faces: tuple[torch.Tensor, ...]

    @property
    def num_parts(self) -> int:
        """Return the number of articulated body parts."""
        return len(self.faces)


def segment_vertices(weights_vk: torch.Tensor, indices_vk: torch.Tensor) -> torch.Tensor:
    """Assign each vertex to the joint with its largest skinning influence."""
    if not isinstance(weights_vk, torch.Tensor) or not weights_vk.is_floating_point():
        raise TypeError("weights_vk must be a floating torch.Tensor")
    if not isinstance(indices_vk, torch.Tensor) or indices_vk.dtype not in (torch.int32, torch.int64):
        raise TypeError("indices_vk must be a torch.int32 or torch.int64 tensor")
    if weights_vk.ndim != 2 or weights_vk.shape[1] == 0:
        raise ValueError(f"weights_vk must have non-empty shape (V, K), got {tuple(weights_vk.shape)}")
    if indices_vk.shape != weights_vk.shape:
        raise ValueError(
            f"indices_vk must match weights_vk shape {tuple(weights_vk.shape)}, got {tuple(indices_vk.shape)}"
        )
    if indices_vk.device != weights_vk.device:
        raise ValueError("weights_vk and indices_vk must use the same device")
    if not bool(torch.isfinite(weights_vk).all()) or bool((weights_vk < 0).any()):
        raise ValueError("weights_vk must contain only finite non-negative values")
    if bool((weights_vk.sum(dim=-1) <= 0).any()):
        raise ValueError("every vertex must have a positive total skinning weight")
    if bool((indices_vk < 0).any()):
        raise ValueError("indices_vk must contain non-negative part indices")

    strongest = weights_vk.argmax(dim=-1, keepdim=True)
    return indices_vk.gather(-1, strongest).squeeze(-1).to(dtype=torch.long)


def _majority_face_parts(faces: torch.Tensor, part_ids: torch.Tensor) -> torch.Tensor:
    labels = part_ids[faces]
    first, second, third = labels.unbind(dim=-1)
    return torch.where(
        (first == second) | (first == third),
        first,
        torch.where(second == third, second, first),
    )


def _oriented_boundary_loops(part_faces: list[tuple[int, int, int]]) -> tuple[tuple[int, ...], ...]:
    edge_occurrences: dict[tuple[int, int], list[tuple[int, int, int, int]]] = {}
    for face_index, (a, b, c) in enumerate(part_faces):
        for edge_index, (start, end) in enumerate(((a, b), (b, c), (c, a))):
            key = (min(start, end), max(start, end))
            edge_occurrences.setdefault(key, []).append((face_index, edge_index, start, end))

    non_manifold = sorted(edge for edge, occurrences in edge_occurrences.items() if len(occurrences) > 2)
    if non_manifold:
        raise ValueError(f"part surface is non-manifold at edges {non_manifold[:5]}")
    boundary_edges = {
        (occurrences[0][0], occurrences[0][1])
        for occurrences in edge_occurrences.values()
        if len(occurrences) == 1
    }
    if not boundary_edges:
        return ()

    def next_boundary(face_index: int, edge_index: int) -> tuple[int, int]:
        current = (face_index, (edge_index + 1) % 3)
        visited: set[tuple[int, int]] = set()
        while current not in boundary_edges:
            if current in visited:
                raise ValueError("part boundary traversal cycled without reaching an edge")
            visited.add(current)
            current_face, current_edge = current
            face = part_faces[current_face]
            start = face[current_edge]
            end = face[(current_edge + 1) % 3]
            occurrences = edge_occurrences[(min(start, end), max(start, end))]
            if len(occurrences) != 2:
                raise ValueError("part boundary is open")
            other = next(
                occurrence
                for occurrence in occurrences
                if (occurrence[0], occurrence[1]) != current
            )
            if (other[2], other[3]) != (end, start):
                raise ValueError("part surface has inconsistent face winding")
            current = (other[0], (other[1] + 1) % 3)
        return current

    successors = {edge: next_boundary(*edge) for edge in boundary_edges}
    if set(successors.values()) != boundary_edges:
        raise ValueError("part boundary half-edges do not form closed cycles")

    remaining = set(boundary_edges)
    loops: list[tuple[int, ...]] = []
    while remaining:
        start = min(remaining)
        edge = start
        loop: list[int] = []
        while True:
            if edge not in remaining:
                raise ValueError("part boundary repeats before closing a loop")
            remaining.remove(edge)
            face_index, edge_index = edge
            loop.append(part_faces[face_index][edge_index])
            edge = successors[edge]
            if edge == start:
                break
        if len(loop) < 3:
            raise ValueError("part boundary loop must contain at least three vertices")
        loops.append(tuple(loop))
    return tuple(loops)


def build_part_topology(
    faces: torch.Tensor,
    part_ids: torch.Tensor,
    num_parts: int,
) -> PartMeshTopology:
    """Build compact closed-part topology using majority face assignment.

    A mixed face belongs to the label held by at least two vertices. If all
    three labels differ, the first face vertex wins. Every resulting boundary
    loop is capped by a reverse-wound triangle fan about its centroid.
    """
    if not isinstance(faces, torch.Tensor) or faces.dtype not in (torch.int32, torch.int64):
        raise TypeError("faces must be a torch.int32 or torch.int64 tensor")
    if faces.ndim != 2 or faces.shape[-1] != 3 or faces.shape[0] == 0:
        raise ValueError(f"faces must have non-empty shape (F, 3), got {tuple(faces.shape)}")
    if not isinstance(part_ids, torch.Tensor) or part_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("part_ids must be a torch.int32 or torch.int64 tensor")
    if part_ids.ndim != 1 or part_ids.shape[0] == 0:
        raise ValueError(f"part_ids must have non-empty shape (V,), got {tuple(part_ids.shape)}")
    if faces.device != part_ids.device:
        raise ValueError("faces and part_ids must use the same device")
    if not isinstance(num_parts, int) or isinstance(num_parts, bool) or num_parts < 1:
        raise ValueError("num_parts must be a positive integer")
    if bool((faces < 0).any()) or bool((faces >= part_ids.shape[0]).any()):
        raise ValueError("faces contains an index outside part_ids")
    if bool((part_ids < 0).any()) or bool((part_ids >= num_parts).any()):
        raise ValueError(f"part_ids must be in [0, {num_parts})")

    face_parts = _majority_face_parts(faces, part_ids)
    faces_cpu = faces.detach().cpu().tolist()
    face_parts_cpu = face_parts.detach().cpu().tolist()
    vertex_tables: list[torch.Tensor] = []
    loop_tables: list[tuple[torch.Tensor, ...]] = []
    face_tables: list[torch.Tensor] = []
    for part in range(num_parts):
        surface_faces = [tuple(face) for face, owner in zip(faces_cpu, face_parts_cpu, strict=True) if owner == part]
        if not surface_faces:
            raise ValueError(f"part {part} owns no surface faces")
        loops = _oriented_boundary_loops(surface_faces)
        used_vertices = tuple(sorted({vertex for face in surface_faces for vertex in face}))
        local_index = {vertex: index for index, vertex in enumerate(used_vertices)}
        local_faces = [[local_index[vertex] for vertex in face] for face in surface_faces]
        for loop_index, loop in enumerate(loops):
            center = len(used_vertices) + loop_index
            for start, end in zip(loop, (*loop[1:], loop[0]), strict=True):
                local_faces.append([local_index[end], local_index[start], center])

        vertex_tables.append(torch.tensor(used_vertices, dtype=torch.long, device=faces.device))
        loop_tables.append(
            tuple(torch.tensor(loop, dtype=torch.long, device=faces.device) for loop in loops)
        )
        face_tables.append(torch.tensor(local_faces, dtype=torch.long, device=faces.device))
    return PartMeshTopology(tuple(vertex_tables), tuple(loop_tables), tuple(face_tables))


def meshes_from_topology(
    v_shaped: torch.Tensor,
    topology: PartMeshTopology,
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    """Materialize differentiable compact vertices for static part topology."""
    if not isinstance(v_shaped, torch.Tensor) or not v_shaped.is_floating_point():
        raise TypeError("v_shaped must be a floating torch.Tensor")
    if v_shaped.ndim < 2 or v_shaped.shape[-1] != 3:
        raise ValueError(f"v_shaped must have shape (..., V, 3), got {tuple(v_shaped.shape)}")
    if topology.num_parts == 0:
        raise ValueError("topology must contain at least one part")

    meshes: list[tuple[torch.Tensor, torch.Tensor]] = []
    for vertex_indices, loops, faces in zip(
        topology.vertex_indices,
        topology.boundary_loops,
        topology.faces,
        strict=True,
    ):
        indices = vertex_indices.to(device=v_shaped.device)
        if indices.numel() == 0:
            raise ValueError("part topology contains an empty shaped-vertex table")
        vertices = v_shaped.index_select(-2, indices)
        if loops:
            centroids = torch.stack(
                tuple(
                    v_shaped.index_select(-2, loop.to(device=v_shaped.device)).mean(dim=-2)
                    for loop in loops
                ),
                dim=-2,
            )
            vertices = torch.cat((vertices, centroids), dim=-2)
        meshes.append((vertices, faces.to(device=v_shaped.device)))
    return tuple(meshes)


def part_meshes(
    v_shaped: torch.Tensor,
    faces: torch.Tensor,
    part_ids: torch.Tensor,
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    """Partition and close one shaped mesh using per-vertex part labels."""
    if part_ids.numel() == 0:
        raise ValueError("part_ids must not be empty")
    num_parts = int(part_ids.max().item()) + 1
    return meshes_from_topology(v_shaped, build_part_topology(faces, part_ids, num_parts))


def body_inertias_from_shape(
    v_shaped: torch.Tensor,
    rest_joints: torch.Tensor,
    topology: PartMeshTopology,
    density: Real | torch.Tensor = 1000.0,
) -> torch.Tensor:
    """Integrate closed part meshes and return BetterRobot packed inertias.

    The returned shape is ``(..., num_parts + 1, 10)``. Row zero is the
    massless BetterRobot universe; native part ``j`` occupies row ``j + 1``.
    Each physical inertia is expressed in its rest-joint frame.
    """
    if not isinstance(rest_joints, torch.Tensor) or not rest_joints.is_floating_point():
        raise TypeError("rest_joints must be a floating torch.Tensor")
    expected_joint_shape = (*v_shaped.shape[:-2], topology.num_parts, 3)
    if rest_joints.shape != expected_joint_shape:
        raise ValueError(f"rest_joints must have shape {expected_joint_shape}, got {tuple(rest_joints.shape)}")
    if rest_joints.dtype != v_shaped.dtype or rest_joints.device != v_shaped.device:
        raise ValueError("v_shaped and rest_joints must use the same dtype and device")

    if isinstance(density, Real) and not isinstance(density, bool):
        density_by_part: Real | torch.Tensor = density
    elif isinstance(density, torch.Tensor) and density.is_floating_point():
        if density.shape not in ((), (topology.num_parts,)):
            raise ValueError(
                f"density tensor must be scalar or have shape ({topology.num_parts},), "
                f"got {tuple(density.shape)}"
            )
        if density.dtype != v_shaped.dtype or density.device != v_shaped.device:
            raise ValueError("per-part density must match v_shaped dtype and device")
        density_by_part = density
    else:
        raise TypeError("density must be a positive real number or floating per-part tensor")

    packed_parts: list[torch.Tensor] = []
    for part, (vertices, faces) in enumerate(meshes_from_topology(v_shaped, topology)):
        local_vertices = vertices - rest_joints[..., part, None, :]
        part_density = (
            density_by_part[part]
            if isinstance(density_by_part, torch.Tensor) and density_by_part.ndim == 1
            else density_by_part
        )
        packed_parts.append(Inertia.from_mesh(local_vertices, faces, density=part_density).data)
    native = torch.stack(packed_parts, dim=-2)
    universe = native.new_zeros((*native.shape[:-2], 1, 10))
    return torch.cat((universe, native), dim=-2)


__all__ = [
    "PartMeshTopology",
    "body_inertias_from_shape",
    "build_part_topology",
    "meshes_from_topology",
    "part_meshes",
    "segment_vertices",
]
