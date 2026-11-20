"""``FamilySpec``: a body family as data — mesh, skin weights, blend groups, corrective, rig, parts, landmarks.

A family loader builds the spec once at the requested dtype and device; nothing in
BetterHuman moves or casts it afterwards. Every field has a type: ``provenance`` is the only free-form
mapping. Every mapping field is a read-only copy (``MappingProxyType``), nested mappings too.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, final

import torch
from better_robot.exceptions import ShapeError, UnknownNameError

if TYPE_CHECKING:
    from .classic import ClassicFormat
    from .rig import Rig


class Corrective(ABC):
    """Pose-dependent vertex offsets computed from the features the family's rig gives."""

    @abstractmethod
    def apply(self, feature: torch.Tensor, vertex_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Offsets ``(..., V, 3)`` of features ``(..., P)``, or ``(..., M, 3)`` for ``vertex_ids (M,)``."""


@final
@dataclass(frozen=True, eq=False)
class LinearCorrective(Corrective):
    """``offsets = feature @ posedirs`` with ``posedirs (P, V * 3)`` (the SMPL family)."""

    posedirs: torch.Tensor

    def apply(self, feature: torch.Tensor, vertex_ids: torch.Tensor | None = None) -> torch.Tensor:
        return _offsets(feature, self.posedirs, vertex_ids)

    def __repr__(self) -> str:
        features, width = self.posedirs.shape
        return (
            f"LinearCorrective(features={features}, vertices={width // 3}, dtype={self.posedirs.dtype}, "
            f"device={self.posedirs.device})"
        )


@final
@dataclass(frozen=True, eq=False, kw_only=True)
class MLPCorrective(Corrective):
    """``offsets = relu(feature @ first_layer.T) @ output`` with ``first_layer (H, P)`` and a sparse ``output (H, V * 3)``.

    ``output`` is kept as N blocks of R hidden rows by T vertices: block ``n`` holds ``output_weights[n] (T * 3, R)``,
    the transposed entries of rows ``output_rows[n] (R,)`` on vertices ``output_vertices[n] (T,)``, zero in a vertex
    slot past the block's last vertex. Entry ``n * T + t`` is slot ``t`` of block ``n``; ``vertex_entries (V, K)``
    lists the entries of each vertex (-1 past its last), which a vertex subset reads. :meth:`from_dense` builds it.
    """

    first_layer: torch.Tensor
    output_rows: torch.Tensor
    output_vertices: torch.Tensor
    output_weights: torch.Tensor
    vertex_entries: torch.Tensor

    @classmethod
    def from_dense(cls, first_layer: torch.Tensor, output: torch.Tensor, *, rows: int) -> MLPCorrective:
        """The corrective of ``first_layer (H, P)`` and ``output (H, V * 3)`` whose hidden rows come in runs of ``rows``
        that move the same vertices (MHR and SOMA: a joint's 24 rows), each run's vertices cut into blocks of 64."""
        num_vertices, device = output.shape[1] // 3, output.device
        moved = (output != 0).unflatten(1, (num_vertices, 3)).any(-1).unflatten(0, (-1, rows)).any(1)  # (H / R, V)
        runs, vertices = [], []
        for run, run_moved in enumerate(moved):
            ids = run_moved.nonzero()[:, 0]
            for start in range(0, len(ids), _BLOCK_VERTICES):
                block = ids[start : start + _BLOCK_VERTICES]
                runs.append(run)
                vertices.append(torch.nn.functional.pad(block, (0, _BLOCK_VERTICES - len(block)), value=-1))
        output_rows = torch.tensor(runs, device=device)[:, None] * rows + torch.arange(rows, device=device)  # (N, R)
        output_vertices = torch.stack(vertices)  # (N, T), -1 past the end of a run's vertices
        weights = output.unflatten(1, (num_vertices, 3))[output_rows[:, :, None], output_vertices[:, None].clamp(min=0)]
        weights = weights * (output_vertices >= 0)[:, None, :, None]  # (N, R, T, 3)
        entry_vertices = output_vertices.flatten()
        entries = (entry_vertices >= 0).nonzero()[:, 0]
        entries = entries[entry_vertices[entries].argsort(stable=True)]  # grouped by vertex
        vertex = entry_vertices[entries]
        rank = torch.arange(len(vertex), device=device) - torch.searchsorted(vertex, vertex)  # among the vertex's
        vertex_entries = torch.full((num_vertices, int(rank.max()) + 1), -1, device=device)
        return cls(
            first_layer=first_layer,
            output_rows=output_rows,
            output_vertices=output_vertices.clamp(min=0),
            output_weights=weights.permute(0, 2, 3, 1).flatten(1, 2).contiguous(),
            vertex_entries=vertex_entries.index_put_((vertex, rank), entries),
        )

    def apply(self, feature: torch.Tensor, vertex_ids: torch.Tensor | None = None) -> torch.Tensor:
        hidden = torch.nn.functional.linear(feature, self.first_layer).relu()  # (..., H)
        if vertex_ids is not None:  # the subset's columns of the dense output, from its entries
            entries = self.vertex_entries[vertex_ids]  # (M, K); entry -1 reads the last one with weight 0
            weights = self.output_weights.unflatten(1, (-1, 3)).flatten(0, 1)[entries] * (entries >= 0)[..., None, None]
            rows = self.output_rows[entries // self.output_vertices.shape[1]]  # (M, K, R)
            matrix = weights.new_zeros(len(entries), 3, len(self.first_layer)).scatter_add(
                2, rows.flatten(1)[:, None].expand(-1, 3, -1), weights.transpose(1, 2).flatten(2)
            )  # (M, 3, H)
            return (hidden @ matrix.flatten(0, 1).T).unflatten(-1, (-1, 3))
        columns = hidden.reshape(-1, hidden.shape[-1]).T  # (H, B): the batch last, so each vertex adds one row
        # index_select, not indexing: its backward is one index_add (on CUDA, indexing's sorts a dozen kernels)
        inputs = columns.index_select(0, self.output_rows.flatten()).unflatten(0, self.output_rows.shape)  # (N, R, B)
        blocks = torch.bmm(self.output_weights, inputs).unflatten(1, (-1, 3))  # (N, T, 3, B)
        offsets = blocks.new_zeros(len(self.vertex_entries), *blocks.shape[2:])
        offsets = offsets.index_add(0, self.output_vertices.flatten(), blocks.flatten(0, 1))  # (V, 3, B)
        # Contiguous: a compiled backward fixes a batch-last stride at the batch it compiled for (torch 2.13, CUDA).
        return offsets.permute(2, 0, 1).reshape(*hidden.shape[:-1], len(offsets), 3).contiguous()

    def __repr__(self) -> str:
        hidden, features = self.first_layer.shape
        return (
            f"MLPCorrective(features={features}, hidden={hidden}, vertices={len(self.vertex_entries)}, "
            f"blocks={len(self.output_rows)}, dtype={self.output_weights.dtype}, device={self.output_weights.device})"
        )


_BLOCK_VERTICES = 64  # vertices per block of MLPCorrective.output; 32, 128 and 512 measured no faster on MHR


def _read_only(mapping: Mapping[Any, Any]) -> MappingProxyType:
    """A read-only copy of ``mapping`` whose mapping values are read-only too: frozen records stay frozen inside."""
    return MappingProxyType(
        {key: _read_only(value) if isinstance(value, Mapping) else value for key, value in mapping.items()}
    )


def _offsets(feature: torch.Tensor, matrix: torch.Tensor, vertex_ids: torch.Tensor | None) -> torch.Tensor:
    if vertex_ids is not None:
        matrix = matrix.unflatten(-1, (-1, 3))[:, vertex_ids].flatten(-2)
    return (feature @ matrix).unflatten(-1, (-1, 3))


@final
@dataclass(frozen=True, eq=False)
class Parts:
    """Vertex segmentation into rigid parts for mass properties.

    ``ids (V,)`` is the part of every vertex; ``joints[p]`` is the native joint whose body carries part ``p`` and
    whose rest pose is the part's reference frame. ``density_categories[p]`` is part ``p``'s key in the density
    tables, or ``None`` when no category covers it; ``density_tables`` maps a table name (``"Dempster"``) to category
    -> kg/m³. ``closed_faces`` maps part ``p`` to the ``(F_p, 3)`` faces of its authored
    closed mesh over ``[rest_vertices; rest_joint_translations]``, or is ``None`` when the parts are closed from
    ``ids`` alone.
    """

    ids: torch.Tensor
    joints: tuple[int, ...]
    density_categories: tuple[str | None, ...]
    density_tables: Mapping[str, Mapping[str, float]]
    closed_faces: Mapping[int, torch.Tensor] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "density_tables", _read_only(self.density_tables))
        if self.closed_faces is not None:
            object.__setattr__(self, "closed_faces", _read_only(self.closed_faces))

    @property
    def count(self) -> int:
        """``P``, the number of parts."""
        return len(self.joints)

    def __repr__(self) -> str:
        return f"Parts(count={self.count}, vertices={self.ids.shape[0]})"


@final
@dataclass(frozen=True, eq=False)
class Landmarks:
    """Named points of the mesh: landmark ``names[l]`` is vertex ``vertex_ids[l]`` (``(L,)`` long)."""

    names: tuple[str, ...]
    vertex_ids: torch.Tensor

    def __repr__(self) -> str:
        return f"Landmarks(count={len(self.names)})"


@final
@dataclass(frozen=True, eq=False)
class FamilySpec:
    """One body family on one dtype and device. Pure data except for ``rig``.

    Shapes: ``template_vertices (V, 3)``, ``faces (F, 3)``, ``skin_weights``/``skin_indices (V, K)`` (top-K
    influences, rows sum to one), blend groups ``(V, 3, n)``, ``file_joint_ids (J,)`` (the weight file's number of
    every kept joint; ``Rig.joint_ids`` is the BetterRobot numbering). ``identity_bases`` groups are baked once per
    subject (``HumanModel.bake``); ``stream_bases`` groups change per frame (``Identity.forward``). ``classic`` maps
    the family's classic parameters (``better_human.from_classic``).
    """

    name: str
    template_vertices: torch.Tensor
    faces: torch.Tensor
    joint_names: tuple[str, ...]
    parents: tuple[int, ...]
    file_joint_ids: torch.Tensor
    skin_weights: torch.Tensor
    skin_indices: torch.Tensor
    identity_bases: Mapping[str, torch.Tensor]
    stream_bases: Mapping[str, torch.Tensor]
    corrective: Corrective | None
    rig: Rig
    parts: Parts
    classic: ClassicFormat
    landmarks: Landmarks | None = None
    up_axis: Literal["+x", "-x", "+y", "-y", "+z", "-z"] = "+y"
    provenance: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("identity_bases", "stream_bases", "provenance"):
            object.__setattr__(self, name, _read_only(getattr(self, name)))
        num_vertices, num_joints = self.template_vertices.shape[0], len(self.joint_names)
        if self.template_vertices.shape != (num_vertices, 3) or self.faces.ndim != 2 or self.faces.shape[1] != 3:
            raise ShapeError("template_vertices must be (V, 3) and faces (F, 3)")
        if len(self.parents) != num_joints or self.parents[0] != -1:
            raise ShapeError("parents must have one entry per joint and start with the root (-1)")
        if self.file_joint_ids.shape != (num_joints,):
            raise ShapeError("file_joint_ids must be (J,)")
        if self.skin_weights.shape != self.skin_indices.shape or self.skin_weights.shape[0] != num_vertices:
            raise ShapeError("skin_weights and skin_indices must both be (V, K)")
        for group, basis in (*self.identity_bases.items(), *self.stream_bases.items()):
            if basis.shape[:2] != (num_vertices, 3):
                raise ShapeError(f"blend group {group!r} must be (V, 3, n); got {tuple(basis.shape)}")
        if self.parts.ids.shape != (num_vertices,):
            raise ShapeError("parts.ids must be (V,)")
        closed = self.parts.closed_faces
        if len(self.parts.density_categories) != self.parts.count or (
            closed is not None and len(closed) != self.parts.count
        ):
            raise ShapeError("parts.density_categories and parts.closed_faces must have one entry per part")

    @property
    def num_vertices(self) -> int:
        """``V``, the template's vertex count."""
        return self.template_vertices.shape[0]

    @property
    def num_joints(self) -> int:
        """``J``, the native joint count."""
        return len(self.joint_names)

    @property
    def dtype(self) -> torch.dtype:
        """The dtype of every floating tensor of the spec."""
        return self.template_vertices.dtype

    @property
    def device(self) -> torch.device:
        """The device of every tensor of the spec."""
        return self.template_vertices.device

    @property
    def identity_groups(self) -> dict[str, int]:
        """Identity group name to its coefficient count ``n``."""
        return {name: basis.shape[-1] for name, basis in self.identity_bases.items()}

    @property
    def stream_groups(self) -> dict[str, int]:
        """Stream group name to its coefficient count ``n``."""
        return {name: basis.shape[-1] for name, basis in self.stream_bases.items()}

    def joint_id(self, name: str) -> int:
        """Native index of joint ``name``; :class:`~better_robot.exceptions.UnknownNameError` for another name."""
        if name not in self.joint_names:
            raise UnknownNameError("joint", name, valid=self.joint_names)
        return self.joint_names.index(name)

    def __repr__(self) -> str:
        return (
            f"FamilySpec({self.name}, vertices={self.num_vertices}, joints={self.num_joints}, dtype={self.dtype}, "
            f"device={self.device})"
        )

    def skin_matrix(self) -> torch.Tensor:
        """Dense ``(V, J)`` skin weights."""
        dense = self.skin_weights.new_zeros((self.num_vertices, self.num_joints))
        return dense.scatter_add_(1, self.skin_indices, self.skin_weights)


__all__ = ["Corrective", "FamilySpec", "Landmarks", "LinearCorrective", "MLPCorrective", "Parts"]
