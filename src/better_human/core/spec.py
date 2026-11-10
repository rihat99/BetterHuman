"""``FamilySpec``: a body family as data — mesh, skin weights, blend groups, corrective, rig, parts, landmarks.

A family loader builds the spec once at the requested dtype and device (decision record 0005); nothing in
BetterHuman moves or casts it afterwards. Every field has a type (record 0008): ``provenance`` is the only free-form
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
    """Pose-dependent vertex offsets computed from the features the family's rig gives (record 0008)."""

    @abstractmethod
    def apply(self, feature: torch.Tensor, vertex_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Offsets ``(..., V, 3)`` of features ``(..., P)``, or ``(..., M, 3)`` for ``vertex_ids (M,)``."""


@final
@dataclass(frozen=True)
class LinearCorrective(Corrective):
    """``offsets = feature @ posedirs`` with ``posedirs (P, V * 3)`` (the SMPL family)."""

    posedirs: torch.Tensor

    def apply(self, feature: torch.Tensor, vertex_ids: torch.Tensor | None = None) -> torch.Tensor:
        return _offsets(feature, self.posedirs, vertex_ids)


@final
@dataclass(frozen=True)
class MLPCorrective(Corrective):
    """``offsets = relu(feature @ first_layer.T) @ output`` with ``first_layer (H, P)``, ``output (H, V * 3)``."""

    first_layer: torch.Tensor
    output: torch.Tensor

    def apply(self, feature: torch.Tensor, vertex_ids: torch.Tensor | None = None) -> torch.Tensor:
        return _offsets(torch.nn.functional.linear(feature, self.first_layer).relu(), self.output, vertex_ids)


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
@dataclass(frozen=True)
class Parts:
    """Vertex segmentation into rigid parts for mass properties.

    ``ids (V,)`` is the part of every vertex; ``joints[p]`` is the native joint whose body carries part ``p`` and
    whose rest pose is the part's reference frame. ``density_categories[p]`` is part ``p``'s key in the density
    tables, or ``None`` when no category covers it; ``density_tables`` maps a table name (``"Dempster"``) to category
    -> kg/m³ (decision record 0007). ``closed_faces`` maps part ``p`` to the ``(F_p, 3)`` faces of its authored
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
        return len(self.joints)


@final
@dataclass(frozen=True)
class Landmarks:
    """Named points of the mesh: landmark ``names[l]`` is vertex ``vertex_ids[l]`` (``(L,)`` long)."""

    names: tuple[str, ...]
    vertex_ids: torch.Tensor


@final
@dataclass(frozen=True)
class FamilySpec:
    """One body family on one dtype and device. Pure data except for ``rig``.

    Shapes: ``template_vertices (V, 3)``, ``faces (F, 3)``, ``skin_weights``/``skin_indices (V, K)`` (top-K
    influences, rows sum to one), blend groups ``(V, 3, n)``, ``file_joint_ids (J,)`` (the weight file's number of
    every kept joint; ``Rig.joint_ids`` is the BetterRobot numbering). ``identity_bases`` groups are baked once per
    subject (``HumanModel.bake``); ``stream_bases`` groups change per frame (``Identity.forward``). ``classic`` maps
    the family's classic parameters (``better_human.from_classic``, decision record 0004).
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
        return self.template_vertices.shape[0]

    @property
    def num_joints(self) -> int:
        return len(self.joint_names)

    @property
    def dtype(self) -> torch.dtype:
        return self.template_vertices.dtype

    @property
    def device(self) -> torch.device:
        return self.template_vertices.device

    @property
    def identity_groups(self) -> dict[str, int]:
        return {name: basis.shape[-1] for name, basis in self.identity_bases.items()}

    @property
    def stream_groups(self) -> dict[str, int]:
        return {name: basis.shape[-1] for name, basis in self.stream_bases.items()}

    def joint_id(self, name: str) -> int:
        """Native index of joint ``name``; :class:`~better_robot.exceptions.UnknownNameError` for another name."""
        if name not in self.joint_names:
            raise UnknownNameError("joint", name, valid=self.joint_names)
        return self.joint_names.index(name)

    def skin_matrix(self) -> torch.Tensor:
        """Dense ``(V, J)`` skin weights."""
        dense = self.skin_weights.new_zeros((self.num_vertices, self.num_joints))
        return dense.scatter_add_(1, self.skin_indices, self.skin_weights)


__all__ = ["Corrective", "FamilySpec", "Landmarks", "LinearCorrective", "MLPCorrective", "Parts"]
