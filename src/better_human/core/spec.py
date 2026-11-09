"""``FamilySpec``: a body family as data — mesh, skin weights, blend groups, corrective, rig."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal

import torch

if TYPE_CHECKING:
    from .rig import Rig


META_KEYS = frozenset(
    {
        "bind_joint_positions",
        "closed_segmentation",
        "hand_components",
        "hand_joints",
        "hand_mean",
        "native_joint_ids",
        "side",
    }
)
"""Every ``FamilySpec.meta`` key read by ``better_human.core`` and ``better_human.families``."""


@dataclass(frozen=True)
class Corrective:
    """Pose-dependent vertex offsets as data: a linear basis or a one-hidden-layer MLP.

    ``linear``: ``offsets = feature @ posedirs`` with ``posedirs (P, V * 3)``.
    ``mlp``: ``offsets = relu(feature @ first_layer.T) @ output`` with ``first_layer (H, P)`` and
    ``output (H, V * 3)``. ``none``: no offsets. Features come from the rig.
    """

    kind: Literal["none", "linear", "mlp"] = "none"
    posedirs: torch.Tensor | None = None
    first_layer: torch.Tensor | None = None
    output: torch.Tensor | None = None

    def apply(self, feature: torch.Tensor, vertex_ids: torch.Tensor | None = None) -> torch.Tensor | None:
        """Offsets ``(..., V, 3)`` (or ``(..., M, 3)`` for ``vertex_ids (M,)``); ``None`` when ``kind == "none"``."""
        if self.kind == "none":
            return None
        if self.kind == "linear":
            matrix = self.posedirs
        else:
            feature = torch.nn.functional.linear(feature, self.first_layer).relu()
            matrix = self.output
        if vertex_ids is not None:
            matrix = matrix.unflatten(-1, (-1, 3))[:, vertex_ids].flatten(-2)
        return (feature @ matrix).unflatten(-1, (-1, 3))

    def to(self, dtype: torch.dtype, device: torch.device) -> Corrective:
        return Corrective(
            self.kind,
            *(
                None if t is None else t.to(dtype=dtype, device=device)
                for t in (self.posedirs, self.first_layer, self.output)
            ),
        )


@dataclass(frozen=True)
class Parts:
    """Vertex segmentation into rigid parts for mass properties.

    ``ids (V,)`` is the part of every vertex; ``joints[p]`` is the native joint whose body carries
    part ``p`` and whose rest pose is the part's reference frame.
    """

    ids: torch.Tensor
    joints: tuple[int, ...]

    @property
    def count(self) -> int:
        return len(self.joints)


@dataclass(frozen=True)
class FamilySpec:
    """One body family on one dtype and device. Pure data except for ``rig``.

    Shapes: ``template_vertices (V, 3)``, ``faces (F, 3)``, ``skin_weights``/``skin_indices (V, K)``
    (top-K influences, rows sum to one), blend groups ``(V, 3, n)``. ``identity_bases`` groups are baked
    once per subject (``HumanModel.bake``); ``stream_bases`` groups change per frame (``Identity.forward``).
    ``landmarks`` maps a set name to ``(L,)`` vertex ids. ``meta`` carries family extras that the
    loader and its classic converters own; every key core and the families read
    (:data:`META_KEYS`):

    - ``native_joint_ids`` ``(J,)`` long: file joint numbering of the kept joints (set by ``drop_joints``).
    - ``hand_joints`` ``set[str]``: sides whose hand joints survived loading.
    - ``hand_mean`` ``{side: (45,)}`` and ``hand_components`` ``{side: (C, 45)}``: hand pose PCA.
    - ``side`` ``str``: ``"left"`` or ``"right"`` (MANO).
    - ``closed_segmentation``: authored part segmentation for mass properties (MHR), or absent.
    - ``bind_joint_positions`` ``(J, 3)``: pivots of the accumulated joint scales in skinning (MHR).
    """

    name: str
    template_vertices: torch.Tensor
    faces: torch.Tensor
    joint_names: tuple[str, ...]
    parents: tuple[int, ...]
    skin_weights: torch.Tensor
    skin_indices: torch.Tensor
    identity_bases: Mapping[str, torch.Tensor]
    stream_bases: Mapping[str, torch.Tensor]
    corrective: Corrective
    rig: Rig
    parts: Parts
    landmarks: Mapping[str, torch.Tensor] = field(default_factory=dict)
    unit: str = "m"
    up_axis: str = "+y"
    forward_axis: str = "+z"
    meta: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        num_vertices, num_joints = self.template_vertices.shape[0], len(self.joint_names)
        if self.template_vertices.shape != (num_vertices, 3) or self.faces.ndim != 2 or self.faces.shape[1] != 3:
            raise ValueError("template_vertices must be (V, 3) and faces (F, 3)")
        if len(self.parents) != num_joints or self.parents[0] != -1:
            raise ValueError("parents must have one entry per joint and start with the root (-1)")
        if self.skin_weights.shape != self.skin_indices.shape or self.skin_weights.shape[0] != num_vertices:
            raise ValueError("skin_weights and skin_indices must both be (V, K)")
        for group, basis in (*self.identity_bases.items(), *self.stream_bases.items()):
            if basis.shape[:2] != (num_vertices, 3):
                raise ValueError(f"blend group {group!r} must be (V, 3, n); got {tuple(basis.shape)}")
        if self.parts.ids.shape != (num_vertices,):
            raise ValueError("parts.ids must be (V,)")

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
        return self.joint_names.index(name)

    def skin_matrix(self) -> torch.Tensor:
        """Dense ``(V, J)`` skin weights."""
        dense = self.skin_weights.new_zeros((self.num_vertices, self.num_joints))
        return dense.scatter_add_(1, self.skin_indices, self.skin_weights)

    def to(self, *, dtype: torch.dtype | None = None, device: torch.device | str | None = None) -> FamilySpec:
        """Copy with every floating tensor cast to ``dtype`` and every tensor moved to ``device``."""
        dtype = dtype or self.dtype
        device = torch.device(device) if device is not None else self.device

        def cast(value: Any) -> Any:
            if isinstance(value, torch.Tensor):
                return value.to(device=device, dtype=dtype) if value.is_floating_point() else value.to(device=device)
            if isinstance(value, Mapping):
                return {key: cast(item) for key, item in value.items()}
            return value

        return replace(
            self,
            template_vertices=cast(self.template_vertices),
            faces=cast(self.faces),
            skin_weights=cast(self.skin_weights),
            skin_indices=cast(self.skin_indices),
            identity_bases=cast(self.identity_bases),
            stream_bases=cast(self.stream_bases),
            corrective=self.corrective.to(dtype, device),
            rig=self.rig.to(dtype, device),
            parts=Parts(cast(self.parts.ids), self.parts.joints),
            landmarks=cast(self.landmarks),
            meta=cast(self.meta),
        )


def top_k_skin(weights: torch.Tensor, k: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Dense ``(V, J)`` weights to padded ``(V, K)`` values and indices (``K`` = largest row support)."""
    nonzero = weights != 0
    if k is None:
        k = int(nonzero.sum(dim=-1).max().item())
    order = torch.argsort(nonzero.to(torch.int8), dim=-1, descending=True, stable=True)[:, :k]
    values = torch.gather(weights, 1, order)
    indices = torch.where(values != 0, order, torch.zeros_like(order))
    return values, indices


__all__ = ["Corrective", "FamilySpec", "Parts", "top_k_skin"]
