"""``FamilySpec``: a body family as data — mesh, skin weights, blend groups, corrective, rig."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal

import torch

from .reduction import reduce_joint_data

if TYPE_CHECKING:
    from .rig import Rig


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

    Shapes: ``vertices (V, 3)``, ``faces (F, 3)``, ``skin_weights``/``skin_indices (V, K)``
    (top-K influences, rows sum to one), blend groups ``(V, 3, n)``. ``identity`` groups are baked
    once per subject (``Body.bake``); ``streams`` change per frame (``Identity.forward``).
    ``landmarks`` maps a set name to ``(L,)`` vertex ids. ``meta`` carries family extras that the
    loader and its classic converters own (hand PCA, MHR bindings).
    """

    name: str
    vertices: torch.Tensor
    faces: torch.Tensor
    joint_names: tuple[str, ...]
    parents: tuple[int, ...]
    skin_weights: torch.Tensor
    skin_indices: torch.Tensor
    identity: Mapping[str, torch.Tensor]
    streams: Mapping[str, torch.Tensor]
    corrective: Corrective
    rig: Rig
    parts: Parts
    landmarks: Mapping[str, torch.Tensor] = field(default_factory=dict)
    unit: str = "m"
    up_axis: str = "+y"
    forward_axis: str = "+z"
    meta: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        num_vertices, num_joints = self.vertices.shape[0], len(self.joint_names)
        if self.vertices.shape != (num_vertices, 3) or self.faces.ndim != 2 or self.faces.shape[1] != 3:
            raise ValueError("vertices must be (V, 3) and faces (F, 3)")
        if len(self.parents) != num_joints or self.parents[0] != -1:
            raise ValueError("parents must have one entry per joint and start with the root (-1)")
        if self.skin_weights.shape != self.skin_indices.shape or self.skin_weights.shape[0] != num_vertices:
            raise ValueError("skin_weights and skin_indices must both be (V, K)")
        for group, basis in (*self.identity.items(), *self.streams.items()):
            if basis.shape[:2] != (num_vertices, 3):
                raise ValueError(f"blend group {group!r} must be (V, 3, n); got {tuple(basis.shape)}")
        if self.parts.ids.shape != (num_vertices,):
            raise ValueError("parts.ids must be (V,)")

    @property
    def num_vertices(self) -> int:
        return self.vertices.shape[0]

    @property
    def num_joints(self) -> int:
        return len(self.joint_names)

    @property
    def dtype(self) -> torch.dtype:
        return self.vertices.dtype

    @property
    def device(self) -> torch.device:
        return self.vertices.device

    @property
    def identity_groups(self) -> dict[str, int]:
        return {name: basis.shape[-1] for name, basis in self.identity.items()}

    @property
    def stream_groups(self) -> dict[str, int]:
        return {name: basis.shape[-1] for name, basis in self.streams.items()}

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
            vertices=cast(self.vertices),
            faces=cast(self.faces),
            skin_weights=cast(self.skin_weights),
            skin_indices=cast(self.skin_indices),
            identity=cast(self.identity),
            streams=cast(self.streams),
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


def drop_joints(spec: FamilySpec, names: Sequence[str]) -> FamilySpec:
    """Spec without the terminal subtrees rooted at ``names`` (their skin weights go to the surviving ancestor).

    Only families with a regressed rig and a linear ``(R - I)`` corrective support this (SMPL family).
    """
    from .rig import RegressedRig  # noqa: PLC0415 - rig imports spec

    if not names:
        return spec
    if not isinstance(spec.rig, RegressedRig) or spec.corrective.kind != "linear":
        raise ValueError("drop_joints needs a regressed rig with a linear corrective")
    if spec.parts.joints != tuple(range(spec.num_joints)):
        raise ValueError("drop_joints expects one part per joint")
    dropped = [spec.joint_id(name) for name in names]
    posedirs = spec.corrective.posedirs.T.reshape(spec.num_vertices, 3, -1)
    joint_names, parents, regressor, dense, posedirs, survivors = reduce_joint_data(
        spec.joint_names, spec.parents, spec.rig.joint_regressor, spec.skin_matrix(), posedirs, dropped
    )
    weights, indices = top_k_skin(dense)
    survivor_tensor = torch.tensor(survivors, dtype=torch.long, device=spec.device)
    if "native_joint_ids" in spec.meta:  # an earlier drop already renumbered joints; keep file numbering
        survivor_tensor = spec.meta["native_joint_ids"].to(spec.device)[survivor_tensor]
    old_to_new = {old: new for new, old in enumerate(survivors)}
    part_joints = tuple(old_to_new[j] for j in spec.parts.joints if j in old_to_new)
    ancestor = torch.arange(spec.num_joints, device=spec.device)
    for joint in range(spec.num_joints):  # every dropped joint maps to its nearest surviving ancestor
        while int(ancestor[joint]) not in old_to_new:
            ancestor[joint] = spec.parents[int(ancestor[joint])]
    part_ids = torch.tensor([old_to_new[int(ancestor[j])] for j in range(spec.num_joints)], device=spec.device)
    return replace(
        spec,
        joint_names=joint_names,
        parents=parents,
        skin_weights=weights,
        skin_indices=indices,
        corrective=Corrective("linear", posedirs=posedirs.reshape(spec.num_vertices * 3, -1).T),
        rig=replace(spec.rig, joint_regressor=regressor),
        parts=Parts(part_ids[spec.parts.ids], part_joints),
        meta={**spec.meta, "native_joint_ids": survivor_tensor},
    )


__all__ = ["Corrective", "FamilySpec", "Parts", "drop_joints", "top_k_skin"]
