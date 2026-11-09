"""MHR (Momentum Human Rig) family: loader, classic parameter container and converters.

``mhr(...)`` returns a :class:`~better_human.core.spec.FamilySpec` with an :class:`~better_human.families.mhr_rig.MHRRig`. Its identity group
``identity`` takes the 45 upstream ``identity_coeffs``; ``proportions`` (a zero mesh basis) carries the 73 proportion
parameters to the rig. Upstream ``model_parameters`` enter through :func:`mhr_from_classic`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, final

import torch
from better_robot.exceptions import ArgumentError, DtypeMismatchError
from better_robot.lie import so3

from better_human.core.classic import broadcast_configuration, classic_tensor
from better_human.core.spec import FamilySpec, MLPCorrective, Parts
from better_human.io import FormatError, weight_path
from better_human.io.load import load_model_npz
from better_human.io.mass_data import shipped_density_tables
from better_human.io.schema import MHR_CORRECTIVE_FIELDS, MHR_SCHEMA
from better_human.mass import segment_vertices

from .densities import density_categories

from .mhr_rig import ROOT_COUNT, MHRRig

if TYPE_CHECKING:
    from better_human.core.human_model import HumanModel, Identity

IDENTITY_COUNT = 45
EXPRESSION_COUNT = 72
PARAMETER_COUNT = 204
LODS = tuple(range(7))


def mhr(
    path: str | Path | None = None,
    *,
    lod: int = 1,
    expression: bool = True,
    correctives: bool = True,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> FamilySpec:
    """Load ``MHR/converted/mhr_lod{lod}.npz`` (``tools/convert/mhr``): 127 joints, ``identity (V, 3, 45)``.

    ``proportions (V, 3, 73)`` is a zero mesh basis whose values reach the rig. ``expression=False`` drops the
    ``expression (V, 3, 72)`` stream, ``correctives=False`` the pose-corrective MLP. The spec is built
    once in ``dtype`` on ``device`` (decision record 0005).
    """
    if not isinstance(lod, int) or isinstance(lod, bool) or lod not in LODS:
        raise ArgumentError(f"lod must be one of {LODS}, got {lod!r}")
    if not isinstance(expression, bool) or not isinstance(correctives, bool):
        raise ArgumentError(f"expression and correctives must be bool, got {expression!r} and {correctives!r}")
    if dtype not in (torch.float32, torch.float64):
        raise DtypeMismatchError(
            f"dtype={dtype!r} is not torch.float32 or torch.float64, the floating dtypes BetterRobot computes in; "
            "pass dtype=torch.float32 or dtype=torch.float64"
        )
    device = torch.device("cpu" if device is None else device)
    excluded = set() if correctives else set(MHR_CORRECTIVE_FIELDS)
    if not expression:
        excluded.add("expression_basis")
    arrays = load_model_npz(
        weight_path("MHR", f"converted/mhr_lod{lod}.npz", path=path), MHR_SCHEMA, exclude_fields=excluded
    )
    if int(arrays["lod"]) != lod:
        raise FormatError(f"the MHR archive holds lod {int(arrays['lod'])}, expected lod {lod}; pass lod= or its path")

    def floats(name: str) -> torch.Tensor:
        return arrays[name].to(device=device, dtype=dtype)

    corrective = None
    if correctives:
        first_layer = _dense_weight_from_sparse(
            arrays["corrective_first_layer_indices"].to(device),
            floats("corrective_first_layer_weights"),
            arrays["corrective_first_layer_mask"].to(device),
        )
        corrective = MLPCorrective(first_layer, floats("corrective_output_weights"))
    # Parts: the segmentation groups but ``zero_weight``; a vertex joins the group of its strongest skin joint.
    skin_weights, skin_indices = floats("skin_weights"), arrays["skin_indices"].to(device)
    kept = [group for group, name in enumerate(arrays["segmentation_group_names"]) if name != "zero_weight"]
    joint_groups = arrays["segmentation_joint_group_indices"].tolist()
    if not set(kept) <= set(joint_groups):
        raise FormatError("every MHR segmentation group but zero_weight must contain a joint")
    joint_parts = torch.tensor([kept.index(group) if group in kept else -1 for group in joint_groups], device=device)
    part_ids = joint_parts[segment_vertices(skin_weights, skin_indices)]
    if bool((part_ids < 0).any()):
        raise FormatError("MHR positive skin weights unexpectedly map to zero_weight")
    rig = MHRRig.from_arrays(arrays, dtype=dtype, device=device)
    template = floats("template_vertices")
    joint_names = tuple(str(name) for name in arrays["joint_names"])
    part_joints = tuple(joint_groups.index(group) for group in kept)
    density_tables = shipped_density_tables()
    categories = density_categories([joint_names[joint] for joint in part_joints], density_tables)
    return FamilySpec(
        name="mhr",
        template_vertices=template,
        faces=arrays["faces"].to(device),
        joint_names=joint_names,
        parents=tuple(arrays["joint_parents"].tolist()),
        file_joint_ids=torch.arange(len(joint_names), device=device),
        skin_weights=skin_weights,
        skin_indices=skin_indices,
        identity_bases={
            "identity": floats("identity_basis"),
            "proportions": template.new_zeros(len(template), 3, len(rig.proportion_columns)),
        },
        stream_bases={"expression": floats("expression_basis")} if expression else {},
        corrective=corrective,
        rig=rig,
        parts=Parts(part_ids, part_joints, categories, density_tables),
        up_axis=str(arrays["up_axis"]),
        provenance={"archive": str(arrays["provenance"])},
    )


@final
@dataclass(frozen=True)
class MHRClassic:
    """Reference MHR parameters.

    ``identity_coeffs (..., 45)``, ``model_parameters (..., 204)`` and ``expression (..., 72)`` (or ``None``).
    """

    identity_coeffs: torch.Tensor
    model_parameters: torch.Tensor
    expression: torch.Tensor | None = None


def mhr_from_classic(human: HumanModel, classic: MHRClassic) -> tuple[Identity, torch.Tensor, dict[str, torch.Tensor]]:
    """``(Identity, q (..., 132), stream)`` for ``human`` (a :class:`~better_human.core.human_model.HumanModel` of ``mhr``)."""
    spec, rig = human.spec, human.spec.rig
    exemplar = spec.template_vertices
    identity = classic_tensor("identity_coeffs", classic.identity_coeffs, IDENTITY_COUNT, exemplar)
    model_parameters = classic_tensor("model_parameters", classic.model_parameters, PARAMETER_COUNT, exemplar)
    if identity is None or model_parameters is None:
        raise TypeError("identity_coeffs and model_parameters must be floating torch.Tensors")
    use_expression = "expression" in spec.stream_bases
    if not use_expression and classic.expression is not None:
        raise ArgumentError("expression requires an MHR spec loaded with expression=True")
    expression = (
        classic_tensor("expression", classic.expression, EXPRESSION_COUNT, exemplar) if use_expression else None
    )
    shapes = [identity.shape[:-1], model_parameters.shape[:-1]] + (
        [expression.shape[:-1]] if expression is not None else []
    )
    batch = torch.broadcast_shapes(*shapes)
    identity = identity.expand(*batch, IDENTITY_COUNT)
    model_parameters = model_parameters.expand(*batch, PARAMETER_COUNT)
    baked = human.bake(identity=identity, proportions=model_parameters.index_select(-1, rig.proportion_columns))
    stream = {}
    if use_expression:
        zeros = exemplar.new_zeros((*batch, EXPRESSION_COUNT))
        stream["expression"] = zeros if expression is None else expression.expand(*batch, EXPRESSION_COUNT)
    root = model_parameters[..., :ROOT_COUNT] @ rig.root_transform.mT
    pose = model_parameters.index_select(-1, rig.pose_columns) * rig.host_weights
    return baked, torch.cat((root[..., :3], so3.from_euler(root[..., 3:]), pose), dim=-1), stream


def mhr_to_classic(identity: Identity, q: torch.Tensor, stream: dict[str, torch.Tensor] | None = None) -> MHRClassic:
    """Classic parameters for a baked MHR ``identity`` and ``q (..., 132)``; Euler angles on BR's principal branch."""
    spec, rig = identity.human.spec, identity.human.spec.rig
    coeffs = identity.values["identity"]
    q, batch = broadcast_configuration(q, identity.model, coeffs)
    root_quaternion = so3.normalize(q[..., 3:7])
    root_euler = so3.to_euler(root_quaternion)
    flip = (so3.from_euler(root_euler) * root_quaternion).sum(dim=-1, keepdim=True) < 0
    roll = root_euler[..., :1] + torch.where(flip, root_euler.new_tensor(2.0 * torch.pi), root_euler.new_tensor(0.0))
    root_channels = torch.cat((q[..., :3], roll, root_euler[..., 1:]), dim=-1)
    root_parameters = root_channels @ torch.linalg.inv(rig.root_transform).mT
    model_parameters = torch.cat((root_parameters, q.new_zeros((*batch, PARAMETER_COUNT - ROOT_COUNT))), dim=-1)
    model_parameters = model_parameters.index_copy(-1, rig.pose_columns, q[..., 7:] / rig.host_weights)
    proportions = identity.values["proportions"].expand(*batch, len(rig.proportion_columns))
    model_parameters = model_parameters.index_copy(-1, rig.proportion_columns, proportions)
    expression = None
    if "expression" in spec.stream_bases:
        value = (stream or {}).get("expression")
        expression = (
            q.new_zeros((*batch, EXPRESSION_COUNT)) if value is None else value.expand(*batch, EXPRESSION_COUNT)
        )
    return MHRClassic(coeffs.expand(*batch, IDENTITY_COUNT), model_parameters, expression)


def _dense_weight_from_sparse(
    sparse_indices: torch.Tensor,
    sparse_values: torch.Tensor,
    sparse_mask: torch.Tensor,
) -> torch.Tensor:
    """Bake coordinate-list values into one immutable dense weight matrix."""
    if sparse_indices.dtype != torch.long or sparse_indices.ndim != 2 or sparse_indices.shape[0] != 2:
        raise FormatError("sparse_indices must be an int64 tensor with shape (2, K)")
    if sparse_values.ndim != 1 or sparse_values.shape[0] != sparse_indices.shape[1]:
        raise FormatError("sparse_values must have shape (K,) matching sparse_indices")
    if sparse_mask.dtype != torch.bool or sparse_mask.ndim != 2:
        raise FormatError("sparse_mask must be a bool tensor with shape (H, P)")
    if not torch.equal(sparse_indices, sparse_mask.nonzero().T):
        raise FormatError("sparse_indices must exactly match sparse_mask.nonzero().T")
    dense_weight = sparse_values.new_zeros(sparse_mask.shape)
    return dense_weight.index_put((sparse_indices[0], sparse_indices[1]), sparse_values)


__all__ = ["MHRClassic", "mhr", "mhr_from_classic", "mhr_to_classic"]
