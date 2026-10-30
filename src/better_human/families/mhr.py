"""MHR (Momentum Human Rig) family: loader, classic parameter container and converters.

``mhr(...)`` returns a :class:`~better_human.core.spec.FamilySpec` with an :class:`MhrRig`. Its identity group
``identity`` takes the 45 upstream ``identity_coeffs``; ``proportions`` (a zero mesh basis) carries the 73 proportion
parameters to the rig. Upstream ``model_parameters`` enter through :func:`mhr_from_classic`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from better_robot.lie import so3

from better_human.core.classic import broadcast_configuration, classic_tensor
from better_human.core.mass import segment_vertices
from better_human.core.pipeline import dense_weight_from_sparse
from better_human.core.spec import Corrective, FamilySpec, Parts
from better_human.io import MHR_CORRECTIVE_FIELDS, MHR_SCHEMA, load_model_npz, resolve_model_path

from .mhr_rig import ROOT_COUNT, MhrRig

if TYPE_CHECKING:
    from better_human.core.body import Body, Identity

IDENTITY_COUNT = 45
EXPRESSION_COUNT = 72
PARAMETER_COUNT = 204
LODS = tuple(range(7))
METADATA = {"format_version": 1, "up_axis": "+y", "forward_axis": "+z", "coordinate_axis": "+x"}


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
    ``expression (V, 3, 72)`` stream, ``correctives=False`` the pose-corrective MLP. ``dtype=torch.float64`` derives
    the rig constants in float64; ``Body(spec, dtype=torch.float64)`` upcasts a float32 spec's rounded pre-rotations
    and inverse bind poses (the BR model is float64 either way).
    """
    if not isinstance(lod, int) or isinstance(lod, bool) or lod not in LODS:
        raise ValueError(f"lod must be one of {LODS}, got {lod!r}")
    if not isinstance(expression, bool) or not isinstance(correctives, bool):
        raise TypeError("expression and correctives must be bool")
    if dtype not in (torch.float32, torch.float64):
        raise TypeError(f"dtype must be torch.float32 or torch.float64, got {dtype}")
    device = torch.device("cpu" if device is None else device)
    excluded = set() if correctives else set(MHR_CORRECTIVE_FIELDS)
    if not expression:
        excluded.add("expression_basis")
    arrays = load_model_npz(
        resolve_model_path("MHR", f"converted/mhr_lod{lod}.npz", path), MHR_SCHEMA, exclude_fields=excluded
    )
    metadata = {name: arrays[name].item() for name in (*METADATA, "lod")}
    if metadata != {**METADATA, "lod": lod}:
        raise ValueError(f"unsupported MHR archive {metadata}; expected {METADATA} and lod {lod}")

    def floats(name: str) -> torch.Tensor:
        return arrays[name].to(device=device, dtype=dtype)

    corrective = Corrective("none")
    if correctives:
        first_layer = dense_weight_from_sparse(
            arrays["corrective_first_layer_indices"].to(device),
            floats("corrective_first_layer_weights"),
            arrays["corrective_first_layer_mask"].to(device),
        )
        corrective = Corrective("mlp", first_layer=first_layer, output=floats("corrective_output_weights"))
    # Parts: the segmentation groups but ``zero_weight``; a vertex joins the group of its strongest skin joint.
    skin_weights, skin_indices = floats("skin_weights"), arrays["skin_indices"].to(device)
    kept = [group for group, name in enumerate(arrays["segmentation_group_names"]) if name != "zero_weight"]
    joint_groups = arrays["segmentation_joint_group_indices"].tolist()
    if not set(kept) <= set(joint_groups):
        raise ValueError("every MHR segmentation group but zero_weight must contain a joint")
    joint_parts = torch.tensor([kept.index(group) if group in kept else -1 for group in joint_groups], device=device)
    part_ids = joint_parts[segment_vertices(skin_weights, skin_indices)]
    if bool((part_ids < 0).any()):
        raise ValueError("MHR positive skin weights unexpectedly map to zero_weight")
    rig = MhrRig.from_arrays(arrays, dtype=dtype, device=device)
    template = floats("template_vertices")
    return FamilySpec(
        name="mhr",
        vertices=template,
        faces=arrays["faces"].to(device),
        joint_names=tuple(str(name) for name in arrays["joint_names"]),
        parents=tuple(arrays["joint_parents"].tolist()),
        skin_weights=skin_weights,
        skin_indices=skin_indices,
        identity={
            "identity": floats("identity_basis"),
            "proportions": template.new_zeros(len(template), 3, len(rig.proportion_columns)),
        },
        streams={"expression": floats("expression_basis")} if expression else {},
        corrective=corrective,
        rig=rig,
        parts=Parts(part_ids, tuple(joint_groups.index(group) for group in kept)),
        unit="m",
        up_axis=METADATA["up_axis"],
        forward_axis=METADATA["forward_axis"],
        meta={"bind_joint_positions": floats("bind_matrices")[..., :3, 3]},
    )


@dataclass(frozen=True)
class MHRClassic:
    """Reference MHR parameters.

    ``identity_coeffs (..., 45)``, ``model_parameters (..., 204)`` and ``expression (..., 72)`` (or ``None``).
    """

    identity_coeffs: torch.Tensor
    model_parameters: torch.Tensor
    expression: torch.Tensor | None = None


def mhr_from_classic(body: Body, classic: MHRClassic) -> tuple[Identity, torch.Tensor, dict[str, torch.Tensor]]:
    """``(Identity, q (..., 132), stream)`` for ``body`` (a :class:`~better_human.core.body.Body` of ``mhr``)."""
    spec, rig = body.spec, body.spec.rig
    exemplar = spec.vertices
    identity = classic_tensor("identity_coeffs", classic.identity_coeffs, IDENTITY_COUNT, exemplar)
    model_parameters = classic_tensor("model_parameters", classic.model_parameters, PARAMETER_COUNT, exemplar)
    if identity is None or model_parameters is None:
        raise TypeError("identity_coeffs and model_parameters must be floating torch.Tensors")
    use_expression = "expression" in spec.streams
    if not use_expression and classic.expression is not None:
        raise ValueError("expression requires an MHR spec loaded with expression=True")
    expression = (
        classic_tensor("expression", classic.expression, EXPRESSION_COUNT, exemplar) if use_expression else None
    )
    shapes = [identity.shape[:-1], model_parameters.shape[:-1]] + (
        [expression.shape[:-1]] if expression is not None else []
    )
    batch = torch.broadcast_shapes(*shapes)
    identity = identity.expand(*batch, IDENTITY_COUNT)
    model_parameters = model_parameters.expand(*batch, PARAMETER_COUNT)
    baked = body.bake(identity=identity, proportions=model_parameters.index_select(-1, rig.proportion_columns))
    stream = {}
    if use_expression:
        zeros = exemplar.new_zeros((*batch, EXPRESSION_COUNT))
        stream["expression"] = zeros if expression is None else expression.expand(*batch, EXPRESSION_COUNT)
    root = model_parameters[..., :ROOT_COUNT] @ rig.root_transform.mT
    pose = model_parameters.index_select(-1, rig.pose_columns) * rig.host_weights
    return baked, torch.cat((root[..., :3], so3.from_euler(root[..., 3:]), pose), dim=-1), stream


def mhr_to_classic(identity: Identity, q: torch.Tensor, stream: dict[str, torch.Tensor] | None = None) -> MHRClassic:
    """Classic parameters for a baked MHR ``identity`` and ``q (..., 132)``; Euler angles on BR's principal branch."""
    spec, rig = identity.spec, identity.spec.rig
    coeffs = identity.params["identity"]
    q, batch = broadcast_configuration(q, identity.model, coeffs)
    root_quaternion = so3.normalize(q[..., 3:7])
    root_euler = so3.to_euler(root_quaternion)
    flip = (so3.from_euler(root_euler) * root_quaternion).sum(dim=-1, keepdim=True) < 0
    roll = root_euler[..., :1] + torch.where(flip, root_euler.new_tensor(2.0 * torch.pi), root_euler.new_tensor(0.0))
    root_channels = torch.cat((q[..., :3], roll, root_euler[..., 1:]), dim=-1)
    root_parameters = root_channels @ torch.linalg.inv(rig.root_transform).mT
    model_parameters = torch.cat((root_parameters, q.new_zeros((*batch, PARAMETER_COUNT - ROOT_COUNT))), dim=-1)
    model_parameters = model_parameters.index_copy(-1, rig.pose_columns, q[..., 7:] / rig.host_weights)
    proportions = identity.params["proportions"].expand(*batch, len(rig.proportion_columns))
    model_parameters = model_parameters.index_copy(-1, rig.proportion_columns, proportions)
    expression = None
    if "expression" in spec.streams:
        value = (stream or {}).get("expression")
        expression = (
            q.new_zeros((*batch, EXPRESSION_COUNT)) if value is None else value.expand(*batch, EXPRESSION_COUNT)
        )
    return MHRClassic(coeffs.expand(*batch, IDENTITY_COUNT), model_parameters, expression)


__all__ = ["MHRClassic", "MhrRig", "mhr", "mhr_from_classic", "mhr_to_classic"]
