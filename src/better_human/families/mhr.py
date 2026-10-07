"""MHR (Momentum Human Rig) family: loader, classic parameter container and converters.

``mhr(...)`` returns a :class:`~better_human.core.spec.FamilySpec` with an :class:`AuthoredRig`.
The classic API (``MHRClassic``, ``mhr_from_classic``, ``mhr_to_classic``) lives here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from better_robot.lie import se3, so3

from better_human.core.classic import broadcast_configuration, classic_tensor
from better_human.core.pipeline import dense_weight_from_sparse
from better_human.core.spec import Corrective, FamilySpec, Parts
from better_human.io import MHR_CORRECTIVE_FIELDS, MHR_SCHEMA, load_model_npz, resolve_model_path

from . import mhr_archive as archive
from .mhr_archive import EXPRESSION_COUNT, IDENTITY_COUNT, MODEL_PARAMETER_COUNT, ROOT_PARAMETER_COUNT
from .mhr_config import MHR_CONFIG
from .mhr_rig import AuthoredRig
from .mhr_tree import build_mhr_tree

if TYPE_CHECKING:
    from better_human.core.body import Body, Identity


def mhr(
    path: str | Path | None = None,
    *,
    lod: int = MHR_CONFIG.default_lod,
    expression: bool = True,
    correctives: bool = True,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> FamilySpec:
    """Load an MHR LOD archive as a family spec (identity ``(V,3,45)``, proportions ``(V,3,73)`` of zeros)."""
    if not isinstance(lod, int) or isinstance(lod, bool) or lod not in MHR_CONFIG.lods:
        raise ValueError(f"lod must be one of {MHR_CONFIG.lods}, got {lod!r}")
    if not isinstance(expression, bool) or not isinstance(correctives, bool):
        raise TypeError("expression and correctives must be bool")
    if dtype not in (torch.float32, torch.float64):
        raise TypeError(f"dtype must be torch.float32 or torch.float64, got {dtype}")
    device = torch.device("cpu" if device is None else device)
    excluded = set() if correctives else set(MHR_CORRECTIVE_FIELDS)
    if not expression:
        excluded.add("expression_basis")
    arrays = load_model_npz(
        resolve_model_path("MHR", MHR_CONFIG.asset_pattern.format(lod=lod), path), MHR_SCHEMA, exclude_fields=excluded
    )
    archive.validate_metadata(arrays, lod)

    def floats(name: str) -> torch.Tensor:
        return archive.tensor(arrays, name).to(device=device, dtype=dtype)

    def longs(name: str) -> torch.Tensor:
        return archive.tensor(arrays, name).to(device=device, dtype=torch.long)

    tree = build_mhr_tree(arrays, dtype, device)
    template = floats("template_vertices")
    skin_indices, skin_weights = longs("skin_indices"), floats("skin_weights")
    part_ids, part_joints = archive.segmentation(arrays, skin_weights, skin_indices)
    translation_basis, scale_basis, root_transform = archive.parameter_bases(
        arrays, tree.proportion_parameter_indices, dtype, device
    )
    corrective = Corrective("none")
    if correctives:
        first_layer = dense_weight_from_sparse(
            longs("corrective_first_layer_indices"),
            floats("corrective_first_layer_weights"),
            archive.tensor(arrays, "corrective_first_layer_mask").to(device=device, dtype=torch.bool),
        )
        corrective = Corrective("mlp", first_layer=first_layer, output=floats("corrective_output_weights"))
    joint_names = archive.strings(arrays, "joint_names")
    proportions = len(tree.proportion_parameter_indices)
    return FamilySpec(
        name="mhr",
        vertices=template,
        faces=longs("faces"),
        joint_names=joint_names,
        parents=tuple(int(value) for value in archive.tensor(arrays, "joint_parents").tolist()),
        skin_weights=skin_weights,
        skin_indices=skin_indices,
        identity={
            "identity": floats("identity_basis"),
            "proportions": template.new_zeros(len(template), 3, proportions),
        },
        streams={"expression": floats("expression_basis")} if expression else {},
        corrective=corrective,
        rig=AuthoredRig(arrays, tree),
        parts=Parts(part_ids, tuple(int(joint) for joint in part_joints.tolist())),
        unit=MHR_CONFIG.unit,
        up_axis=MHR_CONFIG.up_axis,
        forward_axis=MHR_CONFIG.forward_axis,
        meta={
            "inverse_bind_poses": se3.from_matrix(floats("inverse_bind_matrices")),
            "bind_joint_positions": floats("bind_matrices")[..., :3, 3],
            "translation_basis": translation_basis,
            "scale_basis": scale_basis,
            "root_parameter_transform": root_transform,
            "joint_offsets": floats("joint_translation_offsets"),
            "joint_pre_rotations": so3.normalize(floats("joint_pre_rotations")),
            "native_joint_ids": torch.arange(len(joint_names), device=device),
        },
    )


@dataclass(frozen=True)
class MHRClassic:
    """Reference MHR parameters: ``identity_coeffs (..., 45)``, ``model_parameters (..., 204)``, ``expression (..., 72)``."""

    identity_coeffs: torch.Tensor
    model_parameters: torch.Tensor
    expression: torch.Tensor | None = None


def mhr_from_classic(body: Body, classic: MHRClassic) -> tuple[Identity, torch.Tensor, dict[str, torch.Tensor]]:
    """``(Identity, q (..., 132), stream)`` for ``body`` (a :class:`~better_human.core.body.Body` of ``mhr``)."""
    spec, tree = body.spec, body.spec.rig.tree
    exemplar = spec.vertices
    identity = classic_tensor("identity_coeffs", classic.identity_coeffs, IDENTITY_COUNT, exemplar)
    model_parameters = classic_tensor("model_parameters", classic.model_parameters, MODEL_PARAMETER_COUNT, exemplar)
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
    model_parameters = model_parameters.expand(*batch, MODEL_PARAMETER_COUNT)
    baked = body.bake(
        identity=identity, proportions=model_parameters.index_select(-1, tree.proportion_parameter_indices)
    )
    stream = {}
    if use_expression:
        zeros = exemplar.new_zeros((*batch, EXPRESSION_COUNT))
        stream["expression"] = zeros if expression is None else expression.expand(*batch, EXPRESSION_COUNT)
    root_channels = model_parameters[..., :ROOT_PARAMETER_COUNT] @ spec.meta["root_parameter_transform"].mT
    root = torch.cat((root_channels[..., :3], so3.from_euler(root_channels[..., 3:6])), dim=-1)
    pose = model_parameters.index_select(-1, tree.pose_parameter_indices) * tree.pose_host_weights
    q = torch.cat((root, pose), dim=-1).index_select(-1, tree.q_from_classic_permutation)
    return baked, q, stream


def mhr_to_classic(identity: Identity, q: torch.Tensor, stream: dict[str, torch.Tensor] | None = None) -> MHRClassic:
    """Classic parameters for a baked MHR ``identity`` and ``q (..., 132)``; Euler angles on BR's principal branch."""
    spec = identity.body.spec
    tree = spec.rig.tree
    coeffs = identity.params["identity"]
    q, batch = broadcast_configuration(q, identity.model, coeffs)
    external_q = q.index_select(-1, tree.q_to_classic_permutation)
    root_quaternion = so3.normalize(external_q[..., 3:7])
    root_euler = so3.to_euler(root_quaternion)
    flip = (so3.from_euler(root_euler) * root_quaternion).sum(dim=-1, keepdim=True) < 0
    roll = root_euler[..., :1] + torch.where(flip, root_euler.new_tensor(2.0 * torch.pi), root_euler.new_tensor(0.0))
    root_channels = torch.cat((external_q[..., :3], roll, root_euler[..., 1:]), dim=-1)
    root_parameters = root_channels @ torch.linalg.inv(spec.meta["root_parameter_transform"]).mT
    model_parameters = q.new_zeros((*batch, MODEL_PARAMETER_COUNT))
    model_parameters = model_parameters.index_copy(
        -1, torch.arange(ROOT_PARAMETER_COUNT, device=q.device), root_parameters
    )
    model_parameters = model_parameters.index_copy(
        -1, tree.pose_parameter_indices, external_q[..., 7:] / tree.pose_host_weights
    )
    proportions = identity.params["proportions"].expand(*batch, len(tree.proportion_parameter_indices))
    model_parameters = model_parameters.index_copy(-1, tree.proportion_parameter_indices, proportions)
    expression = None
    if "expression" in spec.streams:
        value = (stream or {}).get("expression")
        expression = (
            q.new_zeros((*batch, EXPRESSION_COUNT)) if value is None else value.expand(*batch, EXPRESSION_COUNT)
        )
    return MHRClassic(coeffs.expand(*batch, IDENTITY_COUNT), model_parameters, expression)


__all__ = ["AuthoredRig", "MHRClassic", "mhr", "mhr_from_classic", "mhr_to_classic"]
