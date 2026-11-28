"""MHR (Momentum Human Rig) family loader.

``mhr(...)`` returns a :class:`~better_human.FamilySpec` with an :class:`MHRRig`. Its identity group
``identity`` takes the 45 upstream ``identity_coeffs``; ``proportions`` (a zero mesh basis) carries the 73 proportion
parameters to the rig. Upstream ``model_parameters`` enter through ``better_human.from_classic`` with an
:class:`~better_human.families.MHRClassic`.
"""

from __future__ import annotations

from pathlib import Path

import torch
from better_robot.exceptions import ArgumentError
from better_robot.lie.checks import check_dtype

from better_human.core.spec import FamilySpec, MLPCorrective, Parts
from better_human.io import FormatError, weight_path
from better_human.io.load import load_model_npz
from better_human.io.mass_data import shipped_density_tables
from better_human.io.schema import MHR_CORRECTIVE_FIELDS, MHR_SCHEMA
from better_human.mass import segment_vertices

from ._checks import check_flags
from .classic import MHRFormat
from .mhr_rig import MHRRig

LODS = tuple(range(7))
#: The density category of each part, by its joint. The categories are SMPL's part names, each
#: named after the joint at its proximal end, so a part takes the SMPL part that covers the same body segment; the one
#: spine part (c_spine0 to c_spine3) takes the lowest, spine1, and the hand part (wrist_twist) the palm, wrist.
_DENSITY_CATEGORIES = {
    "root": "pelvis",
    "c_spine0": "spine1",
    "c_neck": "neck",
    "c_head": "head",
    "c_jaw": "head",
    **{
        f"{prefix}_{segment}": f"{side}_{category}"
        for prefix, side in (("l", "left"), ("r", "right"))
        for segment, category in (
            ("upleg", "hip"),
            ("lowleg", "knee"),
            ("foot", "ankle"),
            ("clavicle", "collar"),
            ("uparm", "shoulder"),
            ("lowarm", "elbow"),
            ("wrist_twist", "wrist"),
        )
    },
}


def mhr(
    path: str | Path | None = None,
    *,
    lod: int = 1,
    expression: bool = True,
    correctives: bool = True,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> FamilySpec:
    """The MHR spec of ``MHR/converted/mhr_lod{lod}.npz`` (``tools/convert/mhr``): 127 joints, ``identity (V, 3, 45)``.

    ``proportions (V, 3, 73)`` is a zero mesh basis whose values reach the rig. ``expression=False`` drops the
    ``expression (V, 3, 72)`` stream, ``correctives=False`` the pose-corrective MLP. The spec is built
    once in ``dtype`` on ``device``.
    """
    if not isinstance(lod, int) or isinstance(lod, bool) or lod not in LODS:
        raise ArgumentError(f"lod must be one of {LODS}, got {lod!r}")
    check_flags(expression=expression, correctives=correctives)
    check_dtype(dtype)
    device = torch.device("cpu" if device is None else device)
    excluded = set() if correctives else set(MHR_CORRECTIVE_FIELDS)
    if not expression:
        excluded.add("expression_basis")
    path = weight_path("MHR", f"converted/mhr_lod{lod}.npz", path=path)
    arrays = load_model_npz(path, MHR_SCHEMA, exclude_fields=excluded)
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
    try:
        categories = tuple(_DENSITY_CATEGORIES[joint_names[joint]] for joint in part_joints)
    except KeyError as error:
        raise FormatError(
            f"{path}: the part of joint {error.args[0]!r} has no MHR density category; "
            "pass an archive tools/convert/mhr wrote from MHR"
        ) from None
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
        classic=MHRFormat(),
        up_axis=str(arrays["up_axis"]),
        provenance={"archive": str(arrays["provenance"])},
    )


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


__all__ = ["mhr"]
