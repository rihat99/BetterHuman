"""One loader for the SMPL family (SMPL, SMPL-H, SMPL-X, MANO, FLAME): a weight file plus a table."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import torch

from better_human.core.mass import segment_vertices
from better_human.core.reduction import slice_parameter_basis
from better_human.core.skinning import top_k_skin
from better_human.core.spec import FamilySpec, Landmarks, LinearCorrective, Parts
from better_human.io import weight_path
from better_human.io.load import load_model_npz
from better_human.io.schema import FLAME_SCHEMA, MANO_SCHEMA, SMPL_SCHEMA, SMPLH_SCHEMA, SMPLX_SCHEMA

from . import tables
from .regressed_rig import RegressedRig, _JointData, _reduce_joints
from .assets import load_closed_segmentation

SMPLX_SHAPE_DIM, SMPLX_EXPRESSION_DIM = 300, 100
FLAME_SHAPE_DIM, FLAME_EXPRESSION_DIM = 300, 100
HAND_DIM = 45


def _gender(gender: str) -> str:
    normalized = gender.lower()
    if normalized not in {"female", "male", "neutral"}:
        raise ValueError("gender must be 'female', 'male' or 'neutral'")
    return normalized


def _check_width(name: str, value: int, limit: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1 or value > limit:
        raise ValueError(f"{name} must be in [1, {limit}], got {value!r}")
    return value


def parents_from_kintree(kintree_table: torch.Tensor) -> tuple[int, ...]:
    """Resolve a two-row native-id kintree table to column parent indices."""
    joint_ids = [int(value) for value in kintree_table[1].tolist()]
    columns_by_id = {joint_id: column for column, joint_id in enumerate(joint_ids)}
    parents = [-1]
    for column in range(1, len(joint_ids)):
        parent_id = int(kintree_table[0, column])
        if parent_id not in columns_by_id:
            raise ValueError(f"kintree parent id {parent_id} for joint column {column} does not exist")
        parents.append(columns_by_id[parent_id])
    return tuple(parents)


def load_family(
    table: tables.FamilyTable,
    arrays: Mapping[str, torch.Tensor],
    *,
    identity: Mapping[str, int],
    streams: Mapping[str, int] = {},
    columns: Mapping[str, slice] | None = None,
    drop: Sequence[str] = (),
    meta: Mapping[str, object] = {},
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> FamilySpec:
    """Build a :class:`FamilySpec` from the ``arrays`` of an SMPL-family ``.npz`` (``io.load.load_model_npz``).

    ``identity`` and ``streams`` give the requested width of each blend group; ``columns`` says
    where each group lives in ``shapedirs`` (default: one group ``betas`` spanning the file).
    ``drop`` names terminal joints to remove (hands, face). The arrays are reduced in float64, then the spec is
    built once in ``dtype`` on ``device``; ``meta`` (the hand pose PCA) must already be there.
    """
    joint_count = arrays["weights"].shape[-1]
    if joint_count != len(table.joint_names):
        raise ValueError(f"{table.name} weights hold {joint_count} joints; expected {len(table.joint_names)}")
    device = torch.device("cpu" if device is None else device)
    shapedirs = arrays["shapedirs"]
    columns = columns or {"betas": slice(0, shapedirs.shape[-1])}
    bases: dict[str, torch.Tensor] = {}
    for name, width in (*identity.items(), *streams.items()):
        basis, _ = slice_parameter_basis(shapedirs, columns, {name: width})
        bases[name] = basis.to(device=device, dtype=dtype)
    posedirs = arrays["posedirs"].to(torch.float64)
    weights, indices = top_k_skin(arrays["weights"].to(torch.float64))
    closed_faces = None
    if table.closed_segmentation_file is not None:
        closed_faces = load_closed_segmentation(table.closed_segmentation_file, device=device)
    data = _reduce_joints(
        _JointData(
            table.joint_names,
            parents_from_kintree(arrays["kintree_table"]),
            torch.arange(joint_count),
            arrays["J_regressor"].to(torch.float64),
            weights,
            indices,
            posedirs.reshape(-1, posedirs.shape[-1]).T.contiguous(),
            Parts(segment_vertices(weights, indices), tuple(range(joint_count))),
        ),
        [table.joint_names.index(name) for name in drop],
    )
    landmarks = None
    if table.keypoint_vertex_ids:
        landmarks = Landmarks(table.keypoint_names, torch.tensor(table.keypoint_vertex_ids, device=device))
    return FamilySpec(
        name=table.name,
        template_vertices=arrays["v_template"].to(device=device, dtype=dtype),
        faces=arrays["f"].to(device=device, dtype=torch.long),
        joint_names=data.joint_names,
        parents=data.parents,
        file_joint_ids=data.file_joint_ids.to(device),
        skin_weights=data.skin_weights.to(device=device, dtype=dtype),
        skin_indices=data.skin_indices.to(device),
        identity_bases={name: bases[name] for name in identity},
        stream_bases={name: bases[name] for name in streams},
        corrective=LinearCorrective(data.posedirs.to(device=device, dtype=dtype)),
        rig=RegressedRig(data.joint_regressor.to(device=device, dtype=dtype)),
        parts=Parts(data.parts.ids.to(device), data.parts.joints, closed_faces),
        landmarks=landmarks,
        meta=meta,
    )


def smpl(
    path: str | Path | None = None,
    *,
    gender: str = "neutral",
    num_betas: int = 10,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> FamilySpec:
    """SMPL (24 joints, 6890 vertices)."""
    gender = _gender(gender)
    path = weight_path(tables.SMPL.folder, tables.SMPL.file_pattern.format(gender=gender.upper()), path=path)
    return load_family(
        tables.SMPL,
        load_model_npz(path, SMPL_SCHEMA),
        identity={"betas": _check_width("num_betas", num_betas, 300)},
        dtype=dtype,
        device=device,
    )


def smplh(
    path: str | Path | None = None,
    *,
    gender: str = "neutral",
    num_betas: int = 16,
    hands: bool = True,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> FamilySpec:
    """SMPL+H (52 joints; the hand pose is given per joint, the file ships no hand PCA)."""
    gender = _gender(gender)
    table = tables.SMPLH
    path = weight_path(table.folder, table.file_pattern.format(gender_lower=gender), path=path)
    drop = () if hands else tuple(table.joint_names[j] for j in (*table.left_hand_joints, *table.right_hand_joints))
    return load_family(
        table,
        load_model_npz(path, SMPLH_SCHEMA),
        identity={"betas": _check_width("num_betas", num_betas, 300)},
        drop=drop,
        dtype=dtype,
        device=device,
    )


def smplx(
    path: str | Path | None = None,
    *,
    gender: str = "neutral",
    num_betas: int = 16,
    num_expression: int = 10,
    hands: bool = True,
    face: bool = True,
    flat_hand_mean: bool = False,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> FamilySpec:
    """SMPL-X (55 joints, 10475 vertices). Expression is a stream group; ``face=False`` drops it and the face joints."""
    gender = _gender(gender)
    table = tables.SMPLX
    path = weight_path(table.folder, table.file_pattern.format(gender=gender.upper()), path=path)
    arrays = load_model_npz(path, SMPLX_SCHEMA)

    def floats(name: str) -> torch.Tensor:
        return arrays[name].to(device=device, dtype=dtype)

    hand_meta = {
        "hand_components": {"left": floats("hands_componentsl"), "right": floats("hands_componentsr")},
        "hand_mean": {
            side: torch.zeros_like(mean) if flat_hand_mean else mean
            for side, mean in (("left", floats("hands_meanl")), ("right", floats("hands_meanr")))
        },
    }
    drop = []
    if not face:
        drop += [table.joint_names[j] for j in table.face_joints]
    if not hands:
        drop += [table.joint_names[j] for j in (*table.left_hand_joints, *table.right_hand_joints)]
    hand_meta["hand_joints"] = {"left", "right"} if hands else set()
    streams = {"expression": _check_width("num_expression", num_expression, SMPLX_EXPRESSION_DIM)} if face else {}
    return load_family(
        table,
        arrays,
        identity={"betas": _check_width("num_betas", num_betas, SMPLX_SHAPE_DIM)},
        streams=streams,
        columns={
            "betas": slice(0, SMPLX_SHAPE_DIM),
            "expression": slice(SMPLX_SHAPE_DIM, SMPLX_SHAPE_DIM + SMPLX_EXPRESSION_DIM),
        },
        drop=drop,
        meta=hand_meta,
        dtype=dtype,
        device=device,
    )


def mano(
    path: str | Path | None = None,
    *,
    side: str = "right",
    num_betas: int = 10,
    flat_hand_mean: bool = False,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> FamilySpec:
    """MANO hand (16 joints, 778 vertices) with its PCA pose space in ``meta``."""
    if side not in {"left", "right"}:
        raise ValueError("side must be 'left' or 'right'")
    table = tables.MANO
    path = weight_path(table.folder, table.file_pattern.format(side=side.upper()), path=path)
    arrays = load_model_npz(path, MANO_SCHEMA)
    mean = arrays["hands_mean"].to(device=device, dtype=dtype)
    meta = {
        "side": side,
        "hand_components": {side: arrays["hands_components"].to(device=device, dtype=dtype)},
        "hand_mean": {side: torch.zeros_like(mean) if flat_hand_mean else mean},
    }
    return load_family(
        table,
        arrays,
        identity={"betas": _check_width("num_betas", num_betas, 10)},
        meta=meta,
        dtype=dtype,
        device=device,
    )


def flame(
    path: str | Path | None = None,
    *,
    gender: str = "generic",
    num_betas: int = 100,
    num_expression: int = 50,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> FamilySpec:
    """FLAME head (5 joints, 5023 vertices); expression is a stream group like SMPL-X."""
    table = tables.FLAME
    path = weight_path(table.folder, table.file_pattern.format(gender=gender.lower()), path=path)
    return load_family(
        table,
        load_model_npz(path, FLAME_SCHEMA),
        identity={"betas": _check_width("num_betas", num_betas, FLAME_SHAPE_DIM)},
        streams={"expression": _check_width("num_expression", num_expression, FLAME_EXPRESSION_DIM)},
        columns={
            "betas": slice(0, FLAME_SHAPE_DIM),
            "expression": slice(FLAME_SHAPE_DIM, FLAME_SHAPE_DIM + FLAME_EXPRESSION_DIM),
        },
        dtype=dtype,
        device=device,
    )


__all__ = ["flame", "load_family", "mano", "parents_from_kintree", "smpl", "smplh", "smplx"]
