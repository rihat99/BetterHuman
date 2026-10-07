"""Read and validate converted Momentum Human Rig archives.

Converted archives separate numeric tensors from Unicode metadata.  Their
joint rows are parent-before-child, and the 204 model parameters partition
into root, pose, and proportion channels according to stored channel types.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch

from better_human.core.mass import segment_vertices

from .mhr_config import MHR_CONFIG

JOINT_CHANNEL_COUNT = 7
TRANSLATION_CHANNELS = (0, 1, 2)
ROTATION_CHANNELS = (3, 4, 5)
SCALE_CHANNEL = 6
ROOT_PARAMETER_COUNT = 6
IDENTITY_COUNT = 45
EXPRESSION_COUNT = 72
MODEL_PARAMETER_COUNT = 204


def tensor(arrays: Mapping[str, torch.Tensor | np.ndarray], name: str) -> torch.Tensor:
    value = arrays[name]
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"MHR field {name!r} must be numeric")
    return value


def strings(arrays: Mapping[str, torch.Tensor | np.ndarray], name: str) -> tuple[str, ...]:
    value = arrays[name]
    if not isinstance(value, np.ndarray):
        raise TypeError(f"MHR field {name!r} must be Unicode metadata")
    return tuple(str(item) for item in value.tolist())


def string_scalar(arrays: Mapping[str, torch.Tensor | np.ndarray], name: str) -> str:
    value = arrays[name]
    if not isinstance(value, np.ndarray) or value.shape != ():
        raise TypeError(f"MHR field {name!r} must be scalar Unicode metadata")
    return str(value.item())


def validate_metadata(
    arrays: Mapping[str, torch.Tensor | np.ndarray],
    lod: int,
) -> None:
    expected_scalars = {
        "format_version": 1,
        "lod": lod,
    }
    for name, expected in expected_scalars.items():
        actual = int(tensor(arrays, name))
        if actual != expected:
            raise ValueError(f"MHR field {name!r} must be {expected}, got {actual}")

    expected_strings = {
        "up_axis": MHR_CONFIG.up_axis,
        "forward_axis": MHR_CONFIG.forward_axis,
        "coordinate_axis": "+x",
    }
    for name, expected in expected_strings.items():
        actual = string_scalar(arrays, name)
        if actual != expected:
            raise ValueError(f"MHR field {name!r} must be {expected!r}, got {actual!r}")

    parents = tensor(arrays, "joint_parents")
    if int(parents[0]) != -1 or bool((parents[1:] >= torch.arange(1, len(parents))).any()):
        raise ValueError("MHR joint parents must be stored in parent-before-child order")


def parameter_partitions(
    arrays: Mapping[str, torch.Tensor | np.ndarray],
) -> tuple[list[int], list[int]]:
    channel_types = tensor(arrays, "model_parameter_channel_types").tolist()
    pose = [index for index, kind in enumerate(channel_types) if kind == 1 and index >= ROOT_PARAMETER_COUNT]
    proportions = [
        index for index, kind in enumerate(channel_types) if kind in (0, 2) and index >= ROOT_PARAMETER_COUNT
    ]
    if len(pose) != 125 or len(proportions) != 73:
        raise ValueError(
            "MHR parameter partition must contain 125 pose and 73 proportion parameters; "
            f"got {len(pose)} and {len(proportions)}"
        )
    return pose, proportions


def parameter_bases(
    arrays: Mapping[str, torch.Tensor | np.ndarray],
    proportion_parameter_indices: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rows = tensor(arrays, "parameter_transform_rows").tolist()
    columns = tensor(arrays, "parameter_transform_columns").tolist()
    weights = tensor(arrays, "parameter_transform_values").tolist()
    proportions = proportion_parameter_indices.cpu().tolist()
    proportion_columns = {column: local for local, column in enumerate(proportions)}
    translation_basis = torch.zeros(127, 3, len(proportions), dtype=dtype, device=device)
    scale_basis = torch.zeros(127, len(proportions), dtype=dtype, device=device)
    root_transform = torch.zeros(ROOT_PARAMETER_COUNT, ROOT_PARAMETER_COUNT, dtype=dtype, device=device)
    for row_value, column_value, weight_value in zip(rows, columns, weights, strict=True):
        row = int(row_value)
        column = int(column_value)
        joint, channel = divmod(row, JOINT_CHANNEL_COUNT)
        weight = float(weight_value)
        if joint == 1 and channel < ROOT_PARAMETER_COUNT and column < ROOT_PARAMETER_COUNT:
            root_transform[channel, column] = weight
        local_column = proportion_columns.get(column)
        if local_column is None:
            continue
        if channel in TRANSLATION_CHANNELS:
            translation_basis[joint, channel, local_column] = weight
        elif channel == SCALE_CHANNEL:
            scale_basis[joint, local_column] = weight
    if torch.linalg.matrix_rank(root_transform) != ROOT_PARAMETER_COUNT:
        raise ValueError("MHR root parameter transform must be invertible")
    return translation_basis, scale_basis, root_transform


def segmentation(
    arrays: Mapping[str, torch.Tensor | np.ndarray],
    skinning_weights: torch.Tensor,
    skinning_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    group_names = strings(arrays, "segmentation_group_names")
    try:
        zero_group = group_names.index(MHR_CONFIG.zero_weight_group)
    except ValueError as error:
        raise ValueError("MHR segmentation has no zero_weight group") from error
    physical_groups = [index for index in range(len(group_names)) if index != zero_group]
    joint_groups = tensor(arrays, "segmentation_joint_group_indices").to(
        device=skinning_indices.device,
        dtype=torch.long,
    )
    strongest_joints = segment_vertices(skinning_weights, skinning_indices)
    winning_groups = joint_groups.index_select(0, strongest_joints)
    remap = torch.full((len(group_names),), -1, dtype=torch.long, device=skinning_indices.device)
    physical_group_tensor = torch.tensor(physical_groups, dtype=torch.long, device=skinning_indices.device)
    remap = remap.index_copy(
        0,
        physical_group_tensor,
        torch.arange(len(physical_groups), dtype=torch.long, device=skinning_indices.device),
    )
    part_ids = remap.index_select(0, winning_groups)
    if bool((part_ids < 0).any()):
        raise ValueError("MHR positive skin weights unexpectedly map to zero_weight")

    representative_joints = []
    for group in physical_groups:
        matches = torch.nonzero(joint_groups == group, as_tuple=False).flatten()
        if not len(matches):
            raise ValueError(f"MHR segmentation group {group_names[group]!r} contains no joints")
        representative_joints.append(int(matches[0]))
    return part_ids, torch.tensor(representative_joints, dtype=torch.long, device=skinning_indices.device)
