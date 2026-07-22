"""Translate MHR archive rows into the BetterRobot kinematic tree.

Several authored spine rotation rows stack into one independent revolute joint
plus mimic joints.  Static FBX inverse binds remain authoritative for skinning;
this module only builds the pose tree and its archive-derived index bindings.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math

import numpy as np
import torch
from better_robot import Model, ModelBuilder
from better_robot.io import build_model
from better_robot.lie import so3

from . import archive


@dataclass(frozen=True)
class _TreeBinding:
    robot: Model
    native_pose_joint_indices: torch.Tensor
    native_origin_joint_indices: torch.Tensor
    pose_parameter_indices: torch.Tensor
    proportion_parameter_indices: torch.Tensor
    pose_host_weights: torch.Tensor
    corrective_full_q_indices: torch.Tensor
    corrective_euler_transform: torch.Tensor
    q_from_classic_permutation: torch.Tensor
    q_to_classic_permutation: torch.Tensor


def origin(translation: torch.Tensor, quaternion: torch.Tensor) -> torch.Tensor:
    return torch.cat((translation, so3.normalize(quaternion)), dim=-1)


def build_mhr_tree(  # noqa: PLR0912, PLR0915 - one audited archive-to-tree boundary
    arrays: Mapping[str, torch.Tensor | np.ndarray],
    dtype: torch.dtype,
    device: torch.device,
) -> _TreeBinding:
    joint_names = archive.strings(arrays, "joint_names")
    parameter_names = archive.strings(arrays, "model_parameter_names")
    parents = tuple(int(value) for value in archive.tensor(arrays, "joint_parents").tolist())
    offsets = archive.tensor(arrays, "joint_translation_offsets")
    pre_rotations = so3.normalize(archive.tensor(arrays, "joint_pre_rotations"))
    rows = [int(value) for value in archive.tensor(arrays, "parameter_transform_rows").tolist()]
    columns = [int(value) for value in archive.tensor(arrays, "parameter_transform_columns").tolist()]
    weights = [float(value) for value in archive.tensor(arrays, "parameter_transform_values").tolist()]
    lower_limits = [float(value) for value in archive.tensor(arrays, "model_parameter_lower_limits").tolist()]
    upper_limits = [float(value) for value in archive.tensor(arrays, "model_parameter_upper_limits").tolist()]
    pose_parameter_indices, proportion_parameter_indices = archive.parameter_partitions(arrays)

    rotation_entries_by_row: dict[int, list[tuple[int, int, float]]] = {}
    rotation_entries_by_column: dict[int, list[tuple[int, int, float]]] = {}
    for entry_index, (row, column, weight) in enumerate(zip(rows, columns, weights, strict=True)):
        joint_index, channel = divmod(row, archive.JOINT_CHANNEL_COUNT)
        if joint_index < 2 or channel not in archive.ROTATION_CHANNELS:
            continue
        entry = (entry_index, column, weight)
        rotation_entries_by_row.setdefault(row, []).append(entry)
        rotation_entries_by_column.setdefault(column, []).append((entry_index, row, weight))

    host_entries: dict[int, tuple[int, float]] = {}
    for column in pose_parameter_indices:
        candidates = rotation_entries_by_column.get(column, [])
        if not candidates:
            raise ValueError(f"MHR pose parameter {parameter_names[column]!r} has no non-root rotation entry")
        host = next((candidate for candidate in candidates if candidate[2] == 1.0), candidates[0])
        host_entries[column] = (host[0], host[2])

    builder = ModelBuilder("mhr")
    root_body = builder.add_body(joint_names[1], mass=0.0)
    root_joint_name = builder.add_free_flyer_root(
        "root",
        child=root_body,
        origin=origin(offsets[1], pre_rotations[1]),
    )
    terminal_bodies: list[str] = ["world", root_body]
    terminal_joint_names: list[str] = ["universe", root_joint_name]
    origin_joint_names: list[str] = [root_joint_name]
    entry_joint_names: dict[int, str] = {}
    identity = offsets.new_tensor((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0))
    helper_by_channel = {
        3: builder.add_revolute_x,
        4: builder.add_revolute_y,
        5: builder.add_revolute_z,
    }

    for native_index in range(2, len(joint_names)):
        parent_body = terminal_bodies[parents[native_index]]
        placement = origin(offsets[native_index], pre_rotations[native_index])
        entries: list[tuple[int, int, int, float]] = []
        for channel in (5, 4, 3):
            row = native_index * archive.JOINT_CHANNEL_COUNT + channel
            entries.extend(
                (channel, entry_index, column, weight)
                for entry_index, column, weight in sorted(rotation_entries_by_row.get(row, ()))
            )

        if not entries:
            child = builder.add_body(joint_names[native_index], mass=0.0)
            joint_name = f"mhr_fixed_{joint_names[native_index]}"
            builder.add_fixed(joint_name, parent=parent_body, child=child, origin=placement)
            origin_joint_names.append(joint_name)
            terminal_bodies.append(child)
            terminal_joint_names.append(joint_name)
            continue

        first_joint_name = ""
        for local_index, (channel, entry_index, column, weight) in enumerate(entries):
            is_last = local_index == len(entries) - 1
            child_name = joint_names[native_index] if is_last else f"mhr_node_{native_index}_{local_index}"
            child = builder.add_body(child_name, mass=0.0)
            host_entry, host_weight = host_entries[column]
            is_host = entry_index == host_entry
            joint_name = parameter_names[column] if is_host else f"mhr_mimic_{entry_index}_{parameter_names[column]}"
            kwargs: dict[str, object] = {
                "parent": parent_body,
                "child": child,
                "origin": placement if local_index == 0 else identity,
            }
            if not is_host:
                kwargs.update(
                    mimic_source=parameter_names[column],
                    mimic_multiplier=weight / host_weight,
                    mimic_offset=0.0,
                )
            else:
                scaled_limits = sorted((lower_limits[column] * host_weight, upper_limits[column] * host_weight))
                kwargs.update(
                    lower=scaled_limits[0] if math.isfinite(scaled_limits[0]) else None,
                    upper=scaled_limits[1] if math.isfinite(scaled_limits[1]) else None,
                )
            helper_by_channel[channel](joint_name, **kwargs)
            if local_index == 0:
                first_joint_name = joint_name
            entry_joint_names[entry_index] = joint_name
            parent_body = child

        origin_joint_names.append(first_joint_name)
        terminal_bodies.append(parent_body)
        terminal_joint_names.append(joint_name)

    robot = build_model(
        builder.finalize(),
        preserve_joint_order=True,
        dtype=dtype,
        device=device,
    )
    if (robot.nq, robot.nv, robot.nq_full, robot.nv_full) != (132, 131, 179, 178):
        raise ValueError(
            "Unexpected MHR reduced/full dimensions: "
            f"nq={robot.nq}, nv={robot.nv}, nq_full={robot.nq_full}, nv_full={robot.nv_full}"
        )

    native_pose_joint_indices = torch.tensor(
        [0, *(robot.joint_id(name) for name in terminal_joint_names[1:])],
        dtype=torch.long,
        device=device,
    )
    native_origin_joint_indices = torch.tensor(
        [robot.joint_id(name) for name in origin_joint_names],
        dtype=torch.long,
        device=device,
    )
    pose_host_weights = torch.tensor(
        [host_entries[index][1] for index in pose_parameter_indices],
        dtype=dtype,
        device=device,
    )
    pose_names = tuple(parameter_names[index] for index in pose_parameter_indices)
    q_from_classic, _ = robot.q_permutation(("root", *pose_names))

    corrective_full_q_indices: list[int] = []
    corrective_euler_indices: list[int] = []
    for entry_index, row in enumerate(rows):
        native_index, channel = divmod(row, archive.JOINT_CHANNEL_COUNT)
        if native_index < 2 or channel not in archive.ROTATION_CHANNELS:
            continue
        joint_id = robot.joint_id(entry_joint_names[entry_index])
        corrective_full_q_indices.append(robot.idx_qs_full[joint_id])
        corrective_euler_indices.append((native_index - 2) * 3 + channel - 3)

    corrective_euler_index_tensor = torch.tensor(
        corrective_euler_indices,
        dtype=torch.long,
        device=device,
    )
    corrective_euler_transform = torch.nn.functional.one_hot(
        corrective_euler_index_tensor,
        num_classes=125 * 3,
    ).mT.to(dtype=dtype)

    return _TreeBinding(
        robot=robot,
        native_pose_joint_indices=native_pose_joint_indices,
        native_origin_joint_indices=native_origin_joint_indices,
        pose_parameter_indices=torch.tensor(pose_parameter_indices, dtype=torch.long, device=device),
        proportion_parameter_indices=torch.tensor(
            proportion_parameter_indices,
            dtype=torch.long,
            device=device,
        ),
        pose_host_weights=pose_host_weights,
        corrective_full_q_indices=torch.tensor(
            corrective_full_q_indices,
            dtype=torch.long,
            device=device,
        ),
        corrective_euler_transform=corrective_euler_transform,
        q_from_classic_permutation=q_from_classic,
        q_to_classic_permutation=torch.argsort(q_from_classic),
    )


def accumulated_scales(local_log_scales: torch.Tensor, parents: tuple[int, ...]) -> torch.Tensor:
    accumulated: list[torch.Tensor] = []
    for joint, parent in enumerate(parents):
        local_scale = torch.exp2(local_log_scales[..., joint])
        accumulated.append(local_scale if parent < 0 else accumulated[parent] * local_scale)
    return torch.stack(accumulated, dim=-1)
