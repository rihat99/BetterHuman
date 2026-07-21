"""Momentum Human Rig loading, deformation policies, and classic conversion."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
import math
from pathlib import Path

import numpy as np
import torch
from better_robot import Data, Model, ModelBuilder
from better_robot.data_model.reduced_coordinates import expand_configuration
from better_robot.io import build_model
from better_robot.lie import se3, so3

import better_robot as br
from better_human.core import BodyStructure, BodyValues, ParametricBody
from better_human.core.classic import broadcast_configuration, classic_tensor
from better_human.core.mass import body_inertias_from_shape, build_part_topology, segment_vertices
from better_human.core.pipeline import (
    blend_shapes,
    dense_weight_from_sparse,
    lbs,
    mlp_corrective,
    pose_feature_6d_from_euler,
    skinning_transforms_from_inverse,
)
from better_human.io import MHR_CORRECTIVE_FIELDS, MHR_SCHEMA, load_model_npz, resolve_model_path

from .config_mhr import MHR_CONFIG
from .registry import register
from .smpl_family import density_tensor

JOINT_CHANNEL_COUNT = 7
TRANSLATION_CHANNELS = (0, 1, 2)
ROTATION_CHANNELS = (3, 4, 5)
SCALE_CHANNEL = 6
ROOT_PARAMETER_COUNT = 6
IDENTITY_COUNT = 45
EXPRESSION_COUNT = 72
MODEL_PARAMETER_COUNT = 204


@dataclass(frozen=True)
class MHRClassic:
    """Native MHR inputs with arbitrary leading batch dimensions."""

    #: Identity blendshape coefficients with shape ``(..., 45)``.
    identity_coeffs: torch.Tensor
    #: Native compact model parameters with shape ``(..., 204)``.
    model_parameters: torch.Tensor
    #: Optional facial expression coefficients with shape ``(..., 72)``.
    expression: torch.Tensor | None = None


@dataclass(frozen=True)
class _TreeBinding:
    robot: Model
    native_pose_joint_indices: torch.Tensor
    native_origin_joint_indices: torch.Tensor
    pose_parameter_indices: torch.Tensor
    proportion_parameter_indices: torch.Tensor
    pose_host_weights: torch.Tensor
    corrective_full_q_indices: torch.Tensor
    corrective_euler_indices: torch.Tensor
    corrective_euler_transform: torch.Tensor
    q_from_classic_permutation: torch.Tensor
    q_to_classic_permutation: torch.Tensor


def _tensor(arrays: Mapping[str, torch.Tensor | np.ndarray], name: str) -> torch.Tensor:
    value = arrays[name]
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"MHR field {name!r} must be numeric")
    return value


def _strings(arrays: Mapping[str, torch.Tensor | np.ndarray], name: str) -> tuple[str, ...]:
    value = arrays[name]
    if not isinstance(value, np.ndarray):
        raise TypeError(f"MHR field {name!r} must be Unicode metadata")
    return tuple(str(item) for item in value.tolist())


def _string_scalar(arrays: Mapping[str, torch.Tensor | np.ndarray], name: str) -> str:
    value = arrays[name]
    if not isinstance(value, np.ndarray) or value.shape != ():
        raise TypeError(f"MHR field {name!r} must be scalar Unicode metadata")
    return str(value.item())


def _validate_metadata(
    arrays: Mapping[str, torch.Tensor | np.ndarray],
    lod: int,
) -> None:
    expected_scalars = {
        "format_version": 1,
        "lod": lod,
        "world_joint_index": 0,
        "root_joint_index": 1,
    }
    for name, expected in expected_scalars.items():
        actual = int(_tensor(arrays, name))
        if actual != expected:
            raise ValueError(f"MHR field {name!r} must be {expected}, got {actual}")

    expected_strings = {
        "storage_unit": "meter",
        "native_pose_format": MHR_CONFIG.native_pose_format,
        "quaternion_order": "xyzw",
        "euler_composition_order": "Rz_Ry_Rx",
        "scale_encoding": "exp2",
        "up_axis": MHR_CONFIG.up_axis,
        "forward_axis": MHR_CONFIG.forward_axis,
        "coordinate_axis": "+x",
    }
    for name, expected in expected_strings.items():
        actual = _string_scalar(arrays, name)
        if actual != expected:
            raise ValueError(f"MHR field {name!r} must be {expected!r}, got {actual!r}")

    parents = _tensor(arrays, "joint_parents")
    if int(parents[0]) != -1 or bool((parents[1:] >= torch.arange(1, len(parents))).any()):
        raise ValueError("MHR joint parents must be stored in parent-before-child order")


def _parameter_partitions(
    arrays: Mapping[str, torch.Tensor | np.ndarray],
) -> tuple[list[int], list[int]]:
    channel_types = _tensor(arrays, "model_parameter_channel_types").tolist()
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


def _origin(translation: torch.Tensor, quaternion: torch.Tensor) -> torch.Tensor:
    return torch.cat((translation, so3.normalize(quaternion)), dim=-1)


def _build_mhr_tree(  # noqa: PLR0912, PLR0915 - one audited archive-to-tree boundary
    arrays: Mapping[str, torch.Tensor | np.ndarray],
    dtype: torch.dtype,
    device: torch.device,
) -> _TreeBinding:
    joint_names = _strings(arrays, "joint_names")
    parameter_names = _strings(arrays, "model_parameter_names")
    parents = tuple(int(value) for value in _tensor(arrays, "joint_parents").tolist())
    offsets = _tensor(arrays, "joint_translation_offsets")
    pre_rotations = so3.normalize(_tensor(arrays, "joint_pre_rotations"))
    rows = [int(value) for value in _tensor(arrays, "parameter_transform_rows").tolist()]
    columns = [int(value) for value in _tensor(arrays, "parameter_transform_columns").tolist()]
    weights = [float(value) for value in _tensor(arrays, "parameter_transform_values").tolist()]
    lower_limits = [float(value) for value in _tensor(arrays, "model_parameter_lower_limits").tolist()]
    upper_limits = [float(value) for value in _tensor(arrays, "model_parameter_upper_limits").tolist()]
    pose_parameter_indices, proportion_parameter_indices = _parameter_partitions(arrays)

    rotation_entries_by_row: dict[int, list[tuple[int, int, float]]] = {}
    rotation_entries_by_column: dict[int, list[tuple[int, int, float]]] = {}
    for entry_index, (row, column, weight) in enumerate(zip(rows, columns, weights, strict=True)):
        joint_index, channel = divmod(row, JOINT_CHANNEL_COUNT)
        if joint_index < 2 or channel not in ROTATION_CHANNELS:
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
        origin=_origin(offsets[1], pre_rotations[1]),
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
        placement = _origin(offsets[native_index], pre_rotations[native_index])
        entries: list[tuple[int, int, int, float]] = []
        for channel in (5, 4, 3):
            row = native_index * JOINT_CHANNEL_COUNT + channel
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
        native_index, channel = divmod(row, JOINT_CHANNEL_COUNT)
        if native_index < 2 or channel not in ROTATION_CHANNELS:
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
        corrective_euler_indices=corrective_euler_index_tensor,
        corrective_euler_transform=corrective_euler_transform,
        q_from_classic_permutation=q_from_classic,
        q_to_classic_permutation=torch.argsort(q_from_classic),
    )


def _parameter_bases(
    arrays: Mapping[str, torch.Tensor | np.ndarray],
    proportion_parameter_indices: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rows = _tensor(arrays, "parameter_transform_rows").tolist()
    columns = _tensor(arrays, "parameter_transform_columns").tolist()
    weights = _tensor(arrays, "parameter_transform_values").tolist()
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


def _segmentation(
    arrays: Mapping[str, torch.Tensor | np.ndarray],
    skinning_weights: torch.Tensor,
    skinning_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    group_names = _strings(arrays, "segmentation_group_names")
    try:
        zero_group = group_names.index(MHR_CONFIG.zero_weight_group)
    except ValueError as error:
        raise ValueError("MHR segmentation has no zero_weight group") from error
    physical_groups = [index for index in range(len(group_names)) if index != zero_group]
    joint_groups = _tensor(arrays, "segmentation_joint_group_indices").to(
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


def _accumulated_scales(local_log_scales: torch.Tensor, parents: tuple[int, ...]) -> torch.Tensor:
    accumulated: list[torch.Tensor] = []
    for joint, parent in enumerate(parents):
        local_scale = torch.exp2(local_log_scales[..., joint])
        accumulated.append(local_scale if parent < 0 else accumulated[parent] * local_scale)
    return torch.stack(accumulated, dim=-1)


def _bake_mhr_shape(
    body: ParametricBody,
    expanded_parameters: Mapping[str, torch.Tensor],
) -> tuple[BodyValues, Model]:
    identity = expanded_parameters["identity"]
    proportions = expanded_parameters["proportions"]
    v_shaped = body.values.v_template + blend_shapes(identity, body._identity_basis)
    if body._use_expression:
        v_shaped = v_shaped + blend_shapes(expanded_parameters["expression"], body._expression_basis)

    translation_delta = torch.einsum("...p,jcp->...jc", proportions, body._translation_basis)
    local_log_scales = torch.einsum("...p,jp->...j", proportions, body._scale_basis)
    joint_scales = _accumulated_scales(local_log_scales, body.structure.parents)
    parent_indices = torch.tensor(body.structure.parents[1:], dtype=torch.long, device=proportions.device)
    parent_scales = joint_scales.index_select(-1, parent_indices)
    translations = body._joint_offsets[1:] + translation_delta[..., 1:, :]
    translations = translations * parent_scales[..., :, None]
    quaternions = body._joint_pre_rotations[1:].expand(*translations.shape[:-1], 4)
    native_placements = torch.cat((translations, quaternions), dim=-1)

    base_placements = body._base_robot.joint_placements
    joint_placements = base_placements.expand(
        *native_placements.shape[:-2],
        *base_placements.shape,
    )
    joint_placements = joint_placements.index_copy(
        -2,
        body._native_origin_joint_indices,
        native_placements,
    )
    robot = body._base_robot.with_values(joint_placements=joint_placements)
    neutral_data = br.forward_kinematics(robot, robot.q_neutral)
    native_poses = neutral_data.joint_pose_world.index_select(
        -2,
        body.structure.native_pose_joint_indices,
    )

    body_inertias = None
    if body.structure.part_topology is not None:
        transforms = skinning_transforms_from_inverse(native_poses, body.values.inverse_bind_poses)
        neutral_vertices = lbs(
            v_shaped,
            transforms,
            body.values.skinning_weights,
            body.structure.skinning_indices,
            joint_scales=joint_scales,
            bind_joint_positions=body.values.bind_joint_positions,
        )
        part_poses = native_poses.index_select(-2, body._part_native_joint_indices)
        compact_inertias = body_inertias_from_shape(
            neutral_vertices,
            part_poses,
            body.structure.part_topology,
            body.values.density,
        )
        target_indices = torch.cat(
            (
                body._part_body_indices.new_zeros(1),
                body._part_body_indices,
            )
        )
        body_inertias = compact_inertias.new_zeros(
            *compact_inertias.shape[:-2],
            robot.nbodies,
            10,
        ).index_copy(-2, target_indices, compact_inertias)
        robot = robot.with_values(body_inertias=body_inertias)

    values = replace(
        body.values,
        shape_parameters=expanded_parameters,
        v_shaped=v_shaped,
        rest_joints=native_poses[..., :3],
        joint_scales=joint_scales,
        body_inertias=body_inertias,
    )
    return values, robot


def _mhr_pose_corrective(body: ParametricBody, data: Data) -> torch.Tensor:
    if body._corrective_first_layer_weight is None:
        return torch.zeros_like(body.values.v_shaped)
    q_full = expand_configuration(body.robot.structure, data.q)
    entries = q_full.index_select(-1, body._corrective_full_q_indices)
    euler = entries @ body._corrective_euler_transform.mT
    feature = pose_feature_6d_from_euler(euler.reshape(*euler.shape[:-1], 125, 3))
    return mlp_corrective(
        feature,
        body._corrective_first_layer_weight,
        body._corrective_output_weight,
    )


@register("mhr")
class MHR(ParametricBody):
    """Load one MHR LOD as a free-flyer plus 125 named revolute parameters.

    Six shared spine coordinates store their scaled first-row contribution in
    ``q`` (for example ``q[spine_twist0] = 0.01 * spine_twist0``). The classic
    converter applies and removes those archive-derived host weights.
    """

    def __init__(  # noqa: PLR0915 - one audited archive binding boundary
        self,
        model_path: str | Path | None = None,
        lod: int = MHR_CONFIG.default_lod,
        use_expression: bool = True,
        use_correctives: bool = True,
        density: float | torch.Tensor = 1000.0,
        compute_mass: bool = True,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> None:
        if not isinstance(lod, int) or isinstance(lod, bool) or lod not in MHR_CONFIG.lods:
            raise ValueError(f"lod must be one of {MHR_CONFIG.lods}, got {lod!r}")
        for name, value in (
            ("use_expression", use_expression),
            ("use_correctives", use_correctives),
            ("compute_mass", compute_mass),
        ):
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be bool")
        if dtype not in (torch.float32, torch.float64):
            raise TypeError(f"dtype must be torch.float32 or torch.float64, got {dtype}")

        target_device = torch.device("cpu" if device is None else device)
        path = resolve_model_path(
            "MHR",
            MHR_CONFIG.asset_pattern.format(lod=lod),
            model_path,
        )
        excluded = set(MHR_CORRECTIVE_FIELDS) if not use_correctives else set()
        if not use_expression:
            excluded.add("expression_basis")
        arrays = load_model_npz(path, MHR_SCHEMA, exclude_fields=excluded)
        _validate_metadata(arrays, lod)
        tree = _build_mhr_tree(arrays, dtype, target_device)

        v_template = _tensor(arrays, "template_vertices").to(device=target_device, dtype=dtype)
        faces = _tensor(arrays, "faces").to(device=target_device, dtype=torch.long)
        skinning_indices = _tensor(arrays, "skin_indices").to(device=target_device, dtype=torch.long)
        skinning_weights = _tensor(arrays, "skin_weights").to(device=target_device, dtype=dtype)
        identity_basis = _tensor(arrays, "identity_basis").to(device=target_device, dtype=dtype)
        expression_basis = (
            _tensor(arrays, "expression_basis").to(device=target_device, dtype=dtype) if use_expression else None
        )
        inverse_bind_poses = se3.from_matrix(
            _tensor(arrays, "inverse_bind_matrices").to(device=target_device, dtype=dtype)
        )
        bind_joint_positions = _tensor(arrays, "bind_matrices").to(
            device=target_device,
            dtype=dtype,
        )[..., :3, 3]

        parameter_groups: dict[str, int] = {"identity": IDENTITY_COUNT}
        if use_expression:
            parameter_groups["expression"] = EXPRESSION_COUNT
        parameter_groups["proportions"] = len(tree.proportion_parameter_indices)
        parameters = {name: v_template.new_zeros(width) for name, width in parameter_groups.items()}
        part_ids, part_native_joint_indices = _segmentation(
            arrays,
            skinning_weights,
            skinning_indices,
        )
        part_topology = build_part_topology(faces, part_ids, len(part_native_joint_indices)) if compute_mass else None
        density_value = density_tensor(density, len(part_native_joint_indices), v_template)
        part_body_indices = torch.tensor(
            [tree.robot.body_id(_strings(arrays, "joint_names")[int(index)]) for index in part_native_joint_indices],
            dtype=torch.long,
            device=target_device,
        )

        structure = BodyStructure(
            num_vertices=len(v_template),
            faces=faces,
            joint_names=_strings(arrays, "joint_names"),
            parents=tuple(int(value) for value in _tensor(arrays, "joint_parents").tolist()),
            native_joint_indices=tuple(range(len(tree.native_pose_joint_indices))),
            skinning_indices=skinning_indices,
            part_ids=part_ids,
            part_topology=part_topology,
            parameter_groups=parameter_groups,
            landmarks={},
            unit=MHR_CONFIG.unit,
            up_axis=MHR_CONFIG.up_axis,
            forward_axis=MHR_CONFIG.forward_axis,
            native_pose_format=MHR_CONFIG.native_pose_format,
            native_pose_joint_indices=tree.native_pose_joint_indices,
        )
        neutral_data = br.forward_kinematics(tree.robot, tree.robot.q_neutral)
        neutral_native_poses = neutral_data.joint_pose_world.index_select(
            -2,
            tree.native_pose_joint_indices,
        )
        values = BodyValues(
            v_template=v_template,
            shape_basis=None,
            posedirs=None,
            joint_regressor=None,
            skinning_weights=skinning_weights,
            shape_parameters=parameters,
            v_shaped=v_template,
            rest_joints=neutral_native_poses[..., :3],
            pelvis_offset=None,
            density=density_value,
            body_inertias=None,
            inverse_bind_poses=inverse_bind_poses,
            bind_joint_positions=bind_joint_positions,
            joint_scales=v_template.new_ones(len(structure.joint_names)),
        )
        translation_basis, scale_basis, root_transform = _parameter_bases(
            arrays,
            tree.proportion_parameter_indices,
            dtype,
            target_device,
        )

        first_layer_weight = None
        output_weight = None
        if use_correctives:
            expected_corrective_joints = torch.arange(2, 127, dtype=torch.long)
            if not torch.equal(_tensor(arrays, "corrective_joint_indices"), expected_corrective_joints):
                raise ValueError("MHR corrective joints must be native joints 2:127 in order")
            expected_joint_names = tuple(structure.joint_names[index] for index in range(2, 127))
            if _strings(arrays, "corrective_joint_names") != expected_joint_names:
                raise ValueError("MHR corrective joint names must match native joints 2:127")
            expected_matrix_indices = torch.tensor((0, 3, 6, 1, 4, 7), dtype=torch.long)
            if not torch.equal(_tensor(arrays, "corrective_feature_matrix_indices"), expected_matrix_indices):
                raise ValueError("MHR corrective feature indices must select the first two rotation columns")
            expected_identity_feature = torch.tensor((1, 0, 0, 0, 1, 0), dtype=torch.float32)
            if not torch.equal(_tensor(arrays, "corrective_identity_feature"), expected_identity_feature):
                raise ValueError("MHR corrective identity feature must be [1, 0, 0, 0, 1, 0]")
            first_layer_weight = dense_weight_from_sparse(
                _tensor(arrays, "corrective_first_layer_indices").to(
                    device=target_device,
                    dtype=torch.long,
                ),
                _tensor(arrays, "corrective_first_layer_weights").to(
                    device=target_device,
                    dtype=dtype,
                ),
                _tensor(arrays, "corrective_first_layer_mask").to(
                    device=target_device,
                    dtype=torch.bool,
                ),
            )
            output_weight = _tensor(arrays, "corrective_output_weights").to(
                device=target_device,
                dtype=dtype,
            )

        object.__setattr__(self, "structure", structure)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "robot", tree.robot)
        object.__setattr__(self, "_shape_bake_policy", _bake_mhr_shape)
        object.__setattr__(self, "_pose_corrective_policy", _mhr_pose_corrective)
        object.__setattr__(self, "_base_robot", tree.robot)
        object.__setattr__(self, "_identity_basis", identity_basis)
        object.__setattr__(self, "_expression_basis", expression_basis)
        object.__setattr__(self, "_use_expression", use_expression)
        object.__setattr__(
            self, "_joint_offsets", _tensor(arrays, "joint_translation_offsets").to(target_device, dtype)
        )
        object.__setattr__(
            self,
            "_joint_pre_rotations",
            so3.normalize(_tensor(arrays, "joint_pre_rotations").to(target_device, dtype)),
        )
        object.__setattr__(self, "_translation_basis", translation_basis)
        object.__setattr__(self, "_scale_basis", scale_basis)
        object.__setattr__(self, "_root_parameter_transform", root_transform)
        object.__setattr__(self, "_root_parameter_transform_inverse", torch.linalg.inv(root_transform))
        object.__setattr__(self, "_native_origin_joint_indices", tree.native_origin_joint_indices)
        object.__setattr__(self, "_pose_parameter_indices", tree.pose_parameter_indices)
        object.__setattr__(self, "_proportion_parameter_indices", tree.proportion_parameter_indices)
        object.__setattr__(self, "_pose_host_weights", tree.pose_host_weights)
        object.__setattr__(self, "_corrective_full_q_indices", tree.corrective_full_q_indices)
        object.__setattr__(self, "_corrective_euler_indices", tree.corrective_euler_indices)
        object.__setattr__(self, "_corrective_euler_transform", tree.corrective_euler_transform)
        object.__setattr__(self, "_q_from_classic_permutation", tree.q_from_classic_permutation)
        object.__setattr__(self, "_q_to_classic_permutation", tree.q_to_classic_permutation)
        object.__setattr__(self, "_part_native_joint_indices", part_native_joint_indices)
        object.__setattr__(self, "_part_body_indices", part_body_indices)
        object.__setattr__(self, "_corrective_first_layer_weight", first_layer_weight)
        object.__setattr__(self, "_corrective_output_weight", output_weight)

        baked_values, baked_robot = _bake_mhr_shape(self, parameters)
        object.__setattr__(self, "values", baked_values)
        object.__setattr__(self, "robot", baked_robot)

    def from_classic(self, classic: MHRClassic) -> tuple[MHR, torch.Tensor]:
        """Bake classic shape state and return ``(shaped_body, q)``.

        ``expression=None`` is the reference's zero-expression shorthand and
        is canonicalized to an explicit zero tensor when expression is enabled.
        """
        exemplar = self.values.v_template
        identity = classic_tensor("identity_coeffs", classic.identity_coeffs, IDENTITY_COUNT, exemplar)
        model_parameters = classic_tensor(
            "model_parameters",
            classic.model_parameters,
            MODEL_PARAMETER_COUNT,
            exemplar,
        )
        if identity is None or model_parameters is None:
            raise TypeError("identity_coeffs and model_parameters must be floating torch.Tensors")
        if not self._use_expression and classic.expression is not None:
            raise ValueError("expression requires an MHR body constructed with use_expression=True")
        expression = (
            classic_tensor("expression", classic.expression, EXPRESSION_COUNT, exemplar)
            if self._use_expression
            else None
        )
        shapes = [identity.shape[:-1], model_parameters.shape[:-1]]
        if expression is not None:
            shapes.append(expression.shape[:-1])
        batch_shape = torch.broadcast_shapes(*shapes)
        identity = identity.expand(*batch_shape, IDENTITY_COUNT)
        model_parameters = model_parameters.expand(*batch_shape, MODEL_PARAMETER_COUNT)
        shape_groups = {
            "identity": identity,
            "proportions": model_parameters.index_select(-1, self._proportion_parameter_indices),
        }
        if self._use_expression:
            shape_groups["expression"] = (
                exemplar.new_zeros((*batch_shape, EXPRESSION_COUNT))
                if expression is None
                else expression.expand(*batch_shape, EXPRESSION_COUNT)
            )
        shaped = self.with_shape(**shape_groups)

        root_parameters = model_parameters[..., :ROOT_PARAMETER_COUNT]
        root_channels = root_parameters @ self._root_parameter_transform.mT
        root = torch.cat((root_channels[..., :3], so3.from_euler(root_channels[..., 3:6])), dim=-1)
        pose = model_parameters.index_select(-1, self._pose_parameter_indices) * self._pose_host_weights
        external_q = torch.cat((root, pose), dim=-1)
        return shaped, external_q.index_select(-1, self._q_from_classic_permutation)

    def to_classic(self, q: torch.Tensor) -> MHRClassic:
        """Convert ``q`` plus this body's baked shape back to native MHR.

        Euler angles use BetterRobot's principal branch away from gimbal lock.
        A ``2*pi`` roll representative preserves a negative root-quaternion
        sign so the vector-valued ``q`` round trip remains lossless.
        """
        identity = self.values.shape_parameters["identity"]
        q, batch_shape = broadcast_configuration(q, self.robot, identity)
        external_q = q.index_select(-1, self._q_to_classic_permutation)
        root_quaternion = so3.normalize(external_q[..., 3:7])
        root_euler = so3.to_euler(root_quaternion)
        reconstructed = so3.from_euler(root_euler)
        needs_sign_flip = (reconstructed * root_quaternion).sum(dim=-1, keepdim=True) < 0
        roll = root_euler[..., :1] + torch.where(
            needs_sign_flip,
            root_euler.new_tensor(2.0 * torch.pi),
            root_euler.new_tensor(0.0),
        )
        root_euler = torch.cat((roll, root_euler[..., 1:]), dim=-1)
        root_channels = torch.cat((external_q[..., :3], root_euler), dim=-1)
        root_parameters = root_channels @ self._root_parameter_transform_inverse.mT
        pose_parameters = external_q[..., 7:] / self._pose_host_weights
        proportions = self.values.shape_parameters["proportions"].expand(*batch_shape, 73)

        model_parameters = q.new_zeros((*batch_shape, MODEL_PARAMETER_COUNT))
        root_indices = torch.arange(ROOT_PARAMETER_COUNT, dtype=torch.long, device=q.device)
        model_parameters = model_parameters.index_copy(-1, root_indices, root_parameters)
        model_parameters = model_parameters.index_copy(-1, self._pose_parameter_indices, pose_parameters)
        model_parameters = model_parameters.index_copy(
            -1,
            self._proportion_parameter_indices,
            proportions,
        )
        expression = (
            self.values.shape_parameters["expression"].expand(*batch_shape, EXPRESSION_COUNT)
            if self._use_expression
            else None
        )
        return MHRClassic(
            identity_coeffs=identity.expand(*batch_shape, IDENTITY_COUNT),
            model_parameters=model_parameters,
            expression=expression,
        )


__all__ = ["MHR", "MHRClassic"]
