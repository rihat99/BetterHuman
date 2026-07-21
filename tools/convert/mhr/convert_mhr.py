"""Convert MHR release assets into self-contained BetterHuman NPZ archives."""

from __future__ import annotations

import argparse
import logging
import re
import struct
from collections.abc import Sequence
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
import pymomentum.geometry as pym_geometry
import torch

LOD_COUNT = 7
LOD_VERTEX_COUNTS = (73639, 18439, 10661, 4899, 2461, 971, 595)
JOINT_COUNT = 127
JOINT_PARAMETER_COUNT = 7
MODEL_PARAMETER_COUNT = 204
IDENTITY_COUNT = 45
EXPRESSION_COUNT = 72
BLENDSHAPE_COUNT = IDENTITY_COUNT + EXPRESSION_COUNT
CORRECTIVE_JOINT_COUNT = 125
CORRECTIVE_FEATURE_COUNT = CORRECTIVE_JOINT_COUNT * 6
CORRECTIVE_HIDDEN_COUNT = CORRECTIVE_JOINT_COUNT * 24
CENTIMETERS_TO_METERS = np.float32(0.01)
CHANNEL_NAMES = ("tx", "ty", "tz", "rx", "ry", "rz", "sc")
TRANSLATION_CHANNELS = frozenset((0, 1, 2))
ROTATION_CHANNELS = frozenset((3, 4, 5))
SCALE_CHANNEL = 6
FBX_HEADER = b"Kaydara FBX Binary  \x00\x1a\x00"
FBX_GLOBAL_PROPERTIES = frozenset(
    (
        "UpAxis",
        "UpAxisSign",
        "FrontAxis",
        "FrontAxisSign",
        "CoordAxis",
        "CoordAxisSign",
        "UnitScaleFactor",
    )
)
ZERO_WEIGHT_PARTS = frozenset(("eye", "tongue", "teeth", "brow", "cheek", "lip", "ear", "nose", "body_world"))
HAND_PARTS = frozenset(("thumb", "index", "middle", "ring", "pinky", "wrist"))
FOOT_PARTS = frozenset(("foot", "talocrural", "subtalar", "transversetarsal", "ball"))

LOGGER = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-dir", type=Path, required=True, help="Directory containing the MHR release assets")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for mhr_lodN.npz archives")
    parser.add_argument(
        "--lod",
        action="append",
        choices=range(LOD_COUNT),
        type=int,
        help="Convert one LOD; repeat for several. Defaults to all seven.",
    )
    return parser.parse_args()


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    value = stream.read(size)
    if len(value) != size:
        raise ValueError("Unexpected end of FBX file")
    return value


def _read_fbx_property(stream: BinaryIO) -> Any:
    type_code = _read_exact(stream, 1)
    scalar_formats = {
        b"Y": "<h",
        b"C": "<?",
        b"I": "<i",
        b"F": "<f",
        b"D": "<d",
        b"L": "<q",
    }
    if type_code in scalar_formats:
        item_format = scalar_formats[type_code]
        return struct.unpack(item_format, _read_exact(stream, struct.calcsize(item_format)))[0]
    if type_code in (b"S", b"R"):
        size = struct.unpack("<I", _read_exact(stream, 4))[0]
        value = _read_exact(stream, size)
        return value.decode("utf-8") if type_code == b"S" else value
    if type_code.lower() in (b"f", b"d", b"l", b"i", b"b", b"c"):
        _, _, encoded_size = struct.unpack("<III", _read_exact(stream, 12))
        stream.seek(encoded_size, 1)
        return None
    raise ValueError(f"Unsupported FBX property type {type_code!r}")


def _read_fbx_node_header(stream: BinaryIO, version: int) -> tuple[int, int, int, str] | None:
    header_format = "<QQQB" if version >= 7500 else "<IIIB"
    header = struct.unpack(header_format, _read_exact(stream, struct.calcsize(header_format)))
    end_offset, property_count, property_size, name_size = header
    if end_offset == 0:
        return None
    name = _read_exact(stream, name_size).decode("utf-8")
    return int(end_offset), int(property_count), int(property_size), name


def _read_fbx_properties70(stream: BinaryIO, version: int, end_offset: int) -> dict[str, int | float]:
    properties: dict[str, int | float] = {}
    while stream.tell() < end_offset:
        item = _read_fbx_node_header(stream, version)
        if item is None:
            break
        item_end, item_property_count, _, item_name = item
        values = [_read_fbx_property(stream) for _ in range(item_property_count)]
        if item_name == "P" and values and values[0] in FBX_GLOBAL_PROPERTIES:
            properties[str(values[0])] = values[-1]
        stream.seek(item_end)
    return properties


def _read_fbx_global_properties(path: Path) -> dict[str, int | float]:
    with path.open("rb") as stream:
        if _read_exact(stream, len(FBX_HEADER)) != FBX_HEADER:
            raise ValueError(f"{path} is not a binary FBX file")
        version = struct.unpack("<I", _read_exact(stream, 4))[0]
        while True:
            header = _read_fbx_node_header(stream, version)
            if header is None:
                break
            end_offset, property_count, _, name = header
            for _ in range(property_count):
                _read_fbx_property(stream)
            if name != "GlobalSettings":
                stream.seek(end_offset)
                continue
            properties: dict[str, int | float] = {}
            while stream.tell() < end_offset:
                child = _read_fbx_node_header(stream, version)
                if child is None:
                    break
                child_end, child_property_count, _, child_name = child
                for _ in range(child_property_count):
                    _read_fbx_property(stream)
                if child_name != "Properties70":
                    stream.seek(child_end)
                    continue
                properties.update(_read_fbx_properties70(stream, version, child_end))
                stream.seek(child_end)
            missing = FBX_GLOBAL_PROPERTIES.difference(properties)
            if missing:
                raise ValueError(f"Missing FBX global properties in {path}: {sorted(missing)}")
            return properties
    raise ValueError(f"GlobalSettings not found in {path}")


def _signed_axis(axis: int, sign: int) -> str:
    try:
        name = ("x", "y", "z")[axis]
    except IndexError as error:
        raise ValueError(f"Invalid FBX axis index {axis}") from error
    if sign not in (-1, 1):
        raise ValueError(f"Invalid FBX axis sign {sign}")
    return f"{'+' if sign > 0 else '-'}{name}"


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    return np.asarray(value)


def _parameter_transform_statement_counts(path: Path) -> np.ndarray:
    counts = np.zeros(3, dtype=np.int64)
    section = ""
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", maxsplit=1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1]
            continue
        if section != "ParameterTransform":
            continue
        channel = line.split("=", maxsplit=1)[0].strip().rsplit(".", maxsplit=1)[-1]
        channel_index = CHANNEL_NAMES.index(channel)
        counts[0 if channel_index in TRANSLATION_CHANNELS else 1 if channel_index in ROTATION_CHANNELS else 2] += 1
    return counts


def _read_model_entries(
    path: Path,
) -> tuple[list[tuple[str, list[str]]], list[tuple[str, float, float, float]], dict[str, dict[str, float]]]:
    parameter_sets: list[tuple[str, list[str]]] = []
    limits: list[tuple[str, float, float, float]] = []
    pose_constraints: dict[str, dict[str, float]] = {}
    section = ""
    limit_pattern = re.compile(r"^limit\s+(\S+)\s+minmax\s+\[\s*([^,]+),\s*([^\]]+)\](?:\s+(\S+))?$")
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", maxsplit=1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1]
            continue
        if section == "ParameterSets":
            tokens = line.split()
            if tokens[0] != "parameterset" or len(tokens) < 3:
                raise ValueError(f"Invalid parameter-set line: {raw_line}")
            parameter_sets.append((tokens[1], tokens[2:]))
        elif section == "Limits":
            match = limit_pattern.match(line)
            if match is None:
                raise ValueError(f"Unsupported limit line: {raw_line}")
            name, lower, upper, weight = match.groups()
            limits.append((name, float(lower), float(upper), 1.0 if weight is None else float(weight)))
        elif section == "PoseConstraints":
            tokens = line.split()
            if tokens[0] != "poseconstraints" or len(tokens) < 3:
                raise ValueError(f"Invalid pose-constraint line: {raw_line}")
            pose_constraints[tokens[1]] = {
                name: float(value) for name, value in (assignment.split("=", maxsplit=1) for assignment in tokens[2:])
            }
    return parameter_sets, limits, pose_constraints


def _encode_parameter_sets(
    parameter_index: dict[str, int],
    parameter_sets: list[tuple[str, list[str]]],
) -> tuple[np.ndarray, np.ndarray]:
    set_names = np.asarray([name for name, _ in parameter_sets], dtype=np.str_)
    set_membership = np.zeros((len(parameter_sets), len(parameter_index)), dtype=np.bool_)
    for row, (_, names) in enumerate(parameter_sets):
        for name in names:
            if name not in parameter_index:
                raise ValueError(f"Unknown parameter {name!r} in parameter set")
            set_membership[row, parameter_index[name]] = True
    expected_set_sizes = {
        "translation": 3,
        "rigid": 6,
        "scale": 68,
        "handPose": 60,
        "handScale": 52,
        "fingers": 54,
        "flexibleDofs": 20,
        "flexiblePose": 6,
    }
    actual_set_sizes = {name: len(names) for name, names in parameter_sets}
    if actual_set_sizes != expected_set_sizes:
        raise ValueError(f"Unexpected MHR parameter sets: {actual_set_sizes}")
    return set_names, set_membership


def _encode_limits(
    parameter_index: dict[str, int],
    limits: list[tuple[str, float, float, float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    parameter_count = len(parameter_index)
    lower_limits = np.full(parameter_count, -np.inf, dtype=np.float32)
    upper_limits = np.full(parameter_count, np.inf, dtype=np.float32)
    limit_indices = np.empty(len(limits), dtype=np.int64)
    limit_weights = np.empty(len(limits), dtype=np.float32)
    for row, (name, lower, upper, weight) in enumerate(limits):
        if name not in parameter_index:
            raise ValueError(f"Unknown limited parameter {name!r}")
        index = parameter_index[name]
        limit_indices[row] = index
        lower_limits[index] = lower
        upper_limits[index] = upper
        limit_weights[row] = weight
    if len(set(limit_indices.tolist())) != len(limit_indices):
        raise ValueError("MHR model definition contains duplicate parameter limits")
    if len(limit_indices) != 198:
        raise ValueError(f"Expected 198 MHR parameter limits, got {len(limit_indices)}")
    weight_counts = tuple(int(np.count_nonzero(np.isclose(limit_weights, value))) for value in (0.1, 1, 10))
    if weight_counts != (27, 170, 1):
        raise ValueError(f"Unexpected MHR limit-weight counts: {weight_counts}")
    return lower_limits, upper_limits, limit_indices, limit_weights


def _encode_tpose(
    parameter_index: dict[str, int],
    pose_constraints: dict[str, dict[str, float]],
) -> tuple[np.ndarray, np.ndarray]:
    if "tpose" not in pose_constraints:
        raise ValueError("The MHR model definition has no tpose pose constraint")
    tpose_values = np.zeros(len(parameter_index), dtype=np.float32)
    tpose_mask = np.zeros(len(parameter_index), dtype=np.bool_)
    for name, value in pose_constraints["tpose"].items():
        if name not in parameter_index:
            raise ValueError(f"Unknown tpose parameter {name!r}")
        index = parameter_index[name]
        tpose_values[index] = value
        tpose_mask[index] = True
    if int(tpose_mask.sum()) != 19:
        raise ValueError(f"Expected 19 tpose parameters, got {int(tpose_mask.sum())}")
    for elbow_name in ("l_elbow_bend", "r_elbow_bend"):
        if tpose_values[parameter_index[elbow_name]] != -0.5:
            raise ValueError(f"Unexpected tpose value for {elbow_name}")
    return tpose_values, tpose_mask


def _encode_model_sections(
    parameter_names: Sequence[str],
    parameter_sets: list[tuple[str, list[str]]],
    limits: list[tuple[str, float, float, float]],
    pose_constraints: dict[str, dict[str, float]],
) -> dict[str, np.ndarray]:
    parameter_index = {name: index for index, name in enumerate(parameter_names)}
    if len(parameter_index) != len(parameter_names):
        raise ValueError("MHR model parameter names are not unique")
    set_names, set_membership = _encode_parameter_sets(parameter_index, parameter_sets)
    lower_limits, upper_limits, limit_indices, limit_weights = _encode_limits(parameter_index, limits)
    tpose_values, tpose_mask = _encode_tpose(parameter_index, pose_constraints)

    return {
        "parameter_set_names": set_names,
        "parameter_set_membership": set_membership,
        "model_parameter_lower_limits": lower_limits,
        "model_parameter_upper_limits": upper_limits,
        "limit_parameter_indices": limit_indices,
        "limit_weights": limit_weights,
        "tpose_values": tpose_values,
        "tpose_mask": tpose_mask,
    }


def _joint_group_key(name: str) -> str:
    lower = name.lower()
    if any(part in lower for part in ZERO_WEIGHT_PARTS):
        return "zero_weight"
    suffix = lower[2:]
    if any(part in suffix for part in HAND_PARTS):
        return f"{lower[:2]}hand"
    if any(part in suffix for part in FOOT_PARTS):
        return f"{lower[:2]}foot"
    return re.sub(r"(_null|_twist\d*|\d+|_twist\d_proc)$", "", name).rstrip("_")


def _segmentation_table(joint_names: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    group_names: list[str] = []
    group_indices: dict[str, int] = {}
    joint_groups = np.empty(len(joint_names), dtype=np.int64)
    for joint_index, joint_name in enumerate(joint_names):
        group_name = _joint_group_key(joint_name)
        if group_name not in group_indices:
            group_indices[group_name] = len(group_names)
            group_names.append(group_name)
        joint_groups[joint_index] = group_indices[group_name]
    return np.asarray(group_names, dtype=np.str_), joint_groups


def _skeleton_state_matrices(skeleton_state: np.ndarray) -> np.ndarray:
    translations = skeleton_state[:, :3]
    quaternions = skeleton_state[:, 3:7]
    scales = skeleton_state[:, 7]
    x, y, z, w = np.moveaxis(quaternions, -1, 0)
    rotations = np.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        axis=-1,
    ).reshape(-1, 3, 3)
    matrices = np.broadcast_to(np.eye(4, dtype=np.float32), (len(skeleton_state), 4, 4)).copy()
    matrices[:, :3, :3] = rotations * scales[:, None, None]
    matrices[:, :3, 3] = translations
    return matrices


def _matrices_to_meters(matrices: np.ndarray) -> np.ndarray:
    result = np.asarray(matrices, dtype=np.float32).copy()
    result[:, :3, 3] *= CENTIMETERS_TO_METERS
    return result


def _classify_parameter_columns(transform: np.ndarray) -> np.ndarray:
    channel_types = np.empty(MODEL_PARAMETER_COUNT, dtype=np.int8)
    for column in range(MODEL_PARAMETER_COUNT):
        rows = np.flatnonzero(transform[:, column])
        if not len(rows):
            raise ValueError(f"Model parameter column {column} drives no joint channel")
        channels = set((rows % JOINT_PARAMETER_COUNT).tolist())
        if channels <= TRANSLATION_CHANNELS:
            channel_types[column] = 0
        elif channels <= ROTATION_CHANNELS:
            channel_types[column] = 1
        elif channels == {SCALE_CHANNEL}:
            channel_types[column] = 2
        else:
            raise ValueError(f"Model parameter column {column} mixes channel types: {sorted(channels)}")
    return channel_types


def _validate_root_parameter_transform(parameter_names: Sequence[str], transform: np.ndarray) -> None:
    expected_root_names = ("root_tx", "root_ty", "root_tz", "root_rx", "root_ry", "root_rz")
    if tuple(parameter_names[:6]) != expected_root_names:
        raise ValueError(f"Unexpected root parameter order: {parameter_names[:6]}")
    expected_root_rows = np.arange(JOINT_PARAMETER_COUNT, JOINT_PARAMETER_COUNT + 6)
    expected_root_weights = np.asarray((10, 10, 10, 1, 1, 1), dtype=np.float32)
    expected_root_transform = np.zeros((transform.shape[0], 6), dtype=np.float32)
    expected_root_transform[expected_root_rows, np.arange(6)] = expected_root_weights
    if not np.array_equal(transform[:, :6], expected_root_transform):
        raise ValueError("Unexpected MHR root parameter mapping")


def _sparse_parameter_transform_data(transform: np.ndarray, model_path: Path) -> dict[str, np.ndarray]:
    driven_rows = np.flatnonzero(np.any(transform[:, :MODEL_PARAMETER_COUNT] != 0, axis=1))
    driven_channels = driven_rows % JOINT_PARAMETER_COUNT
    driven_joint_count = len(np.unique(driven_rows // JOINT_PARAMETER_COUNT))
    if driven_joint_count != 113:
        raise ValueError(f"Expected 113 driven joints, got {driven_joint_count}")
    row_counts = np.bincount(driven_channels, minlength=JOINT_PARAMETER_COUNT)
    driven_type_counts = np.asarray((row_counts[:3].sum(), row_counts[3:6].sum(), row_counts[6]), dtype=np.int64)
    if tuple(driven_type_counts.tolist()) != (136, 157, 2):
        raise ValueError(f"Expected unique driven-row counts (136, 157, 2), got {tuple(driven_type_counts)}")

    transform_rows, transform_columns = np.nonzero(transform)
    entry_counts = np.bincount(transform_rows % JOINT_PARAMETER_COUNT, minlength=JOINT_PARAMETER_COUNT)
    entry_type_counts = np.asarray((entry_counts[:3].sum(), entry_counts[3:6].sum(), entry_counts[6]), dtype=np.int64)
    if tuple(entry_type_counts.tolist()) != (191, 175, 2):
        raise ValueError(f"Expected transform-entry counts (191, 175, 2), got {tuple(entry_type_counts)}")
    statement_type_counts = _parameter_transform_statement_counts(model_path)
    if tuple(statement_type_counts.tolist()) != (191, 157, 2):
        raise ValueError(f"Expected .model assignment counts (191, 157, 2), got {tuple(statement_type_counts)}")

    row_source_counts = np.count_nonzero(transform[:, :MODEL_PARAMETER_COUNT], axis=1)
    multi_source_rows = np.flatnonzero(row_source_counts > 1)
    multi_source_channels = multi_source_rows % JOINT_PARAMETER_COUNT
    multi_source_type_counts = np.asarray(
        (
            np.isin(multi_source_channels, tuple(TRANSLATION_CHANNELS)).sum(),
            np.isin(multi_source_channels, tuple(ROTATION_CHANNELS)).sum(),
            np.count_nonzero(multi_source_channels == SCALE_CHANNEL),
        ),
        dtype=np.int64,
    )
    if tuple(multi_source_type_counts.tolist()) != (55, 12, 0):
        raise ValueError(f"Expected multi-source row counts (55, 12, 0), got {tuple(multi_source_type_counts)}")

    transform_values = transform[transform_rows, transform_columns].copy()
    translation_entries = np.isin(transform_rows % JOINT_PARAMETER_COUNT, tuple(TRANSLATION_CHANNELS))
    transform_values[translation_entries] *= CENTIMETERS_TO_METERS
    channel_types = _classify_parameter_columns(transform[:, :MODEL_PARAMETER_COUNT])
    if tuple(np.bincount(channel_types, minlength=3).tolist()) != (74, 128, 2):
        raise ValueError("Expected 74/128/2 translation/rotation/scale model parameters")
    return {
        "model_parameter_channel_types": channel_types,
        "parameter_transform_shape": np.asarray(transform.shape, dtype=np.int64),
        "parameter_transform_rows": transform_rows.astype(np.int64),
        "parameter_transform_columns": transform_columns.astype(np.int64),
        "parameter_transform_values": transform_values.astype(np.float32),
        "parameter_transform_statement_type_counts": statement_type_counts,
        "parameter_transform_entry_type_counts": entry_type_counts,
        "driven_row_type_counts": driven_type_counts,
        "driven_joint_count": np.asarray(driven_joint_count, dtype=np.int64),
        "multi_source_row_type_counts": multi_source_type_counts,
        "driven_row_indices": driven_rows.astype(np.int64),
        "driven_joint_indices": (driven_rows // JOINT_PARAMETER_COUNT).astype(np.int64),
        "driven_channel_indices": driven_channels.astype(np.int8),
        "driven_channel_names": np.asarray([CHANNEL_NAMES[index] for index in driven_channels], dtype=np.str_),
    }


def _static_signature(data: dict[str, np.ndarray]) -> tuple[np.ndarray, ...]:
    return tuple(
        data[name]
        for name in (
            "joint_names",
            "joint_parents",
            "joint_pre_rotations",
            "joint_translation_offsets",
            "bind_matrices",
            "inverse_bind_matrices",
            "model_parameter_names",
            "parameter_transform_rows",
            "parameter_transform_columns",
            "parameter_transform_values",
        )
    )


def _load_character(fbx_path: Path, model_path: Path) -> tuple[Any, Any]:
    raw_character = pym_geometry.Character.load_fbx(str(fbx_path), str(model_path), load_blendshapes=True)
    if raw_character.parameter_transform.size != MODEL_PARAMETER_COUNT:
        raise ValueError(f"Expected {MODEL_PARAMETER_COUNT} model parameters")
    if raw_character.blend_shape is None or raw_character.blend_shape.n_shapes != BLENDSHAPE_COUNT:
        raise ValueError(f"Expected {BLENDSHAPE_COUNT} blend shapes")
    return raw_character, raw_character.with_blend_shape(raw_character.blend_shape)


def _skeleton_data(character: Any) -> dict[str, np.ndarray]:
    joint_names = list(character.skeleton.joint_names)
    parents = _as_numpy(character.skeleton.joint_parents).astype(np.int64, copy=False)
    pre_rotations = _as_numpy(character.skeleton.pre_rotations).astype(np.float32, copy=False)
    offsets = _as_numpy(character.skeleton.offsets).astype(np.float32, copy=False)
    if len(joint_names) != JOINT_COUNT or parents.shape != (JOINT_COUNT,):
        raise ValueError(f"Expected a {JOINT_COUNT}-joint skeleton")
    if joint_names[:2] != ["body_world", "root"]:
        raise ValueError(f"Expected native joints 0/1 to be body_world/root, got {joint_names[:2]}")
    if parents[0] != -1 or np.any(parents[1:] >= np.arange(1, JOINT_COUNT)):
        raise ValueError("MHR joint order is not topological")
    quaternion_norms = np.linalg.norm(pre_rotations, axis=-1)
    if np.any(quaternion_norms == 0):
        raise ValueError("MHR contains a zero-norm pre-rotation quaternion")
    max_norm_error = float(np.max(np.abs(quaternion_norms - 1)))
    group_names, joint_group_indices = _segmentation_table(joint_names)
    return {
        "world_joint_index": np.asarray(0, dtype=np.int64),
        "root_joint_index": np.asarray(1, dtype=np.int64),
        "joint_names": np.asarray(joint_names, dtype=np.str_),
        "joint_parents": parents,
        "joint_pre_rotations": pre_rotations / quaternion_norms[:, None],
        "joint_translation_offsets": offsets * CENTIMETERS_TO_METERS,
        "pre_rotation_max_norm_error": np.asarray(max_norm_error, dtype=np.float32),
        "segmentation_group_names": group_names,
        "segmentation_joint_group_indices": joint_group_indices,
    }


def _parameter_transform_data(raw_character: Any, character: Any, model_path: Path) -> dict[str, np.ndarray]:
    parameter_names = list(raw_character.parameter_transform.names)
    transform = _as_numpy(character.parameter_transform.transform).astype(np.float32, copy=False)
    expected_shape = (JOINT_COUNT * JOINT_PARAMETER_COUNT, MODEL_PARAMETER_COUNT + BLENDSHAPE_COUNT)
    if transform.shape != expected_shape:
        raise ValueError(f"Expected parameter transform shape {expected_shape}, got {transform.shape}")
    if np.any(transform[:, MODEL_PARAMETER_COUNT:]):
        raise ValueError("Blendshape parameter columns unexpectedly move MHR joints")
    zero_parameters = torch.zeros(MODEL_PARAMETER_COUNT, dtype=torch.float32)
    if np.any(_as_numpy(raw_character.parameter_transform.apply(zero_parameters))):
        raise ValueError("MHR parameter-transform offsets are not zero")
    _validate_root_parameter_transform(parameter_names, transform)
    sparse_transform = _sparse_parameter_transform_data(transform, model_path)
    model_sections = _encode_model_sections(parameter_names, *_read_model_entries(model_path))
    limited = model_sections["limit_parameter_indices"]
    reference_lower, reference_upper = raw_character.model_parameter_limits
    if not np.allclose(model_sections["model_parameter_lower_limits"][limited], _as_numpy(reference_lower)[limited]):
        raise ValueError("Parsed lower limits disagree with pymomentum")
    if not np.allclose(model_sections["model_parameter_upper_limits"][limited], _as_numpy(reference_upper)[limited]):
        raise ValueError("Parsed upper limits disagree with pymomentum")
    return {
        "model_parameter_names": np.asarray(parameter_names, dtype=np.str_),
        **sparse_transform,
        **model_sections,
    }


def _mesh_data(character: Any, lod: int) -> dict[str, np.ndarray]:
    mesh_vertices = _as_numpy(character.mesh.vertices).astype(np.float32, copy=False)
    template_vertices = _as_numpy(character.blend_shape.base_shape).astype(np.float32, copy=False)
    if len(template_vertices) != LOD_VERTEX_COUNTS[lod]:
        raise ValueError(f"Expected {LOD_VERTEX_COUNTS[lod]} LOD{lod} vertices, got {len(template_vertices)}")
    template_mesh_max_error_cm = float(np.max(np.abs(template_vertices - mesh_vertices)))
    if template_mesh_max_error_cm > 1e-5:
        raise ValueError(f"Blendshape base and FBX mesh differ by {template_mesh_max_error_cm} cm")
    faces = _as_numpy(character.mesh.faces).astype(np.int64, copy=False)
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.min() < 0 or faces.max() >= len(template_vertices):
        raise ValueError("Invalid triangular MHR mesh topology")

    skin_indices = _as_numpy(character.skin_weights.index).astype(np.int64, copy=False)
    skin_weights = _as_numpy(character.skin_weights.weight).astype(np.float32, copy=False)
    if skin_indices.shape != skin_weights.shape or skin_indices.shape != (len(template_vertices), 8):
        raise ValueError("Expected padded [V, 8] MHR skinning data")
    weight_sums = skin_weights.sum(axis=-1)
    if np.any(weight_sums <= 0):
        raise ValueError("MHR contains a vertex with zero total skin weight")
    skin_weight_max_sum_error = float(np.max(np.abs(weight_sums - 1)))
    skin_weights_renormalized = skin_weight_max_sum_error > 1e-6
    if skin_weights_renormalized:
        skin_weights = skin_weights / weight_sums[:, None]

    blendshape_basis = _as_numpy(character.blend_shape.shape_vectors).astype(np.float32, copy=False)
    if blendshape_basis.shape != (BLENDSHAPE_COUNT, len(template_vertices), 3):
        raise ValueError("Unexpected MHR blendshape basis shape")
    blendshape_basis = np.transpose(blendshape_basis, (1, 2, 0)) * CENTIMETERS_TO_METERS
    return {
        "template_vertices": template_vertices * CENTIMETERS_TO_METERS,
        "faces": faces,
        "skin_indices": skin_indices,
        "skin_weights": skin_weights,
        "skin_weight_max_sum_error": np.asarray(skin_weight_max_sum_error, dtype=np.float32),
        "skin_weights_renormalized": np.asarray(skin_weights_renormalized, dtype=np.bool_),
        "identity_basis": blendshape_basis[:, :, :IDENTITY_COUNT],
        "expression_basis": blendshape_basis[:, :, IDENTITY_COUNT:],
        "template_mesh_max_error_cm": np.asarray(template_mesh_max_error_cm, dtype=np.float32),
    }


def _corrective_data(
    corrective_path: Path,
    vertex_count: int,
    activation: dict[str, np.ndarray],
    joint_names: np.ndarray,
) -> dict[str, np.ndarray]:
    with np.load(corrective_path, allow_pickle=False) as corrective_file:
        if corrective_file.files != ["corrective_blendshapes"]:
            raise ValueError(f"Unexpected corrective fields: {corrective_file.files}")
        output = np.asarray(corrective_file["corrective_blendshapes"], dtype=np.float32)
    if output.shape != (CORRECTIVE_HIDDEN_COUNT, vertex_count, 3):
        raise ValueError("Unexpected MHR corrective output shape")
    first_layer_indices = np.asarray(activation["0.sparse_indices"], dtype=np.int64)
    first_layer_weights = np.asarray(activation["0.sparse_weight"], dtype=np.float32)
    first_layer_mask = np.asarray(activation["posedirs_sparse_mask"], dtype=np.bool_)
    if first_layer_mask.shape != (CORRECTIVE_HIDDEN_COUNT, CORRECTIVE_FEATURE_COUNT):
        raise ValueError("Unexpected MHR corrective activation mask shape")
    if first_layer_indices.shape != (2, len(first_layer_weights)):
        raise ValueError("Unexpected MHR corrective sparse activation shape")
    if not np.array_equal(first_layer_indices, np.asarray(first_layer_mask.nonzero())):
        raise ValueError("Corrective sparse indices disagree with the activation mask")
    corrective_joint_indices = np.arange(2, JOINT_COUNT, dtype=np.int64)
    return {
        "corrective_first_layer_indices": first_layer_indices,
        "corrective_first_layer_weights": first_layer_weights,
        "corrective_first_layer_mask": first_layer_mask,
        "corrective_output_weights": output.reshape(CORRECTIVE_HIDDEN_COUNT, -1) * CENTIMETERS_TO_METERS,
        "corrective_joint_indices": corrective_joint_indices,
        "corrective_joint_names": joint_names[corrective_joint_indices],
        "corrective_feature_matrix_indices": np.asarray((0, 3, 6, 1, 4, 7), dtype=np.int64),
        "corrective_identity_feature": np.asarray((1, 0, 0, 0, 1, 0), dtype=np.float32),
    }


def _bind_and_convention_data(character: Any, fbx_path: Path) -> dict[str, np.ndarray]:
    bind_matrices_cm = _as_numpy(character.bind_pose).astype(np.float32, copy=False)
    inverse_bind_matrices_cm = _as_numpy(character.inverse_bind_pose).astype(np.float32, copy=False)
    zero_joint_parameters = torch.zeros(JOINT_COUNT * JOINT_PARAMETER_COUNT, dtype=torch.float32)
    zero_skeleton_state = _as_numpy(
        pym_geometry.joint_parameters_to_skeleton_state(character, zero_joint_parameters)
    ).astype(np.float32, copy=False)
    zero_fk_matrices_cm = _skeleton_state_matrices(zero_skeleton_state)
    zero_fk_bind_max_error_cm = float(np.max(np.abs(zero_fk_matrices_cm - bind_matrices_cm)))
    bind_inverse_error = float(
        np.max(np.abs(bind_matrices_cm @ inverse_bind_matrices_cm - np.eye(4, dtype=np.float32)))
    )
    if bind_inverse_error > 1e-3:
        raise ValueError(f"FBX bind matrices are not inverses; max error {bind_inverse_error}")

    properties = _read_fbx_global_properties(fbx_path)
    unit_scale_factor_cm = float(properties["UnitScaleFactor"])
    if unit_scale_factor_cm != 1.0:
        raise ValueError(f"Expected centimeter FBX units, got UnitScaleFactor={unit_scale_factor_cm}")
    return {
        "native_unit": np.asarray("centimeter", dtype=np.str_),
        "storage_unit": np.asarray("meter", dtype=np.str_),
        "length_scale_to_meters": np.asarray(CENTIMETERS_TO_METERS, dtype=np.float32),
        "native_pose_format": np.asarray("euler_xyz", dtype=np.str_),
        "quaternion_order": np.asarray("xyzw", dtype=np.str_),
        "euler_composition_order": np.asarray("Rz_Ry_Rx", dtype=np.str_),
        "scale_encoding": np.asarray("exp2", dtype=np.str_),
        "axis_metadata_source": np.asarray("fbx_global_settings", dtype=np.str_),
        "up_axis": np.asarray(_signed_axis(int(properties["UpAxis"]), int(properties["UpAxisSign"])), dtype=np.str_),
        "forward_axis": np.asarray(
            _signed_axis(int(properties["FrontAxis"]), int(properties["FrontAxisSign"])), dtype=np.str_
        ),
        "coordinate_axis": np.asarray(
            _signed_axis(int(properties["CoordAxis"]), int(properties["CoordAxisSign"])), dtype=np.str_
        ),
        "bind_matrices": _matrices_to_meters(bind_matrices_cm),
        "inverse_bind_matrices": _matrices_to_meters(inverse_bind_matrices_cm),
        "zero_fk_bind_max_error_cm": np.asarray(zero_fk_bind_max_error_cm, dtype=np.float32),
        "bind_inverse_max_error": np.asarray(bind_inverse_error, dtype=np.float32),
    }


def _validate_segmentation_skinning(data: dict[str, np.ndarray]) -> None:
    group_names = data["segmentation_group_names"].tolist()
    if len(group_names) != 20 or group_names.count("zero_weight") != 1:
        raise ValueError(f"Expected 20 segmentation groups including zero_weight, got {group_names}")
    zero_weight_group = group_names.index("zero_weight")
    joint_groups = data["segmentation_joint_group_indices"]
    active_joint_indices = data["skin_indices"][data["skin_weights"] > 0]
    if np.any(joint_groups[active_joint_indices] == zero_weight_group):
        raise ValueError("A positive skin influence maps to the zero_weight segmentation group")


def _convert_lod(
    assets_dir: Path,
    output_dir: Path,
    lod: int,
    activation: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], int]:
    fbx_path = assets_dir / f"lod{lod}.fbx"
    model_path = assets_dir / "compact_v6_1.model"
    corrective_path = assets_dir / f"corrective_blendshapes_lod{lod}.npz"
    for path in (fbx_path, model_path, corrective_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required MHR asset not found: {path}")
    raw_character, character = _load_character(fbx_path, model_path)
    data = {
        "format_version": np.asarray(1, dtype=np.int64),
        "lod": np.asarray(lod, dtype=np.int64),
        **_skeleton_data(character),
        **_parameter_transform_data(raw_character, character, model_path),
        **_mesh_data(character, lod),
        **_bind_and_convention_data(character, fbx_path),
    }
    data.update(_corrective_data(corrective_path, len(data["template_vertices"]), activation, data["joint_names"]))
    _validate_segmentation_skinning(data)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"mhr_lod{lod}.npz"
    temporary_path = output_path.with_suffix(".tmp.npz")
    np.savez(temporary_path, **data)
    temporary_path.replace(output_path)
    return data, output_path.stat().st_size


def main() -> None:
    """Convert the selected MHR LODs and validate all cross-LOD invariants."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args()
    lods = sorted(set(args.lod if args.lod is not None else range(LOD_COUNT)))
    activation_path = args.assets_dir / "corrective_activation.npz"
    if not activation_path.is_file():
        raise FileNotFoundError(f"Required MHR asset not found: {activation_path}")
    with np.load(activation_path, allow_pickle=False) as activation_file:
        activation = {name: np.asarray(activation_file[name]) for name in activation_file.files}

    reference_signature: tuple[np.ndarray, ...] | None = None
    for lod in lods:
        LOGGER.info("Converting MHR LOD%d", lod)
        data, archive_size = _convert_lod(args.assets_dir, args.output_dir, lod, activation)
        signature = _static_signature(data)
        if reference_signature is None:
            reference_signature = signature
        elif not all(
            np.array_equal(current, reference)
            for current, reference in zip(signature, reference_signature, strict=True)
        ):
            raise ValueError(f"LOD{lod} skeleton or parameter transform differs from the first converted LOD")
        LOGGER.info(
            "LOD%d: %d vertices, %d faces, %.2f MiB, bind-vs-zero-FK max %.9g cm",
            lod,
            len(data["template_vertices"]),
            len(data["faces"]),
            archive_size / (1024 * 1024),
            float(data["zero_fk_bind_max_error_cm"]),
        )


if __name__ == "__main__":
    main()
