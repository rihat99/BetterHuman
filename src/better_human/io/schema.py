"""Declarative schemas for SMPL-family weight files."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)

#: Sentinel axis for the single derived width ``(num_joints - 1) * 9``.
POSE_FEATURES = "(J-1)*9"
#: Sentinel axis for flattened MHR vertex coordinates ``num_vertices * 3``.
VERTEX_COORDINATES = "V*3"

Axis = int | str
DTypeExpectation = str | tuple[str, ...]


@dataclass(frozen=True)
class FieldSpec:
    """Expected shape and dtype for one logical model field."""

    #: Per-axis size: an ``int`` literal, a named axis bound on first sighting,
    #: or :data:`POSE_FEATURES` for the single derived width.
    shape: tuple[Axis, ...]
    #: Accepted dtype name, ``"floating"``/``"integer"`` family, or a tuple of
    #: accepted alternatives.
    dtype: DTypeExpectation


@dataclass(frozen=True)
class ModelSchema:
    """Required fields and any pre-bound axes for one model family."""

    name: str
    fields: Mapping[str, FieldSpec]
    dimensions: Mapping[str, int] = field(default_factory=dict)


def _base_fields(*, shapedirs_dtype: str, other_float_dtype: str) -> dict[str, FieldSpec]:
    return {
        "v_template": FieldSpec(("V", 3), other_float_dtype),
        "shapedirs": FieldSpec(("V", 3, "S"), shapedirs_dtype),
        "posedirs": FieldSpec(("V", 3, POSE_FEATURES), other_float_dtype),
        "J_regressor": FieldSpec(("J", "V"), other_float_dtype),
        "weights": FieldSpec(("V", "J"), other_float_dtype),
        "kintree_table": FieldSpec((2, "J"), "integer"),
        "f": FieldSpec(("F", 3), "integer"),
    }


_smpl_fields = _base_fields(shapedirs_dtype="float64", other_float_dtype="float64")
_smpl_fields.update(
    {
        "kintree_table": FieldSpec((2, "J"), ("int32", "uint32")),
        "f": FieldSpec(("F", 3), "uint32"),
    }
)
SMPL_SCHEMA = ModelSchema(name="smpl", fields=_smpl_fields)

_smplx_fields = _base_fields(shapedirs_dtype="float64", other_float_dtype="float32")
_smplx_fields.update(
    {
        "kintree_table": FieldSpec((2, "J"), "int64"),
        "f": FieldSpec(("F", 3), "uint32"),
        "hands_componentsl": FieldSpec((45, 45), "float32"),
        "hands_componentsr": FieldSpec((45, 45), "float32"),
        "hands_meanl": FieldSpec((45,), "float32"),
        "hands_meanr": FieldSpec((45,), "float32"),
    }
)
SMPLX_SCHEMA = ModelSchema(name="smplx", fields=_smplx_fields, dimensions={"S": 400})

MHR_CORRECTIVE_FIELDS = frozenset(
    {
        "corrective_first_layer_indices",
        "corrective_first_layer_weights",
        "corrective_first_layer_mask",
        "corrective_output_weights",
        "corrective_joint_indices",
        "corrective_joint_names",
        "corrective_feature_matrix_indices",
        "corrective_identity_feature",
    }
)

MHR_SCHEMA = ModelSchema(
    name="mhr",
    fields={
        "format_version": FieldSpec((), "int64"),
        "lod": FieldSpec((), "int64"),
        "world_joint_index": FieldSpec((), "int64"),
        "root_joint_index": FieldSpec((), "int64"),
        "joint_names": FieldSpec((127,), "<U20"),
        "joint_parents": FieldSpec((127,), "int64"),
        "joint_pre_rotations": FieldSpec((127, 4), "float32"),
        "joint_translation_offsets": FieldSpec((127, 3), "float32"),
        "pre_rotation_max_norm_error": FieldSpec((), "float32"),
        "segmentation_group_names": FieldSpec((20,), "<U11"),
        "segmentation_joint_group_indices": FieldSpec((127,), "int64"),
        "model_parameter_names": FieldSpec((204,), "<U24"),
        "model_parameter_channel_types": FieldSpec((204,), "int8"),
        "parameter_transform_shape": FieldSpec((2,), "int64"),
        "parameter_transform_rows": FieldSpec((368,), "int64"),
        "parameter_transform_columns": FieldSpec((368,), "int64"),
        "parameter_transform_values": FieldSpec((368,), "float32"),
        "parameter_transform_statement_type_counts": FieldSpec((3,), "int64"),
        "parameter_transform_entry_type_counts": FieldSpec((3,), "int64"),
        "driven_row_type_counts": FieldSpec((3,), "int64"),
        "multi_source_row_type_counts": FieldSpec((3,), "int64"),
        "driven_joint_count": FieldSpec((), "int64"),
        "driven_row_indices": FieldSpec((295,), "int64"),
        "driven_joint_indices": FieldSpec((295,), "int64"),
        "driven_channel_indices": FieldSpec((295,), "int8"),
        "driven_channel_names": FieldSpec((295,), "<U2"),
        "parameter_set_names": FieldSpec((8,), "<U12"),
        "parameter_set_membership": FieldSpec((8, 204), "bool"),
        "model_parameter_lower_limits": FieldSpec((204,), "float32"),
        "model_parameter_upper_limits": FieldSpec((204,), "float32"),
        "limit_parameter_indices": FieldSpec((198,), "int64"),
        "limit_weights": FieldSpec((198,), "float32"),
        "tpose_values": FieldSpec((204,), "float32"),
        "tpose_mask": FieldSpec((204,), "bool"),
        "template_vertices": FieldSpec(("V", 3), "float32"),
        "faces": FieldSpec(("F", 3), "int64"),
        "skin_indices": FieldSpec(("V", 8), "int64"),
        "skin_weights": FieldSpec(("V", 8), "float32"),
        "skin_weight_max_sum_error": FieldSpec((), "float32"),
        "skin_weights_renormalized": FieldSpec((), "bool"),
        "identity_basis": FieldSpec(("V", 3, 45), "float32"),
        "expression_basis": FieldSpec(("V", 3, 72), "float32"),
        "template_mesh_max_error_cm": FieldSpec((), "float32"),
        "native_unit": FieldSpec((), "<U10"),
        "storage_unit": FieldSpec((), "<U5"),
        "length_scale_to_meters": FieldSpec((), "float32"),
        "native_pose_format": FieldSpec((), "<U9"),
        "quaternion_order": FieldSpec((), "<U4"),
        "euler_composition_order": FieldSpec((), "<U8"),
        "scale_encoding": FieldSpec((), "<U4"),
        "axis_metadata_source": FieldSpec((), "<U19"),
        "up_axis": FieldSpec((), "<U2"),
        "forward_axis": FieldSpec((), "<U2"),
        "coordinate_axis": FieldSpec((), "<U2"),
        "bind_matrices": FieldSpec((127, 4, 4), "float32"),
        "inverse_bind_matrices": FieldSpec((127, 4, 4), "float32"),
        "zero_fk_bind_max_error_cm": FieldSpec((), "float32"),
        "bind_inverse_max_error": FieldSpec((), "float32"),
        "corrective_first_layer_indices": FieldSpec((2, 53136), "int64"),
        "corrective_first_layer_weights": FieldSpec((53136,), "float32"),
        "corrective_first_layer_mask": FieldSpec((3000, 750), "bool"),
        "corrective_output_weights": FieldSpec((3000, VERTEX_COORDINATES), "float32"),
        "corrective_joint_indices": FieldSpec((125,), "int64"),
        "corrective_joint_names": FieldSpec((125,), "<U20"),
        "corrective_feature_matrix_indices": FieldSpec((6,), "int64"),
        "corrective_identity_feature": FieldSpec((6,), "float32"),
    },
)


def validate(
    data: Mapping[str, Any],
    schema: ModelSchema,
    *,
    source: str = "<mapping>",
) -> None:
    """Validate required fields while allowing unknown vendor metadata."""
    bindings = dict(schema.dimensions)

    for name, spec in schema.fields.items():
        if name not in data:
            raise _validation_error(
                source, name, f"a required field with shape {_format_shape(spec.shape)}", "missing", schema
            )
        array = _as_array(data, name, source, schema)
        if array.ndim != len(spec.shape):
            raise _validation_error(
                source,
                name,
                f"rank {len(spec.shape)} with shape {_format_shape(spec.shape)}",
                f"shape {array.shape}",
                schema,
            )
        _bind_axes(array.shape, spec.shape, bindings, source, name, schema)

    for name, spec in schema.fields.items():
        _validate_array(_as_array(data, name, source, schema), spec, bindings, source, name, schema)

    extras = sorted(set(data) - set(schema.fields))
    if extras:
        LOGGER.debug("Ignoring unknown fields in %s: %s", source, ", ".join(extras))


def _bind_axes(
    actual_shape: tuple[int, ...],
    expected_shape: tuple[Axis, ...],
    bindings: dict[str, int],
    source: str,
    field_name: str,
    schema: ModelSchema,
) -> None:
    for actual, expected in zip(actual_shape, expected_shape, strict=True):
        if not isinstance(expected, str) or expected in (POSE_FEATURES, VERTEX_COORDINATES):
            continue
        previous = bindings.setdefault(expected, actual)
        if previous != actual:
            raise _validation_error(source, field_name, f"dimension {expected} = {previous}", str(actual), schema)


def _validate_array(
    array: np.ndarray,
    spec: FieldSpec,
    bindings: Mapping[str, int],
    source: str,
    field_name: str,
    schema: ModelSchema,
) -> None:
    expected_shape = tuple(_resolve_axis(axis, bindings) for axis in spec.shape)
    if array.shape != expected_shape:
        raise _validation_error(
            source, field_name, f"shape {_format_shape(spec.shape)}", f"shape {array.shape}", schema
        )
    if not _dtype_matches(array.dtype, spec.dtype):
        raise _validation_error(
            source, field_name, f"dtype {_format_dtype(spec.dtype)}", f"dtype {array.dtype}", schema
        )


def _resolve_axis(axis: Axis, bindings: Mapping[str, int]) -> int:
    if isinstance(axis, int):
        return axis
    if axis == POSE_FEATURES:
        return (bindings["J"] - 1) * 9
    if axis == VERTEX_COORDINATES:
        return bindings["V"] * 3
    return bindings[axis]


def _format_shape(shape: tuple[Axis, ...]) -> str:
    parts = [str(axis) for axis in shape]
    suffix = "," if len(parts) == 1 else ""
    return f"({', '.join(parts)}{suffix})"


def _dtype_matches(actual: np.dtype[Any], expected: DTypeExpectation) -> bool:
    if isinstance(expected, tuple):
        return any(_dtype_matches(actual, candidate) for candidate in expected)
    if expected == "floating":
        return np.issubdtype(actual, np.floating)
    if expected == "integer":
        return np.issubdtype(actual, np.integer)
    return actual == np.dtype(expected)


def _format_dtype(expected: DTypeExpectation) -> str:
    return " or ".join(expected) if isinstance(expected, tuple) else expected


def _as_array(
    data: Mapping[str, Any],
    field_name: str,
    source: str,
    schema: ModelSchema,
) -> np.ndarray:
    try:
        return np.asarray(data[field_name])
    except Exception as error:
        raise _validation_error(
            source,
            field_name,
            "a readable numeric array",
            f"an unreadable value ({error})",
            schema,
        ) from error


def _validation_error(
    source: str,
    field_name: str,
    expectation: str,
    actual: str,
    schema: ModelSchema,
) -> ValueError:
    return ValueError(
        f"{source}: field {field_name!r} expected {expectation}, got {actual}. "
        f"Reconvert the asset or provide a valid {schema.name} .npz file."
    )
