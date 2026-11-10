"""Declarative schemas for the archives BetterHuman reads (weight files, the shipped mass data): fields, shapes, dtypes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, final

import numpy as np

from .formats import DENSITY_TABLES, MHR, SEGMENTATION, SOMA, FormatError

#: Sentinel axis for the single derived width ``(num_joints - 1) * 9``.
POSE_FEATURES = "(J-1)*9"
#: Sentinel axis for flattened MHR vertex coordinates ``num_vertices * 3``.
VERTEX_COORDINATES = "V*3"

Axis = int | str
DTypeExpectation = str | tuple[str, ...]


@final
@dataclass(frozen=True)
class FieldSpec:
    """Expected shape and dtype for one logical model field."""

    #: Per-axis size: an ``int`` literal, a named axis bound on first sighting,
    #: or :data:`POSE_FEATURES` for the single derived width.
    shape: tuple[Axis, ...]
    #: Accepted dtype name, ``"floating"``/``"integer"``/``"str"`` family, or a
    #: tuple of accepted alternatives.
    dtype: DTypeExpectation


@final
@dataclass(frozen=True)
class ModelSchema:
    """Required fields and any pre-bound axes for one model family."""

    name: str
    fields: Mapping[str, FieldSpec]
    dimensions: Mapping[str, int] = field(default_factory=dict)
    #: The ``format`` the file names (:mod:`better_human.io.formats`); ``None`` for vendor files that name none.
    format: str | None = None


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

_smplh_fields = _base_fields(shapedirs_dtype="floating", other_float_dtype="floating")
_smplh_fields.update({"kintree_table": FieldSpec((2, "J"), "integer"), "f": FieldSpec(("F", 3), "integer")})
SMPLH_SCHEMA = ModelSchema(name="smplh", fields=_smplh_fields)

_mano_fields = dict(_smplh_fields)
_mano_fields.update({"hands_components": FieldSpec((45, 45), "floating"), "hands_mean": FieldSpec((45,), "floating")})
MANO_SCHEMA = ModelSchema(name="mano", fields=_mano_fields)

FLAME_SCHEMA = ModelSchema(name="flame", fields=_smplh_fields, dimensions={"S": 400})

MHR_CORRECTIVE_FIELDS = frozenset(
    {
        "corrective_first_layer_indices",
        "corrective_first_layer_weights",
        "corrective_first_layer_mask",
        "corrective_output_weights",
    }
)

MHR_SCHEMA = ModelSchema(
    name="mhr",
    format=MHR,
    fields={
        "format": FieldSpec((), "str"),
        "format_version": FieldSpec((), "int64"),
        "provenance": FieldSpec((), "str"),
        "lod": FieldSpec((), "int64"),
        "joint_names": FieldSpec((127,), "<U20"),
        "joint_parents": FieldSpec((127,), "int64"),
        "joint_pre_rotations": FieldSpec((127, 4), "float32"),
        "joint_translation_offsets": FieldSpec((127, 3), "float32"),
        "segmentation_group_names": FieldSpec((20,), "<U11"),
        "segmentation_joint_group_indices": FieldSpec((127,), "int64"),
        "model_parameter_names": FieldSpec((204,), "<U24"),
        "model_parameter_channel_types": FieldSpec((204,), "int8"),
        "parameter_transform_rows": FieldSpec((368,), "int64"),
        "parameter_transform_columns": FieldSpec((368,), "int64"),
        "parameter_transform_values": FieldSpec((368,), "float32"),
        "model_parameter_lower_limits": FieldSpec((204,), "float32"),
        "model_parameter_upper_limits": FieldSpec((204,), "float32"),
        "template_vertices": FieldSpec(("V", 3), "float32"),
        "faces": FieldSpec(("F", 3), "int64"),
        "skin_indices": FieldSpec(("V", 8), "int64"),
        "skin_weights": FieldSpec(("V", 8), "float32"),
        "identity_basis": FieldSpec(("V", 3, 45), "float32"),
        "expression_basis": FieldSpec(("V", 3, 72), "float32"),
        "up_axis": FieldSpec((), "<U2"),
        "bind_matrices": FieldSpec((127, 4, 4), "float32"),
        "inverse_bind_matrices": FieldSpec((127, 4, 4), "float32"),
        "corrective_first_layer_indices": FieldSpec((2, 53136), "int64"),
        "corrective_first_layer_weights": FieldSpec((53136,), "float32"),
        "corrective_first_layer_mask": FieldSpec((3000, 750), "bool"),
        "corrective_output_weights": FieldSpec((3000, VERTEX_COORDINATES), "float32"),
    },
)


SOMA_CORRECTIVE_FIELDS = frozenset({"corrective_first_layer", "corrective_output_indices", "corrective_output_values"})

SOMA_SCHEMA = ModelSchema(
    name="soma",
    format=SOMA,
    fields={
        "format": FieldSpec((), "str"),
        "format_version": FieldSpec((), "int64"),
        "provenance": FieldSpec((), "str"),
        "up_axis": FieldSpec((), "<U2"),
        "shape_mean": FieldSpec(("V", 3), "float32"),
        "shape_basis": FieldSpec(("V", 3, "S"), "float32"),
        "bind_vertices": FieldSpec(("V", 3), "float32"),
        "faces": FieldSpec(("F", 3), "int64"),
        "joint_names": FieldSpec(("J",), "<U18"),
        "joint_parents": FieldSpec(("J",), "int64"),
        "bind_pose_world": FieldSpec(("J", 4, 4), "float32"),
        "t_pose_world": FieldSpec(("J", 4, 4), "float32"),
        "skin_indices": FieldSpec(("V", "K"), "int64"),
        "skin_weights": FieldSpec(("V", "K"), "float32"),
        "public_joint_ids": FieldSpec(("P",), "int64"),
        "public_parents": FieldSpec(("P",), "int64"),
        "public_skin_weights": FieldSpec(("V", "P"), "float32"),
        "joint_regressor": FieldSpec(("P", "V"), "float32"),
        "twist_joint_ids": FieldSpec(("T",), "int64"),
        "twist_segment": FieldSpec(("T",), "int64"),
        "twist_fraction": FieldSpec(("T",), "float32"),
        "twist_rotation_matrix": FieldSpec(("T", "P"), "float32"),
        "segment_start": FieldSpec((8,), "int64"),
        "segment_end": FieldSpec((8,), "int64"),
        "segment_parent": FieldSpec((8,), "int64"),
        "segment_reverse": FieldSpec((8,), "bool"),
        "corrective_bind": FieldSpec(("P", 3, 3), "float32"),
        "corrective_first_layer": FieldSpec(("C", "H"), "float32"),
        "corrective_output_indices": FieldSpec((2, "N"), "int64"),
        "corrective_output_values": FieldSpec(("N",), "float32"),
        "corrective_output_shape": FieldSpec((2,), "int64"),
        "corrective_tanh": FieldSpec((), "bool"),
    },
    dimensions={"V": 18056, "S": 128, "J": 110, "P": 78, "T": 32},
)

#: Closed part meshes over ``[vertices; joints]``: part ``p`` is ``faces[part_offsets[p]:part_offsets[p + 1]]``.
SEGMENTATION_SCHEMA = ModelSchema(
    name="segmentation",
    format=SEGMENTATION,
    fields={
        "format": FieldSpec((), "str"),
        "format_version": FieldSpec((), "int64"),
        "provenance": FieldSpec((), "str"),
        "num_vertices": FieldSpec((), "int64"),
        "faces": FieldSpec(("F", 3), "int64"),
        "part_offsets": FieldSpec(("P+1",), "int64"),
    },
)

#: Named density tables: ``densities[t, c]`` is table ``table_names[t]``'s kg/m³ for category ``categories[c]``.
DENSITY_TABLES_SCHEMA = ModelSchema(
    name="density_tables",
    format=DENSITY_TABLES,
    fields={
        "format": FieldSpec((), "str"),
        "format_version": FieldSpec((), "int64"),
        "provenance": FieldSpec((), "str"),
        "table_names": FieldSpec(("T",), "str"),
        "categories": FieldSpec(("C",), "str"),
        "densities": FieldSpec(("T", "C"), "float64"),
    },
)


def validate(
    data: Mapping[str, Any],
    schema: ModelSchema,
    *,
    source: str = "<mapping>",
) -> None:
    """Validate required fields while allowing unknown vendor metadata; raises :class:`FormatError` naming the field."""
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
    if expected == "str":
        return np.issubdtype(actual, np.str_)
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
) -> FormatError:
    return FormatError(
        f"{source}: field {field_name!r} expected {expectation}, got {actual}. "
        f"Reconvert the asset or provide a valid {schema.name} .npz file."
    )
