"""Declarative schemas for SMPL-family weight files."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)

CSR_DATA = "J_regressor_data"
CSR_INDICES = "J_regressor_indices"
CSR_INDPTR = "J_regressor_indptr"
CSR_SHAPE = "J_regressor_shape"
CSR_KEYS = frozenset({CSR_DATA, CSR_INDICES, CSR_INDPTR, CSR_SHAPE})


@dataclass(frozen=True)
class Dimension:
    """Symbolic array dimension resolved from fields in the same schema."""

    name: str
    scale: int = 1
    offset: int = 0

    def resolve(self, bindings: Mapping[str, int]) -> int | None:
        """Resolve this expression from known dimension bindings."""
        value = bindings.get(self.name)
        return None if value is None else self.scale * value + self.offset

    def describe(self) -> str:
        """Return a compact human-readable dimension expression."""
        if self.scale == 1 and self.offset == 0:
            return self.name
        if self.scale == 1:
            sign = "+" if self.offset > 0 else "-"
            return f"{self.name} {sign} {abs(self.offset)}"
        if self.offset == -self.scale:
            return f"({self.name} - 1) * {self.scale}"
        suffix = f" + {self.offset}" if self.offset > 0 else f" - {abs(self.offset)}"
        return f"{self.scale} * {self.name}{suffix}"


ShapeAxis = int | Dimension
DTypeExpectation = str | tuple[str, ...]


@dataclass(frozen=True)
class FieldSpec:
    """Expected shape, dtype, and storage for one logical model field."""

    shape: tuple[ShapeAxis, ...]
    dtype: DTypeExpectation
    dense_or_csr: bool = False


@dataclass(frozen=True)
class ModelSchema:
    """Required fields and shared symbolic dimensions for one model family."""

    name: str
    fields: Mapping[str, FieldSpec]
    dimensions: Mapping[str, int] = field(default_factory=dict)
    minimum_dimensions: Mapping[str, int] = field(default_factory=dict)


V = Dimension("V")
J = Dimension("J")
S = Dimension("S")
F = Dimension("F")
POSE_FEATURES = Dimension("J", scale=9, offset=-9)


def _base_fields(*, shapedirs_dtype: str, other_float_dtype: str) -> dict[str, FieldSpec]:
    return {
        "v_template": FieldSpec((V, 3), other_float_dtype),
        "shapedirs": FieldSpec((V, 3, S), shapedirs_dtype),
        "posedirs": FieldSpec((V, 3, POSE_FEATURES), other_float_dtype),
        "J_regressor": FieldSpec((J, V), other_float_dtype, dense_or_csr=True),
        "weights": FieldSpec((V, J), other_float_dtype),
        "kintree_table": FieldSpec((2, J), "integer"),
        "f": FieldSpec((F, 3), "integer"),
    }


_smpl_fields = _base_fields(shapedirs_dtype="float64", other_float_dtype="float64")
_smpl_fields.update(
    {
        "kintree_table": FieldSpec((2, J), ("int32", "uint32")),
        "f": FieldSpec((F, 3), "uint32"),
    }
)
SMPL_SCHEMA = ModelSchema(name="smpl", fields=_smpl_fields, minimum_dimensions={"S": 10})

_smplx_fields = _base_fields(shapedirs_dtype="float64", other_float_dtype="float32")
_smplx_fields.update(
    {
        "kintree_table": FieldSpec((2, J), "int64"),
        "f": FieldSpec((F, 3), "uint32"),
        "hands_componentsl": FieldSpec((45, 45), "float32"),
        "hands_componentsr": FieldSpec((45, 45), "float32"),
        "hands_meanl": FieldSpec((45,), "float32"),
        "hands_meanr": FieldSpec((45,), "float32"),
    }
)
SMPLX_SCHEMA = ModelSchema(name="smplx", fields=_smplx_fields, dimensions={"S": 400})

_smplh_fields = _base_fields(shapedirs_dtype="floating", other_float_dtype="floating")
_smplh_fields.update(
    {
        "hands_componentsl": FieldSpec((45, 45), "floating"),
        "hands_componentsr": FieldSpec((45, 45), "floating"),
        "hands_meanl": FieldSpec((45,), "floating"),
        "hands_meanr": FieldSpec((45,), "floating"),
    }
)
SMPLH_SCHEMA = ModelSchema(name="smplh", fields=_smplh_fields, dimensions={"S": 16})

_mano_fields = _base_fields(shapedirs_dtype="floating", other_float_dtype="floating")
_mano_fields.update(
    {
        "hands_components": FieldSpec((45, 45), "floating"),
        "hands_mean": FieldSpec((45,), "floating"),
    }
)
MANO_SCHEMA = ModelSchema(name="mano", fields=_mano_fields, dimensions={"S": 10})

SCHEMAS: Mapping[str, ModelSchema] = {
    "smpl": SMPL_SCHEMA,
    "smplh": SMPLH_SCHEMA,
    "smplx": SMPLX_SCHEMA,
    "mano": MANO_SCHEMA,
}


def schema_for_model(name: str) -> ModelSchema:
    """Return the schema registered for a converter model name."""
    try:
        return SCHEMAS[name.lower()]
    except KeyError as error:
        supported = ", ".join(sorted(SCHEMAS))
        raise ValueError(f"Unsupported model {name!r}; choose one of: {supported}.") from error


def validate(
    data: Mapping[str, Any],
    schema: ModelSchema,
    *,
    source: str = "<mapping>",
) -> None:
    """Validate required fields while allowing unknown vendor metadata.

    ``J_regressor`` may be a dense array or the four CSR component arrays
    named by :data:`CSR_KEYS`.
    """
    keys = set(data)
    bindings = dict(schema.dimensions)

    for name, spec in schema.fields.items():
        if name in data:
            array = _as_array(data, name, source, schema)
            if array.ndim != len(spec.shape):
                raise _validation_error(
                    source,
                    name,
                    f"rank {len(spec.shape)} with shape {_format_shape(spec.shape, bindings)}",
                    f"shape {array.shape}",
                    schema,
                )
            _bind_plain_dimensions(array.shape, spec.shape, bindings, source, name, schema)
            continue
        if spec.dense_or_csr and CSR_KEYS.issubset(keys):
            continue
        expectation = _format_shape(spec.shape, bindings)
        raise _validation_error(
            source,
            name,
            f"required field with shape {expectation}",
            "missing",
            schema,
        )

    for name, minimum in schema.minimum_dimensions.items():
        actual = bindings.get(name)
        if actual is not None and actual < minimum:
            raise _validation_error(
                source,
                "shapedirs",
                f"dimension {name} >= {minimum}",
                str(actual),
                schema,
            )

    for name, spec in schema.fields.items():
        if name in data:
            _validate_array(_as_array(data, name, source, schema), spec, bindings, source, name, schema)
        else:
            _validate_csr(data, spec, bindings, source, schema)

    known_keys = set(schema.fields) | CSR_KEYS
    extras = sorted(keys - known_keys)
    if extras:
        LOGGER.debug("Ignoring unknown fields in %s: %s", source, ", ".join(extras))


def _bind_plain_dimensions(
    actual_shape: tuple[int, ...],
    expected_shape: tuple[ShapeAxis, ...],
    bindings: dict[str, int],
    source: str,
    field_name: str,
    schema: ModelSchema,
) -> None:
    if len(actual_shape) != len(expected_shape):
        return
    for actual, expected in zip(actual_shape, expected_shape, strict=True):
        if not isinstance(expected, Dimension) or expected.scale != 1 or expected.offset != 0:
            continue
        previous = bindings.setdefault(expected.name, actual)
        if previous != actual:
            raise _validation_error(
                source,
                field_name,
                f"dimension {expected.name} = {previous}",
                str(actual),
                schema,
            )


def _validate_array(
    array: np.ndarray,
    spec: FieldSpec,
    bindings: Mapping[str, int],
    source: str,
    field_name: str,
    schema: ModelSchema,
) -> None:
    expected_shape = _resolved_shape_or_error(
        spec.shape,
        bindings,
        source=source,
        field_name=field_name,
        actual_shape=array.shape,
        schema=schema,
    )
    if array.shape != expected_shape:
        raise _validation_error(
            source,
            field_name,
            f"shape {_format_shape(spec.shape, bindings)}",
            f"shape {array.shape}",
            schema,
        )
    if not _dtype_matches(array.dtype, spec.dtype):
        raise _validation_error(
            source,
            field_name,
            f"dtype {_format_dtype(spec.dtype)}",
            f"dtype {array.dtype}",
            schema,
        )


def _validate_csr(
    data: Mapping[str, Any],
    spec: FieldSpec,
    bindings: Mapping[str, int],
    source: str,
    schema: ModelSchema,
) -> None:
    values = _as_array(data, CSR_DATA, source, schema)
    indices = _as_array(data, CSR_INDICES, source, schema)
    indptr = _as_array(data, CSR_INDPTR, source, schema)
    stored_shape = _as_array(data, CSR_SHAPE, source, schema)
    _expect_component(stored_shape, (2,), "int64", CSR_SHAPE, source, schema)
    expected_shape = _resolved_shape_or_error(
        spec.shape,
        bindings,
        source=source,
        field_name="J_regressor",
        actual_shape=tuple(int(value) for value in stored_shape) if stored_shape.shape == (2,) else stored_shape.shape,
        schema=schema,
    )

    _expect_component(values, (values.size,), spec.dtype, CSR_DATA, source, schema)
    index_dtypes = ("int32", "int64")
    _expect_component(indices, (values.size,), index_dtypes, CSR_INDICES, source, schema)
    _expect_component(indptr, (expected_shape[0] + 1,), index_dtypes, CSR_INDPTR, source, schema)
    if indices.dtype != indptr.dtype:
        raise _validation_error(
            source,
            CSR_INDICES,
            f"the same dtype as {CSR_INDPTR} ({indptr.dtype})",
            f"dtype {indices.dtype}",
            schema,
        )

    actual_shape = tuple(int(value) for value in stored_shape)
    if actual_shape != expected_shape:
        raise _validation_error(
            source,
            CSR_SHAPE,
            f"value {expected_shape}",
            f"value {actual_shape}",
            schema,
        )
    if indptr.size and (int(indptr[0]) != 0 or int(indptr[-1]) != values.size):
        raise _validation_error(
            source,
            CSR_INDPTR,
            f"CSR offsets from 0 to nnz={values.size}",
            f"endpoints {(int(indptr[0]), int(indptr[-1]))}",
            schema,
        )
    if np.any(indptr[1:] < indptr[:-1]):
        raise _validation_error(source, CSR_INDPTR, "nondecreasing offsets", "decreasing offsets", schema)
    if np.any(indices < 0) or np.any(indices >= expected_shape[1]):
        raise _validation_error(
            source,
            CSR_INDICES,
            f"column indices in [0, {expected_shape[1]})",
            "out-of-range index",
            schema,
        )
    for row in range(expected_shape[0]):
        row_indices = indices[int(indptr[row]) : int(indptr[row + 1])]
        if np.any(row_indices[1:] <= row_indices[:-1]):
            raise _validation_error(
                source,
                CSR_INDICES,
                "strictly increasing, unique column indices within each row",
                f"noncanonical row {row}",
                schema,
            )


def _expect_component(
    array: np.ndarray,
    shape: tuple[int, ...],
    dtype: DTypeExpectation,
    field_name: str,
    source: str,
    schema: ModelSchema,
) -> None:
    if array.shape != shape:
        raise _validation_error(source, field_name, f"shape {shape}", f"shape {array.shape}", schema)
    if not _dtype_matches(array.dtype, dtype):
        raise _validation_error(
            source,
            field_name,
            f"dtype {_format_dtype(dtype)}",
            f"dtype {array.dtype}",
            schema,
        )


def _resolved_shape(shape: tuple[ShapeAxis, ...], bindings: Mapping[str, int]) -> tuple[int, ...]:
    resolved: list[int] = []
    for axis in shape:
        if isinstance(axis, int):
            resolved.append(axis)
            continue
        value = axis.resolve(bindings)
        if value is None:
            raise RuntimeError(f"Schema dimension {axis.name!r} was not bound.")
        resolved.append(value)
    return tuple(resolved)


def _resolved_shape_or_error(
    shape: tuple[ShapeAxis, ...],
    bindings: Mapping[str, int],
    *,
    source: str,
    field_name: str,
    actual_shape: tuple[int, ...],
    schema: ModelSchema,
) -> tuple[int, ...]:
    try:
        return _resolved_shape(shape, bindings)
    except RuntimeError as error:
        raise _validation_error(
            source,
            field_name,
            f"shape {_format_shape(shape, bindings)} with dimensions bound by the other required fields",
            f"shape {actual_shape} with an unresolved symbolic dimension",
            schema,
        ) from error


def _format_shape(shape: tuple[ShapeAxis, ...], bindings: Mapping[str, int]) -> str:
    parts: list[str] = []
    for axis in shape:
        if isinstance(axis, int):
            parts.append(str(axis))
            continue
        value = axis.resolve(bindings)
        parts.append(str(value) if value is not None else axis.describe())
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
