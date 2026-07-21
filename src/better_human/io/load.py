"""Resolve and load validated model weights.

The conventional local layout is::

    $BETTERHUMAN_MODELS_DIR/
    ├── smpl/SMPL_NEUTRAL.npz
    ├── smplx/SMPLX_NEUTRAL.npz
    └── MHR/converted/mhr_lod1.npz

An explicit ``model_path`` always takes precedence. Licensed weights are never
downloaded or cached by BetterHuman.
"""

from __future__ import annotations

import os
from collections.abc import Collection
from pathlib import Path
from typing import Any
from zipfile import BadZipFile, ZipFile

import numpy as np
import torch

from .schema import ModelSchema, validate

MODELS_ENV = "BETTERHUMAN_MODELS_DIR"


def resolve_model_path(
    family: str,
    filename: str,
    model_path: str | Path | None = None,
) -> Path:
    """Resolve a licensed model file from an explicit path or environment root."""
    if model_path is not None:
        path = Path(model_path).expanduser()
    else:
        models_root = os.environ.get(MODELS_ENV)
        if models_root is None:
            raise FileNotFoundError(
                f"No {family} model path was provided. Pass model_path=... or set "
                f"${MODELS_ENV} so {family}/{filename} exists."
            )
        path = Path(models_root).expanduser() / family / filename

    if not path.is_file():
        raise FileNotFoundError(
            f"Model file not found: {path}. Pass model_path=... to an existing file or set "
            f"${MODELS_ENV} so {family}/{filename} exists."
        )
    return path


def load_model_npz(
    path: str | Path,
    schema: ModelSchema,
    *,
    exclude_fields: Collection[str] = (),
) -> dict[str, torch.Tensor | np.ndarray]:
    """Load required fields without changing their stored dtype.

    Numeric arrays become dense tensors. Unicode metadata remains NumPy arrays
    because PyTorch has no string tensor dtype. Unknown vendor fields and
    explicitly excluded schema fields are not returned.
    """
    model_path = Path(path)
    if model_path.suffix.lower() != ".npz":
        raise ValueError(f"Expected an .npz model file, got: {model_path}.")
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}.")

    excluded = frozenset(exclude_fields)
    unknown_exclusions = excluded - set(schema.fields)
    if unknown_exclusions:
        raise ValueError(
            f"exclude_fields contains names outside the {schema.name} schema: {sorted(unknown_exclusions)}"
        )
    selected_schema = ModelSchema(
        name=schema.name,
        fields={name: spec for name, spec in schema.fields.items() if name not in excluded},
        dimensions=schema.dimensions,
    )

    with np.load(model_path, allow_pickle=False) as archive:
        missing = [name for name in schema.fields if name not in archive.files]
        if missing:
            raise ValueError(
                f"{model_path}: field {missing[0]!r} expected a required field, got missing. "
                f"Reconvert the asset or provide a valid {schema.name} .npz file."
            )
        arrays = _read_archive_arrays(archive, selected_schema, model_path)
        arrays.update(_read_array_header_views(model_path, excluded, schema))
        validate(arrays, schema, source=str(model_path))
        return {name: _to_loaded_value(arrays[name]) for name in selected_schema.fields}


def _to_loaded_value(array: np.ndarray) -> torch.Tensor | np.ndarray:
    return array if array.dtype.kind == "U" else torch.from_numpy(array)


def _read_archive_arrays(
    archive: np.lib.npyio.NpzFile,
    schema: ModelSchema,
    model_path: Path,
) -> dict[str, Any]:
    names = set(archive.files)
    arrays: dict[str, Any] = {name: None for name in names - set(schema.fields)}
    for name in names & set(schema.fields):
        try:
            arrays[name] = np.asarray(archive[name])
        except Exception as error:
            raise ValueError(
                f"{model_path}: field {name!r} expected a non-pickled numeric array, "
                f"but it could not be read ({error}). Reconvert the asset or provide a valid "
                f"{schema.name} .npz file."
            ) from error
    return arrays


def _read_array_header_views(
    model_path: Path,
    field_names: Collection[str],
    schema: ModelSchema,
) -> dict[str, np.ndarray]:
    """Read shape/dtype metadata for excluded fields without their payloads."""
    views: dict[str, np.ndarray] = {}
    try:
        with ZipFile(model_path) as archive:
            for name in field_names:
                with archive.open(f"{name}.npy") as member:
                    version = np.lib.format.read_magic(member)
                    if version == (1, 0):
                        shape, _, dtype = np.lib.format.read_array_header_1_0(member)
                    elif version == (2, 0):
                        shape, _, dtype = np.lib.format.read_array_header_2_0(member)
                    else:
                        raise ValueError(f"unsupported NPY header version {version}")
                views[name] = np.broadcast_to(np.empty((), dtype=dtype), shape)
    except (BadZipFile, EOFError, KeyError, ValueError) as error:
        raise ValueError(
            f"{model_path}: excluded field metadata could not be read ({error}). "
            f"Reconvert the asset or provide a valid {schema.name} .npz file."
        ) from error
    return views
