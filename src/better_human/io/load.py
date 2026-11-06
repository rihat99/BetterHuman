"""Find and load validated model weights.

Every weight file lives under one directory, named by ``$BETTERHUMAN_MODELS_DIR``::

    $BETTERHUMAN_MODELS_DIR/
    ├── smpl/SMPL_NEUTRAL.npz
    ├── smplx/SMPLX_NEUTRAL.npz
    └── MHR/converted/mhr_lod1.npz

:func:`models_dir` is the one reader of that variable, and an explicit ``path`` always takes precedence over it.
Licensed weights are never downloaded or cached by BetterHuman.
"""

from __future__ import annotations

import os
from collections.abc import Collection
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .schema import ModelSchema, validate

MODELS_ENV = "BETTERHUMAN_MODELS_DIR"


def models_dir() -> Path:
    """The weights directory that ``$BETTERHUMAN_MODELS_DIR`` names.

    Raises ``FileNotFoundError`` with the fix when the variable is unset or names no directory; nothing guesses a
    default location.
    """
    value = os.environ.get(MODELS_ENV)
    if not value:
        raise FileNotFoundError(
            f"${MODELS_ENV} is unset. Set it to the directory that holds the licensed weights, "
            f"for example: export {MODELS_ENV}=/data/better_human/models"
        )
    directory = Path(value).expanduser()
    if not directory.is_dir():
        raise FileNotFoundError(
            f"${MODELS_ENV}={value} is not a directory. Set it to the directory that holds the licensed weights."
        )
    return directory


def weight_path(folder: str, filename: str, *, path: str | Path | None = None) -> Path:
    """The weight file ``models_dir() / folder / filename``, or ``path`` when it is given.

    Raises ``FileNotFoundError`` naming the missing file and the fix.
    """
    if path is not None:
        resolved = Path(path).expanduser()
        if not resolved.is_file():
            raise FileNotFoundError(f"Model file not found: {resolved}. Pass path=... to an existing file.")
        return resolved
    resolved = models_dir() / folder / filename
    if not resolved.is_file():
        raise FileNotFoundError(
            f"Model file not found: {resolved}. Place {folder}/{filename} under ${MODELS_ENV} "
            f"or pass path=... to an existing file."
        )
    return resolved


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
        missing = [name for name in selected_schema.fields if name not in archive.files]
        if missing:
            raise ValueError(
                f"{model_path}: field {missing[0]!r} expected a required field, got missing. "
                f"Reconvert the asset or provide a valid {schema.name} .npz file."
            )
        arrays = _read_archive_arrays(archive, selected_schema, model_path)
        validate(arrays, selected_schema, source=str(model_path))
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
