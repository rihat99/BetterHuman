"""Resolve and load validated model weights.

The conventional local layout is::

    $BETTERHUMAN_MODELS_DIR/
    ├── smpl/SMPL_NEUTRAL.npz
    └── smplx/SMPLX_NEUTRAL.npz

An explicit ``model_path`` always takes precedence. Licensed weights are never
downloaded or cached by BetterHuman.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

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


def load_model_npz(path: str | Path, schema: ModelSchema) -> dict[str, torch.Tensor]:
    """Load required fields as dense tensors without changing their stored dtype.

    Unknown vendor fields are validated as ignorable metadata and are not
    returned.
    """
    model_path = Path(path)
    if model_path.suffix.lower() != ".npz":
        raise ValueError(f"Expected an .npz model file, got: {model_path}.")
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}.")

    with np.load(model_path, allow_pickle=False) as archive:
        arrays = _read_archive_arrays(archive, schema, model_path)
        validate(arrays, schema, source=str(model_path))
        return {name: torch.from_numpy(arrays[name]) for name in schema.fields}


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
