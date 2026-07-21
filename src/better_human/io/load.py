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
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .schema import (
    CSR_DATA,
    CSR_INDICES,
    CSR_INDPTR,
    CSR_KEYS,
    CSR_SHAPE,
    ModelSchema,
    validate,
)

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
    """Load required fields without changing their stored dtype.

    Dense arrays become dense tensors. A regressor stored as CSR components
    becomes a :func:`torch.sparse_csr_tensor`; it is never densified. Unknown
    vendor fields are validated as ignorable metadata and are not returned.
    """
    model_path = Path(path)
    if model_path.suffix.lower() != ".npz":
        raise ValueError(f"Expected an .npz model file, got: {model_path}.")
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}.")

    loaded: dict[str, torch.Tensor] = {}
    with np.load(model_path, allow_pickle=False) as archive:
        arrays = _read_archive_arrays(archive, schema, model_path)
        validate(arrays, schema, source=str(model_path))
        for name in schema.fields:
            if name == "J_regressor" and name not in arrays:
                loaded[name] = _load_sparse_regressor(arrays)
            else:
                loaded[name] = torch.from_numpy(arrays[name])
    return loaded


def _read_archive_arrays(
    archive: np.lib.npyio.NpzFile,
    schema: ModelSchema,
    model_path: Path,
) -> dict[str, Any]:
    names = set(archive.files)
    arrays: dict[str, Any] = {name: None for name in names - set(schema.fields) - CSR_KEYS}
    for name in names & (set(schema.fields) | CSR_KEYS):
        try:
            arrays[name] = np.asarray(archive[name])
        except Exception as error:
            raise ValueError(
                f"{model_path}: field {name!r} expected a non-pickled numeric array, "
                f"but it could not be read ({error}). Reconvert the asset or provide a valid "
                f"{schema.name} .npz file."
            ) from error
    return arrays


def _load_sparse_regressor(arrays: Mapping[str, Any]) -> torch.Tensor:
    if not CSR_KEYS.issubset(arrays):
        raise RuntimeError("Validated CSR components disappeared while loading.")
    shape = tuple(int(value) for value in arrays[CSR_SHAPE])
    return torch.sparse_csr_tensor(
        torch.from_numpy(arrays[CSR_INDPTR]),
        torch.from_numpy(arrays[CSR_INDICES]),
        torch.from_numpy(arrays[CSR_DATA]),
        size=shape,
        check_invariants=True,
    )
