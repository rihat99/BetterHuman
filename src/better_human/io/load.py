"""Find and load validated model weights.

Every weight file lives under one directory, named by ``$BETTERHUMAN_MODELS_DIR``, in one folder per family:
``$BETTERHUMAN_MODELS_DIR/smpl/SMPL_NEUTRAL.npz``, ``$BETTERHUMAN_MODELS_DIR/smplx/SMPLX_NEUTRAL.npz``,
``$BETTERHUMAN_MODELS_DIR/MHR/converted/mhr_lod1.npz``.

:func:`models_dir` is the one reader of that variable, and an explicit ``path`` always takes precedence over it.
Licensed weights are never downloaded or cached by BetterHuman.
"""

from __future__ import annotations

import os
import zipfile
from collections.abc import Collection
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
import torch
from better_robot.exceptions import ArgumentError, BetterRobotError

from .formats import FormatError, header, past_fields, upgrade
from .schema import ModelSchema, validate

MODELS_ENV = "BETTERHUMAN_MODELS_DIR"


class AssetNotFoundError(BetterRobotError, FileNotFoundError):
    """The weights directory or a weight file is missing; the message names ``$BETTERHUMAN_MODELS_DIR`` and the fix."""


def models_dir() -> Path:
    """The weights directory that ``$BETTERHUMAN_MODELS_DIR`` names.

    Raises :class:`AssetNotFoundError` with the fix when the variable is unset or names no directory; nothing guesses
    a default location.
    """
    value = os.environ.get(MODELS_ENV)
    if not value:
        raise AssetNotFoundError(
            f"${MODELS_ENV} is unset. Set it to the directory that holds the licensed weights, "
            f"for example: export {MODELS_ENV}=/data/better_human/models, or pass path=... to the loader to name "
            "the weight file directly."
        )
    directory = Path(value).expanduser()
    if not directory.is_dir():
        raise AssetNotFoundError(
            f"${MODELS_ENV}={value} is not a directory. Set it to the directory that holds the licensed weights."
        )
    return directory


def weight_path(folder: str, filename: str, *, path: str | Path | None = None) -> Path:
    """The weight file ``models_dir() / folder / filename``, or ``path`` when it is given.

    Raises :class:`AssetNotFoundError` naming the missing file and the fix.
    """
    if path is not None:
        resolved = Path(path).expanduser()
        if not resolved.is_file():
            raise AssetNotFoundError(
                f"Model file not found: {resolved}. Pass path=... to an existing file, or leave path unset to read "
                f"{folder}/{filename} under ${MODELS_ENV}."
            )
        return resolved
    resolved = models_dir() / folder / filename
    if not resolved.is_file():
        raise AssetNotFoundError(
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
    explicitly excluded schema fields are not returned. A schema with a
    ``format`` reads an older ``format_version`` through its in-memory
    migration (:func:`~better_human.io.formats.upgrade`), reading the fields
    only that version carries for it (:func:`~better_human.io.formats.past_fields`).
    Raises :class:`~better_human.io.formats.FormatError` for a malformed file.
    """
    model_path = Path(path)
    if model_path.suffix.lower() != ".npz":
        raise FormatError(f"Expected an .npz model file, got: {model_path}.")

    excluded = frozenset(exclude_fields)
    unknown_exclusions = excluded - set(schema.fields)
    if unknown_exclusions:
        raise ArgumentError(
            f"exclude_fields names {sorted(unknown_exclusions)}, which the {schema.name} schema does not have; "
            "pass names of its fields only."
        )
    selected_schema = ModelSchema(
        name=schema.name,
        fields={name: spec for name, spec in schema.fields.items() if name not in excluded},
        dimensions=schema.dimensions,
        format=schema.format,
    )

    names = set(selected_schema.fields)
    if schema.format is not None:
        names |= set(header(schema.format))  # read even when excluded: the version selects the migrations
    with model_path.open("rb") as file, _open_archive(file, model_path, schema.name) as archive:
        arrays = _read_archive_arrays(archive, names, schema.name, model_path)
        if schema.format is not None:
            for name in past_fields(arrays, format=schema.format, source=str(model_path)):
                if name in archive.files:
                    arrays[name] = np.asarray(archive[name])
    if schema.format is not None:
        arrays = upgrade(arrays, format=schema.format, source=str(model_path))
    validate(arrays, selected_schema, source=str(model_path))
    return {name: _to_loaded_value(np.asarray(arrays[name])) for name in selected_schema.fields}


def _open_archive(file: BinaryIO, model_path: Path, schema_name: str) -> np.lib.npyio.NpzFile:
    # The caller owns the open file: np.load leaks the file it opened itself when the zip directory is unreadable.
    try:
        archive = np.load(file, allow_pickle=False)
    except (zipfile.BadZipFile, ValueError, OSError, EOFError) as error:
        cause = type(error).__name__
    else:
        if isinstance(archive, np.lib.npyio.NpzFile):
            return archive
        cause = "a single .npy array"
    # Raised outside the except clause: numpy's ValueError for a non-archive advises loading it unsafely.
    raise FormatError(
        f"{model_path}: not a readable .npz archive ({cause}); the file is truncated, corrupt or of another kind. "
        f"Copy or reconvert it again, or pass a valid {schema_name} .npz file."
    )


def _to_loaded_value(array: np.ndarray) -> torch.Tensor | np.ndarray:
    return array if array.dtype.kind == "U" else torch.from_numpy(array)


def _read_archive_arrays(
    archive: np.lib.npyio.NpzFile,
    names: Collection[str],
    schema_name: str,
    model_path: Path,
) -> dict[str, Any]:
    arrays: dict[str, Any] = {}
    for name in set(archive.files) & set(names):
        try:
            arrays[name] = np.asarray(archive[name])
        except Exception as error:
            raise FormatError(
                f"{model_path}: field {name!r} expected a non-pickled numeric array, "
                f"but it could not be read ({error}). Reconvert the asset or provide a valid "
                f"{schema_name} .npz file."
            ) from error
    return arrays
