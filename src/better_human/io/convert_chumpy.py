"""Convert legacy Chumpy-backed SMPL-family pickles into validated NPZ files."""

from __future__ import annotations

import argparse
import inspect
import pickle
import sys
import types
from collections import namedtuple
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

from .schema import (
    CSR_DATA,
    CSR_INDICES,
    CSR_INDPTR,
    CSR_SHAPE,
    ModelSchema,
    schema_for_model,
    validate,
)

FIELD_ALIASES: Mapping[str, tuple[str, ...]] = {
    "f": ("f", "faces"),
    "weights": ("weights", "lbs_weights"),
}


def ensure_chumpy_pickle_compatibility() -> None:
    """Install the minimal compatibility surface needed by legacy pickles."""
    if not hasattr(inspect, "getargspec"):
        arg_spec = namedtuple("ArgSpec", "args varargs keywords defaults")

        def getargspec(function: Any) -> tuple[Any, ...]:
            spec = inspect.getfullargspec(function)
            return arg_spec(spec.args, spec.varargs, spec.varkw, spec.defaults)

        inspect.getargspec = getargspec  # type: ignore[attr-defined]

    removed_numpy_aliases: Mapping[str, type[Any]] = {
        "bool": bool,
        "int": int,
        "float": float,
        "complex": complex,
        "object": object,
        "str": str,
        "unicode": str,
    }
    for name, value in removed_numpy_aliases.items():
        if name not in np.__dict__:
            setattr(np, name, value)

    try:
        import chumpy  # noqa: F401, PLC0415
    except ModuleNotFoundError:
        _install_chumpy_stub()


def _install_chumpy_stub() -> None:
    class Ch:
        def __setstate__(self, state: dict[str, Any]) -> None:
            self.__dict__.update(state)

        @property
        def r(self) -> np.ndarray:
            return np.asarray(self)

        @property
        def shape(self) -> tuple[int, ...]:
            return self.r.shape

        def __array__(self, dtype: np.dtype[Any] | type[Any] | None = None) -> np.ndarray:
            if not hasattr(self, "x"):
                raise TypeError("Unsupported Chumpy object without stored 'x' data.")
            array = np.asarray(self.x)
            return array.astype(dtype, copy=False) if dtype is not None else array

    class Select(Ch):
        def __array__(self, dtype: np.dtype[Any] | type[Any] | None = None) -> np.ndarray:
            source = np.asarray(self.a).reshape(-1)
            selected = source[np.asarray(self.idxs, dtype=np.int64)]
            preferred_shape = getattr(self, "preferred_shape", None)
            if preferred_shape is not None:
                selected = selected.reshape(preferred_shape)
            return selected.astype(dtype, copy=False) if dtype is not None else selected

    Ch.__module__ = "chumpy.ch"
    Select.__module__ = "chumpy.reordering"

    chumpy_module = types.ModuleType("chumpy")
    ch_module = types.ModuleType("chumpy.ch")
    reordering_module = types.ModuleType("chumpy.reordering")
    ch_module.Ch = Ch
    reordering_module.Select = Select
    chumpy_module.Ch = Ch
    chumpy_module.ch = ch_module
    chumpy_module.reordering = reordering_module

    sys.modules.setdefault("chumpy", chumpy_module)
    sys.modules.setdefault("chumpy.ch", ch_module)
    sys.modules.setdefault("chumpy.reordering", reordering_module)


def load_chumpy_pickle(path: str | Path) -> dict[str, Any]:
    """Load a legacy model pickle using the Chumpy compatibility stub."""
    model_path = Path(path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Input model file not found: {model_path}.")
    ensure_chumpy_pickle_compatibility()
    with model_path.open("rb") as handle:
        loaded = pickle.load(handle, encoding="latin1")
    if not isinstance(loaded, dict):
        raise TypeError(f"Expected {model_path} to contain a dictionary, got {type(loaded).__name__}.")
    return loaded


def convert_model(
    input_path: str | Path,
    output_path: str | Path,
    *,
    model: str,
    force: bool = False,
) -> Path:
    """Convert one legacy model pickle and return the written NPZ path."""
    source = Path(input_path)
    destination = Path(output_path)
    if destination.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite existing file: {destination}. Pass --force to replace it.")

    schema = schema_for_model(model)
    raw = load_chumpy_pickle(source)
    payload = _conversion_payload(raw, schema, source)
    validate(payload, schema, source=str(destination))
    with destination.open("wb") as handle:
        np.savez(handle, **payload)
    return destination


def _conversion_payload(
    raw: Mapping[str, Any],
    schema: ModelSchema,
    source: Path,
) -> dict[str, np.ndarray]:
    payload: dict[str, np.ndarray] = {}
    for field_name in schema.fields:
        value = _required_value(raw, field_name, source)
        if field_name == "J_regressor" and sparse.issparse(value):
            regressor = value.tocsr(copy=True)
            regressor.sum_duplicates()
            regressor.sort_indices()
            payload[CSR_DATA] = np.asarray(regressor.data)
            payload[CSR_INDICES] = np.asarray(regressor.indices)
            payload[CSR_INDPTR] = np.asarray(regressor.indptr)
            payload[CSR_SHAPE] = np.asarray(regressor.shape, dtype=np.int64)
        else:
            payload[field_name] = np.asarray(value)
    return payload


def _required_value(raw: Mapping[str, Any], field_name: str, source: Path) -> Any:
    for candidate in FIELD_ALIASES.get(field_name, (field_name,)):
        if candidate in raw:
            return raw[candidate]
    raise KeyError(f"{source}: missing required field {field_name!r}; cannot create a schema-valid NPZ file.")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Original Chumpy-backed .pkl model")
    parser.add_argument("output", type=Path, help="Destination .npz file")
    parser.add_argument("--model", required=True, choices=("smpl", "smplh", "mano"))
    parser.add_argument("--force", action="store_true", help="Replace an existing output file")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line converter."""
    arguments = _parser().parse_args(argv)
    convert_model(arguments.input, arguments.output, model=arguments.model, force=arguments.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
