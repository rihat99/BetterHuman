"""Synthetic SMPL-family weight files: the real family loaders over random arrays, for tests that need no weights."""

from __future__ import annotations

from functools import cache
from pathlib import Path

import numpy as np
import torch

import better_human as bh
from better_human.families import _tables

#: ``family -> (joint parents, shapedirs width, file name)``. FLAME and MANO (the wrist and five three-joint fingers)
#: have their real trees; the others are chains.
LAYOUTS = {
    "smpl": ((-1, *range(23)), 16, "SMPL_NEUTRAL.npz"),
    "smplh": ((-1, *range(51)), 16, "model.npz"),
    "smplx": ((-1, *range(54)), 400, "SMPLX_NEUTRAL.npz"),
    "mano": ((-1, *(0 if joint % 3 == 1 else joint - 1 for joint in range(1, 16))), 10, "MANO_RIGHT.npz"),
    "flame": ((-1, 0, 1, 1, 1), 400, "generic_model.npz"),
}


@cache
def _arrays(family: str, num_vertices: int) -> dict[str, np.ndarray]:
    """Random arrays of a ``family`` file with ``num_vertices`` vertices, in the dtypes its schema requires."""
    parents, width, _ = LAYOUTS[family]
    num_joints = len(parents)
    rng = np.random.default_rng(0)
    floats = np.float32 if family == "smplx" else np.float64  # SMPL-X stores float32, its shapedirs float64
    # FLAME's schema takes any floating shapedirs: float32 halves the 5023 x 3 x 400 array every FLAME test writes.
    shapedirs = np.float32 if family == "flame" else np.float64
    weights, regressor = rng.random((num_vertices, num_joints)), rng.random((num_joints, num_vertices))
    arrays = {
        "v_template": (rng.normal(size=(num_vertices, 3)) * 0.1).astype(floats),
        "shapedirs": (rng.normal(size=(num_vertices, 3, width)) * 1e-3).astype(shapedirs),
        "posedirs": (rng.normal(size=(num_vertices, 3, (num_joints - 1) * 9)) * 1e-3).astype(floats),
        "J_regressor": (regressor / regressor.sum(axis=1, keepdims=True)).astype(floats),
        "weights": (weights / weights.sum(axis=1, keepdims=True)).astype(floats),
        "kintree_table": np.array(
            [[4294967295, *parents[1:]], list(range(num_joints))], dtype=np.int64 if family == "smplx" else np.uint32
        ),
        "f": rng.integers(0, num_vertices, size=(80, 3)).astype(np.uint32),
    }
    if family == "smplx":
        for side in "lr":
            arrays[f"hands_components{side}"] = np.eye(45, dtype=np.float32)
            arrays[f"hands_mean{side}"] = np.zeros(45, dtype=np.float32)
    if family == "mano":  # an orthogonal hand PCA, so its pseudo-inverse round trip is exact
        arrays["hands_components"], arrays["hands_mean"] = (
            np.linalg.qr(rng.normal(size=(45, 45)))[0],
            rng.normal(size=45) * 0.1,
        )
    return arrays


def write_family(directory: Path, family: str, *, num_vertices: int | None = None) -> Path:
    """Write a ``family`` file under ``directory`` with ``num_vertices`` vertices (default: the family's own count).

    The arrays are random and the same on every call; a file with another vertex count is not the family's file.
    """
    count = getattr(_tables, family.upper()).num_vertices if num_vertices is None else num_vertices
    path = directory / LAYOUTS[family][2]
    np.savez(path, **_arrays(family, count))
    return path


def flame_spec(directory: Path, *, dtype: torch.dtype = torch.float64) -> bh.FamilySpec:
    """A FLAME spec (joints global, neck, jaw, left_eye, right_eye) from a synthetic file written under ``directory``."""
    return bh.families.flame(write_family(directory, "flame"), dtype=dtype)
