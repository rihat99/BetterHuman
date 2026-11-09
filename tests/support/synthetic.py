"""A synthetic FLAME-shaped weight file: a real family loader over random arrays, for tests that need no weights."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

import better_human as bh

NUM_VERTICES = 50


def flame_spec(directory: Path, *, dtype: torch.dtype = torch.float64) -> bh.FamilySpec:
    """A FLAME spec (joints global, neck, jaw, left_eye, right_eye) from random arrays written under ``directory``."""
    rng = np.random.default_rng(0)
    weights, regressor = rng.random((NUM_VERTICES, 5)), rng.random((5, NUM_VERTICES))
    path = directory / "generic_model.npz"
    np.savez(
        path,
        v_template=rng.normal(size=(NUM_VERTICES, 3)) * 0.1,
        shapedirs=rng.normal(size=(NUM_VERTICES, 3, 400)) * 1e-3,
        posedirs=rng.normal(size=(NUM_VERTICES, 3, 36)) * 1e-3,
        J_regressor=regressor / regressor.sum(axis=1, keepdims=True),
        weights=weights / weights.sum(axis=1, keepdims=True),
        kintree_table=np.array([[4294967295, 0, 1, 1, 1], [0, 1, 2, 3, 4]], dtype=np.uint32),
        f=rng.integers(0, NUM_VERTICES, size=(80, 3)).astype(np.uint32),
    )
    return bh.families.flame(path, dtype=dtype)
