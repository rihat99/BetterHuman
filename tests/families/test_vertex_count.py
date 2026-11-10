"""The SMPL-family loaders check the shipped data that indexes vertices against the file's vertex count (FormatError).

The SMPL closed segmentation and the SMPL-H landmarks index the authored meshes; a file with another vertex count is
not the family's file. Synthetic files of each family's joint count, without weights.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import better_human as bh
from better_human.io import FormatError

NUM_VERTICES = 50


def _write(path: Path, num_joints: int) -> Path:
    """An SMPL-family ``.npz`` with ``NUM_VERTICES`` random vertices and a chain of ``num_joints`` joints."""
    rng = np.random.default_rng(0)
    weights, regressor = rng.random((NUM_VERTICES, num_joints)), rng.random((num_joints, NUM_VERTICES))
    np.savez(
        path,
        v_template=rng.normal(size=(NUM_VERTICES, 3)) * 0.1,
        shapedirs=rng.normal(size=(NUM_VERTICES, 3, 16)) * 1e-3,
        posedirs=rng.normal(size=(NUM_VERTICES, 3, (num_joints - 1) * 9)) * 1e-3,
        J_regressor=regressor / regressor.sum(axis=1, keepdims=True),
        weights=weights / weights.sum(axis=1, keepdims=True),
        kintree_table=np.array([[4294967295, *range(num_joints - 1)], list(range(num_joints))], dtype=np.uint32),
        f=rng.integers(0, NUM_VERTICES, size=(80, 3)).astype(np.uint32),
    )
    return path


def test_smpl_raises_for_a_file_its_segmentation_does_not_fit(tmp_path: Path) -> None:
    path = _write(tmp_path / "SMPL_NEUTRAL.npz", num_joints=24)
    with pytest.raises(FormatError, match=r"smpl weights hold 50 vertices; the shipped smpl segmentation needs 6890"):
        bh.families.smpl(path)


def test_smplh_raises_for_a_file_its_landmarks_do_not_fit(tmp_path: Path) -> None:
    path = _write(tmp_path / "model.npz", num_joints=52)
    with pytest.raises(FormatError, match=r"smplh weights hold 50 vertices; the smplh landmarks index vertex 6787"):
        bh.families.smplh(path)
