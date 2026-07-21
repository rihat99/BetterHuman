from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from better_human.io.load import load_model_npz, resolve_model_path
from better_human.io.schema import SMPL_SCHEMA


def _write_dense_fixture(path: Path, *, object_template: bool = False) -> None:
    vertex_count = 4
    joint_count = 3
    np.savez(
        path,
        v_template=(
            np.full((vertex_count, 3), object(), dtype=object)
            if object_template
            else np.zeros((vertex_count, 3), dtype=np.float64)
        ),
        shapedirs=np.zeros((vertex_count, 3, 10), dtype=np.float64),
        posedirs=np.zeros((vertex_count, 3, (joint_count - 1) * 9), dtype=np.float64),
        J_regressor=np.zeros((joint_count, vertex_count), dtype=np.float64),
        weights=np.zeros((vertex_count, joint_count), dtype=np.float64),
        kintree_table=np.zeros((2, joint_count), dtype=np.int32),
        f=np.zeros((2, 3), dtype=np.uint32),
    )


def test_explicit_model_path_takes_precedence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    explicit = tmp_path / "explicit.npz"
    explicit.touch()
    monkeypatch.setenv("BETTERHUMAN_MODELS_DIR", str(tmp_path / "unused"))

    assert resolve_model_path("smpl", "SMPL_NEUTRAL.npz", explicit) == explicit


def test_model_path_falls_back_to_environment_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = tmp_path / "smpl" / "SMPL_NEUTRAL.npz"
    expected.parent.mkdir()
    expected.touch()
    monkeypatch.setenv("BETTERHUMAN_MODELS_DIR", str(tmp_path))

    assert resolve_model_path("smpl", "SMPL_NEUTRAL.npz") == expected


def test_missing_model_path_names_both_remedies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BETTERHUMAN_MODELS_DIR", raising=False)

    with pytest.raises(FileNotFoundError, match=r"model_path=.*BETTERHUMAN_MODELS_DIR"):
        resolve_model_path("smpl", "SMPL_NEUTRAL.npz")


def test_dense_regressor_loads_without_dtype_conversion(tmp_path: Path) -> None:
    model_path = tmp_path / "dense.npz"
    _write_dense_fixture(model_path)

    loaded = load_model_npz(model_path, SMPL_SCHEMA)

    regressor = loaded["J_regressor"]
    assert regressor.layout == torch.strided
    assert regressor.dtype == torch.float64
    assert regressor.shape == (3, 4)
    assert loaded["f"].dtype == torch.uint32


def test_object_valued_required_field_is_an_actionable_load_error(tmp_path: Path) -> None:
    model_path = tmp_path / "object_template.npz"
    _write_dense_fixture(model_path, object_template=True)

    with pytest.raises(
        ValueError,
        match=r"object_template\.npz.*'v_template'.*non-pickled numeric array.*Reconvert",
    ):
        load_model_npz(model_path, SMPL_SCHEMA)
