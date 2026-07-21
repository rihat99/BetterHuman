from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from better_human.io.load import load_model_npz, resolve_model_path
from better_human.io.schema import (
    CSR_DATA,
    CSR_INDICES,
    CSR_INDPTR,
    CSR_SHAPE,
    SMPL_SCHEMA,
)


def _write_sparse_fixture(
    path: Path,
    *,
    noncanonical_indices: np.ndarray | None = None,
    mixed_index_dtypes: bool = False,
    stored_shape: np.ndarray | None = None,
    object_template: bool = False,
) -> None:
    vertex_count = 4
    joint_count = 3
    indices = noncanonical_indices if noncanonical_indices is not None else np.array([0, 3], dtype=np.int32)
    indptr = (
        np.array([0, 2, 2, 2], dtype=np.int32)
        if noncanonical_indices is not None
        else np.array([0, 1, 1, 2], dtype=np.int32)
    )
    if mixed_index_dtypes:
        indptr = indptr.astype(np.int64)
    np.savez(
        path,
        v_template=(
            np.full((vertex_count, 3), object(), dtype=object)
            if object_template
            else np.zeros((vertex_count, 3), dtype=np.float64)
        ),
        shapedirs=np.zeros((vertex_count, 3, 10), dtype=np.float64),
        posedirs=np.zeros((vertex_count, 3, (joint_count - 1) * 9), dtype=np.float64),
        weights=np.zeros((vertex_count, joint_count), dtype=np.float64),
        kintree_table=np.zeros((2, joint_count), dtype=np.int32),
        f=np.zeros((2, 3), dtype=np.uint32),
        **{
            CSR_DATA: np.array([0.25, 0.75], dtype=np.float64),
            CSR_INDICES: indices,
            CSR_INDPTR: indptr,
            CSR_SHAPE: (
                stored_shape
                if stored_shape is not None
                else np.array([joint_count, vertex_count], dtype=np.int64)
            ),
        },
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


def test_sparse_regressor_loads_as_csr_without_dtype_conversion(tmp_path: Path) -> None:
    model_path = tmp_path / "sparse.npz"
    _write_sparse_fixture(model_path)

    loaded = load_model_npz(model_path, SMPL_SCHEMA)

    regressor = loaded["J_regressor"]
    assert regressor.layout == torch.sparse_csr
    assert regressor.dtype == torch.float64
    assert regressor.shape == (3, 4)
    assert loaded["f"].dtype == torch.uint32


@pytest.mark.parametrize(
    "indices",
    [
        np.array([3, 0], dtype=np.int32),
        np.array([1, 1], dtype=np.int32),
    ],
)
def test_noncanonical_csr_fails_before_torch_construction(tmp_path: Path, indices: np.ndarray) -> None:
    model_path = tmp_path / "malformed_csr.npz"
    _write_sparse_fixture(model_path, noncanonical_indices=indices)

    with pytest.raises(
        ValueError,
        match=r"malformed_csr\.npz.*'J_regressor_indices'.*strictly increasing, unique.*Reconvert",
    ):
        load_model_npz(model_path, SMPL_SCHEMA)


def test_mixed_csr_index_dtypes_fail_before_torch_construction(tmp_path: Path) -> None:
    model_path = tmp_path / "mixed_index_dtypes.npz"
    _write_sparse_fixture(model_path, mixed_index_dtypes=True)

    with pytest.raises(
        ValueError,
        match=r"mixed_index_dtypes\.npz.*'J_regressor_indices'.*same dtype.*J_regressor_indptr.*Reconvert",
    ):
        load_model_npz(model_path, SMPL_SCHEMA)


def test_nonnumeric_csr_shape_is_an_actionable_schema_error(tmp_path: Path) -> None:
    model_path = tmp_path / "nonnumeric_shape.npz"
    _write_sparse_fixture(model_path, stored_shape=np.array(["bad", "shape"]))

    with pytest.raises(
        ValueError,
        match=r"nonnumeric_shape\.npz.*'J_regressor_shape'.*dtype int64.*Reconvert",
    ):
        load_model_npz(model_path, SMPL_SCHEMA)


def test_object_valued_required_field_is_an_actionable_load_error(tmp_path: Path) -> None:
    model_path = tmp_path / "object_template.npz"
    _write_sparse_fixture(model_path, object_template=True)

    with pytest.raises(
        ValueError,
        match=r"object_template\.npz.*'v_template'.*non-pickled numeric array.*Reconvert",
    ):
        load_model_npz(model_path, SMPL_SCHEMA)
