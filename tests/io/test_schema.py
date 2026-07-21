from __future__ import annotations

import logging

import numpy as np
import pytest

from better_human.io.schema import (
    CSR_DATA,
    CSR_INDICES,
    CSR_INDPTR,
    CSR_SHAPE,
    SMPL_SCHEMA,
    validate,
)


def _valid_smpl_data() -> dict[str, np.ndarray]:
    vertex_count = 4
    joint_count = 3
    return {
        "v_template": np.zeros((vertex_count, 3), dtype=np.float64),
        "shapedirs": np.zeros((vertex_count, 3, 10), dtype=np.float64),
        "posedirs": np.zeros((vertex_count, 3, (joint_count - 1) * 9), dtype=np.float64),
        "J_regressor": np.zeros((joint_count, vertex_count), dtype=np.float64),
        "weights": np.zeros((vertex_count, joint_count), dtype=np.float64),
        "kintree_table": np.zeros((2, joint_count), dtype=np.int32),
        "f": np.zeros((2, 3), dtype=np.uint32),
    }


def test_valid_schema_uses_required_subset_semantics(caplog: pytest.LogCaptureFixture) -> None:
    data = _valid_smpl_data()
    data["vendor_metadata"] = np.array([1], dtype=np.int32)

    with caplog.at_level(logging.DEBUG, logger="better_human.io.schema"):
        validate(data, SMPL_SCHEMA, source="synthetic.npz")

    assert "vendor_metadata" in caplog.text


def test_missing_field_error_names_field_file_and_remedy() -> None:
    data = _valid_smpl_data()
    del data["weights"]

    with pytest.raises(ValueError, match=r"synthetic\.npz.*'weights'.*required field.*Reconvert"):
        validate(data, SMPL_SCHEMA, source="synthetic.npz")


def test_wrong_shape_error_reports_symbolic_expectation() -> None:
    data = _valid_smpl_data()
    data["weights"] = np.zeros((4, 2), dtype=np.float64)

    with pytest.raises(ValueError, match=r"broken\.npz.*'weights'.*dimension J = 3.*got 2.*Reconvert"):
        validate(data, SMPL_SCHEMA, source="broken.npz")


def test_wrong_dtype_error_is_actionable() -> None:
    data = _valid_smpl_data()
    data["shapedirs"] = np.zeros((4, 3, 10), dtype=np.float32)

    with pytest.raises(ValueError, match=r"wrong_dtype\.npz.*'shapedirs'.*dtype float64.*dtype float32"):
        validate(data, SMPL_SCHEMA, source="wrong_dtype.npz")


def test_multiple_wrong_ranks_never_leak_an_unbound_dimension_error() -> None:
    data = _valid_smpl_data()
    del data["J_regressor"]
    data.update(
        {
            CSR_DATA: np.array([1.0], dtype=np.float64),
            CSR_INDICES: np.array([0], dtype=np.int32),
            CSR_INDPTR: np.array([0, 1, 1, 1], dtype=np.int32),
            CSR_SHAPE: np.array([3, 4], dtype=np.int64),
            "weights": np.zeros(4, dtype=np.float64),
            "kintree_table": np.zeros(3, dtype=np.int32),
        }
    )

    with pytest.raises(ValueError, match=r"broken_ranks\.npz.*'weights'.*rank 2.*Reconvert"):
        validate(data, SMPL_SCHEMA, source="broken_ranks.npz")
