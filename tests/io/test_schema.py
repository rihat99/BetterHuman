from __future__ import annotations

import logging

import numpy as np
import pytest

from better_human.io.schema import MHR_SCHEMA, SMPL_SCHEMA, validate


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
    data["weights"] = np.zeros(4, dtype=np.float64)
    data["kintree_table"] = np.zeros(3, dtype=np.int32)

    with pytest.raises(ValueError, match=r"broken_ranks\.npz.*'weights'.*rank 2.*Reconvert"):
        validate(data, SMPL_SCHEMA, source="broken_ranks.npz")


def test_mhr_schema_accepts_the_complete_format_v1_mapping(mhr_data: dict[str, np.ndarray]) -> None:
    assert set(mhr_data) == set(MHR_SCHEMA.fields)
    validate(mhr_data, MHR_SCHEMA, source="mhr_lod6.npz")


def test_mhr_schema_binds_vertex_count_across_geometry_fields(mhr_data: dict[str, np.ndarray]) -> None:
    mhr_data["expression_basis"] = np.zeros((5, 3, 72), dtype=np.float32)

    with pytest.raises(ValueError, match=r"mhr_lod6\.npz.*'expression_basis'.*dimension V = 4.*got 5"):
        validate(mhr_data, MHR_SCHEMA, source="mhr_lod6.npz")


# Each case mutates one key of a fresh fixture (the old sequential test restored the previous key first).
@pytest.mark.parametrize(
    ("key", "bad_value", "source", "match"),
    [
        pytest.param(
            "corrective_output_weights",
            np.zeros((3000, 13), dtype=np.float32),
            "mhr_lod6.npz",
            r"mhr_lod6\.npz.*'corrective_output_weights'.*V\*3.*shape",
            id="flattened_corrective_width",
        ),
        pytest.param(
            "lod", np.zeros(1, dtype=np.int64), "wrong_scalar.npz", r"'lod'.*rank 0.*shape \(1,\)", id="scalar_lod"
        ),
        pytest.param(
            "model_parameter_channel_types",
            np.zeros(204, dtype=np.int64),
            "wrong_dtype.npz",
            r"'model_parameter_channel_types'.*dtype int8.*dtype int64",
            id="storage_dtype",
        ),
        pytest.param(
            "joint_names",
            np.full(127, "joint", dtype="<U21"),
            "wrong_string_dtype.npz",
            r"'joint_names'.*dtype <U20.*dtype <U21",
            id="string_dtype",
        ),
    ],
)
def test_mhr_schema_rejects(
    mhr_data: dict[str, np.ndarray], key: str, bad_value: np.ndarray, source: str, match: str
) -> None:
    mhr_data[key] = bad_value
    with pytest.raises(ValueError, match=match):
        validate(mhr_data, MHR_SCHEMA, source=source)
