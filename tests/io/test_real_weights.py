from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from scipy import sparse

from better_human.io.convert_chumpy import convert_model, load_chumpy_pickle
from better_human.io.load import load_model_npz
from better_human.io.schema import SMPL_SCHEMA, SMPLX_SCHEMA

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MODELS_ROOT = REPOSITORY_ROOT / "models"


def test_all_local_smpl_npz_files_validate_and_load() -> None:
    paths = sorted((MODELS_ROOT / "smpl").glob("*.npz"))
    if not paths:
        pytest.skip("No license-gated SMPL NPZ files found under models/smpl/.")

    for path in paths:
        loaded = load_model_npz(path, SMPL_SCHEMA)
        assert set(loaded) == set(SMPL_SCHEMA.fields)
        assert loaded["J_regressor"].layout == torch.strided


def test_all_local_smplx_npz_files_ignore_vendor_extras_and_load() -> None:
    paths = sorted((MODELS_ROOT / "smplx").glob("*.npz"))
    if not paths:
        pytest.skip("No license-gated SMPL-X NPZ files found under models/smplx/.")

    for path in paths:
        loaded = load_model_npz(path, SMPLX_SCHEMA)
        assert set(loaded) == set(SMPLX_SCHEMA.fields)
        assert loaded["shapedirs"].dtype == torch.float64


def test_real_smpl_pickle_converts_losslessly_with_stub(tmp_path: Path) -> None:
    candidates = sorted((MODELS_ROOT / "smpl" / "models").glob("*neutral*.pkl"))
    if not candidates:
        pytest.skip("No original license-gated neutral SMPL pickle found under models/smpl/models/.")

    source = candidates[0]
    destination = tmp_path / "SMPL_NEUTRAL.npz"
    convert_model(source, destination, model="smpl")
    raw = load_chumpy_pickle(source)

    with np.load(destination, allow_pickle=False) as converted:
        for field_name in SMPL_SCHEMA.fields:
            if field_name == "J_regressor":
                continue
            expected = np.asarray(raw[field_name])
            actual = converted[field_name]
            assert actual.dtype == expected.dtype
            assert actual.shape == expected.shape
            assert actual.tobytes(order="C") == expected.tobytes(order="C")

        assert sparse.issparse(raw["J_regressor"])
        expected_regressor = np.asarray(raw["J_regressor"].todense())
        actual_regressor = converted["J_regressor"]
        assert actual_regressor.dtype == expected_regressor.dtype
        assert actual_regressor.shape == expected_regressor.shape
        assert actual_regressor.tobytes(order="C") == expected_regressor.tobytes(order="C")

    loaded = load_model_npz(destination, SMPL_SCHEMA)
    assert loaded["J_regressor"].layout == torch.strided


def test_converter_refuses_overwrite_before_reading_input(tmp_path: Path) -> None:
    destination = tmp_path / "existing.npz"
    destination.touch()

    with pytest.raises(FileExistsError, match=r"Refusing to overwrite.*--force"):
        convert_model(tmp_path / "missing.pkl", destination, model="smpl")
