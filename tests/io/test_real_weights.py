from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy import sparse

from better_human.io.load import load_model_npz, models_dir
from better_human.io.schema import SMPL_SCHEMA, SMPLX_SCHEMA

# A tool, not a module of the package: load it from its file.
_SPEC = importlib.util.spec_from_file_location(
    "convert_chumpy", Path(__file__).parents[2] / "tools" / "convert" / "smpl" / "convert_chumpy.py"
)
convert_chumpy = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(convert_chumpy)
convert_model, load_chumpy_pickle = convert_chumpy.convert_model, convert_chumpy.load_chumpy_pickle


@pytest.mark.licensed
def test_all_local_smpl_npz_files_validate_and_load() -> None:
    paths = sorted((models_dir() / "smpl").glob("*.npz"))
    assert paths, f"no licensed SMPL .npz file under {models_dir() / 'smpl'}"

    for path in paths:
        loaded = load_model_npz(path, SMPL_SCHEMA)
        assert set(loaded) == set(SMPL_SCHEMA.fields)
        assert loaded["J_regressor"].layout == torch.strided


@pytest.mark.licensed
def test_all_local_smplx_npz_files_ignore_vendor_extras_and_load() -> None:
    paths = sorted((models_dir() / "smplx").glob("*.npz"))
    assert paths, f"no licensed SMPL-X .npz file under {models_dir() / 'smplx'}"

    for path in paths:
        loaded = load_model_npz(path, SMPLX_SCHEMA)
        assert set(loaded) == set(SMPLX_SCHEMA.fields)
        assert loaded["shapedirs"].dtype == torch.float64


@pytest.mark.licensed
def test_real_smpl_pickle_converts_losslessly_with_stub(tmp_path: Path) -> None:
    candidates = sorted((models_dir() / "smpl" / "models").glob("*neutral*.pkl"))
    assert candidates, (
        f"the original neutral SMPL pickle (*neutral*.pkl, from the licensed SMPL download) is absent: "
        f"place it under {models_dir() / 'smpl' / 'models'}"
    )

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
