from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from better_human.io.load import load_model_npz, resolve_model_path
from better_human.io.schema import MHR_CORRECTIVE_FIELDS, MHR_SCHEMA, SMPL_SCHEMA


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


def test_mhr_resolution_uses_nested_canonical_layout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = tmp_path / "MHR" / "converted" / "mhr_lod6.npz"
    expected.parent.mkdir(parents=True)
    expected.touch()
    monkeypatch.setenv("BETTERHUMAN_MODELS_DIR", str(tmp_path))

    assert resolve_model_path("MHR", "converted/mhr_lod6.npz") == expected


def test_mhr_loader_keeps_unicode_metadata_and_tensorizes_numeric_fields(
    tmp_path: Path,
    mhr_data: dict[str, np.ndarray],
) -> None:
    model_path = tmp_path / "mhr_lod6.npz"
    np.savez(model_path, **mhr_data)

    loaded = load_model_npz(model_path, MHR_SCHEMA)

    assert isinstance(loaded["joint_names"], np.ndarray)
    assert loaded["joint_names"].dtype == np.dtype("<U20")
    assert isinstance(loaded["template_vertices"], torch.Tensor)
    assert loaded["template_vertices"].dtype == torch.float32


def test_mhr_loader_can_skip_corrective_members_without_indexing_them(
    tmp_path: Path,
    mhr_data: dict[str, np.ndarray],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_path / "mhr_lod6.npz"
    np.savez(model_path, **mhr_data)
    original_getitem = np.lib.npyio.NpzFile.__getitem__

    def reject_corrective_indexing(archive: np.lib.npyio.NpzFile, name: str) -> np.ndarray:
        if name in MHR_CORRECTIVE_FIELDS:
            raise AssertionError(f"corrective payload {name!r} was indexed")
        return original_getitem(archive, name)

    monkeypatch.setattr(np.lib.npyio.NpzFile, "__getitem__", reject_corrective_indexing)

    loaded = load_model_npz(model_path, MHR_SCHEMA, exclude_fields=MHR_CORRECTIVE_FIELDS)

    assert not (set(loaded) & MHR_CORRECTIVE_FIELDS)
    assert loaded["template_vertices"].shape == (4, 3)


def test_mhr_loader_validates_excluded_corrective_headers(
    tmp_path: Path,
    mhr_data: dict[str, np.ndarray],
) -> None:
    model_path = tmp_path / "mhr_lod6.npz"
    mhr_data["corrective_output_weights"] = np.empty((3000, 12), dtype=object)
    np.savez(model_path, **mhr_data)

    with pytest.raises(ValueError, match=r"'corrective_output_weights'.*dtype float32.*dtype object"):
        load_model_npz(model_path, MHR_SCHEMA, exclude_fields=MHR_CORRECTIVE_FIELDS)


def test_mhr_loader_uses_excluded_headers_to_bind_schema_dimensions(
    tmp_path: Path,
    mhr_data: dict[str, np.ndarray],
) -> None:
    model_path = tmp_path / "mhr_lod6.npz"
    np.savez(model_path, **mhr_data)
    excluded = {
        "template_vertices",
        "skin_indices",
        "skin_weights",
        "identity_basis",
        "expression_basis",
    }

    loaded = load_model_npz(model_path, MHR_SCHEMA, exclude_fields=excluded)

    assert loaded["corrective_output_weights"].shape == (3000, 12)


def test_model_loader_rejects_unknown_field_exclusions(
    tmp_path: Path,
    mhr_data: dict[str, np.ndarray],
) -> None:
    model_path = tmp_path / "mhr_lod6.npz"
    np.savez(model_path, **mhr_data)

    with pytest.raises(ValueError, match=r"exclude_fields.*not_in_schema"):
        load_model_npz(model_path, MHR_SCHEMA, exclude_fields={"not_in_schema"})
