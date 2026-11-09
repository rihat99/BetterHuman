from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from better_robot.exceptions import ArgumentError, BetterRobotError

from better_human.io import AssetNotFoundError
from better_human.io.formats import FormatError
from better_human.io.load import load_model_npz, models_dir, weight_path
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


def test_explicit_path_takes_precedence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    explicit = tmp_path / "explicit.npz"
    explicit.touch()
    monkeypatch.setenv("BETTERHUMAN_MODELS_DIR", str(tmp_path / "unused"))

    assert weight_path("smpl", "SMPL_NEUTRAL.npz", path=explicit) == explicit


def test_weight_path_resolves_under_models_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = tmp_path / "smpl" / "SMPL_NEUTRAL.npz"
    expected.parent.mkdir()
    expected.touch()
    monkeypatch.setenv("BETTERHUMAN_MODELS_DIR", str(tmp_path))

    assert models_dir() == tmp_path
    assert weight_path("smpl", "SMPL_NEUTRAL.npz") == expected


def test_unset_models_dir_names_the_variable_and_the_fix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BETTERHUMAN_MODELS_DIR", raising=False)

    with pytest.raises(
        AssetNotFoundError, match=r"BETTERHUMAN_MODELS_DIR is unset\. Set it to the directory.* or pass path="
    ):
        weight_path("smpl", "SMPL_NEUTRAL.npz")


def test_models_dir_that_is_not_a_directory_is_an_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BETTERHUMAN_MODELS_DIR", str(tmp_path / "absent"))

    with pytest.raises(AssetNotFoundError, match=r"BETTERHUMAN_MODELS_DIR=.*absent is not a directory"):
        models_dir()


def test_missing_weight_file_names_the_file_and_both_remedies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BETTERHUMAN_MODELS_DIR", str(tmp_path))

    with pytest.raises(AssetNotFoundError, match=r"smpl/SMPL_NEUTRAL\.npz.*BETTERHUMAN_MODELS_DIR.*path="):
        weight_path("smpl", "SMPL_NEUTRAL.npz")


def test_missing_explicit_path_names_the_path_and_the_variable(tmp_path: Path) -> None:
    with pytest.raises(
        AssetNotFoundError, match=r"absent\.npz\. Pass path=.*smpl/SMPL_NEUTRAL\.npz under \$BETTERHUMAN"
    ):
        weight_path("smpl", "SMPL_NEUTRAL.npz", path=tmp_path / "absent.npz")


def test_asset_not_found_sits_under_better_robot_error() -> None:
    # One `except BetterRobotError` catches the stack (BetterRobot record 0031); `except FileNotFoundError` still works.
    assert issubclass(AssetNotFoundError, BetterRobotError)
    assert issubclass(AssetNotFoundError, FileNotFoundError)


def test_dense_regressor_loads_without_dtype_conversion(tmp_path: Path) -> None:
    model_path = tmp_path / "dense.npz"
    _write_dense_fixture(model_path)

    loaded = load_model_npz(model_path, SMPL_SCHEMA)

    regressor = loaded["J_regressor"]
    assert regressor.layout == torch.strided
    assert regressor.dtype == torch.float64
    assert regressor.shape == (3, 4)
    assert loaded["f"].dtype == torch.uint32


def test_an_unknown_excluded_field_is_an_argument_error(tmp_path: Path) -> None:
    model_path = tmp_path / "model.npz"
    _write_dense_fixture(model_path)
    with pytest.raises(ArgumentError, match=r"exclude_fields names \['nope'\].*pass names of its fields only"):
        load_model_npz(model_path, SMPL_SCHEMA, exclude_fields=("nope",))


def test_object_valued_required_field_is_an_actionable_load_error(tmp_path: Path) -> None:
    model_path = tmp_path / "object_template.npz"
    _write_dense_fixture(model_path, object_template=True)

    with pytest.raises(
        ValueError,
        match=r"object_template\.npz.*'v_template'.*non-pickled numeric array.*Reconvert",
    ):
        load_model_npz(model_path, SMPL_SCHEMA)


@pytest.mark.parametrize("corruption", ["garbage", "empty", "first_half", "tail_cut", "single_npy"])
def test_an_unreadable_archive_is_a_format_error_naming_the_path(tmp_path: Path, corruption: str) -> None:
    valid = tmp_path / "valid.npz"
    _write_dense_fixture(valid)
    raw = valid.read_bytes()
    model_path = tmp_path / f"{corruption}.npz"
    if corruption == "garbage":
        model_path.write_bytes(b"this is not a zip archive" * 10)
    elif corruption == "empty":
        model_path.write_bytes(b"")
    elif corruption == "first_half":
        model_path.write_bytes(raw[: len(raw) // 2])
    elif corruption == "tail_cut":
        model_path.write_bytes(raw[:-100])
    else:
        with model_path.open("wb") as file:
            np.save(file, np.zeros(3))

    with pytest.raises(FormatError, match=r"not a readable \.npz archive") as raised:
        load_model_npz(model_path, SMPL_SCHEMA)
    assert str(model_path) in str(raised.value)
    assert "allow_pickle" not in str(raised.value)
    assert raised.value.__cause__ is None and raised.value.__context__ is None


def test_mhr_resolution_uses_nested_canonical_layout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = tmp_path / "MHR" / "converted" / "mhr_lod6.npz"
    expected.parent.mkdir(parents=True)
    expected.touch()
    monkeypatch.setenv("BETTERHUMAN_MODELS_DIR", str(tmp_path))

    assert weight_path("MHR", "converted/mhr_lod6.npz") == expected


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
    mhr_data["corrective_output_weights"] = np.empty((3000, 12), dtype=object)
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


def test_mhr_loader_allows_excluded_corrective_members_to_be_absent(
    tmp_path: Path,
    mhr_data: dict[str, np.ndarray],
) -> None:
    model_path = tmp_path / "mhr_lod6.npz"
    for name in MHR_CORRECTIVE_FIELDS:
        del mhr_data[name]
    np.savez(model_path, **mhr_data)

    loaded = load_model_npz(model_path, MHR_SCHEMA, exclude_fields=MHR_CORRECTIVE_FIELDS)

    assert not (set(loaded) & MHR_CORRECTIVE_FIELDS)
    assert loaded["template_vertices"].shape == (4, 3)


def test_mhr_loader_binds_dimensions_from_selected_fields(
    tmp_path: Path,
    mhr_data: dict[str, np.ndarray],
) -> None:
    model_path = tmp_path / "mhr_lod6.npz"
    np.savez(model_path, **mhr_data)
    excluded = {
        "template_vertices",
        "identity_basis",
        "expression_basis",
    }

    loaded = load_model_npz(model_path, MHR_SCHEMA, exclude_fields=excluded)

    assert loaded["corrective_output_weights"].shape == (3000, 12)


def test_mhr_v1_archive_loads_through_its_migration(tmp_path: Path, mhr_data: dict[str, np.ndarray]) -> None:
    # Record 0012: a v1 archive on disk stays valid; v2's format and provenance are added in memory.
    model_path = tmp_path / "mhr_lod6.npz"
    del mhr_data["format"], mhr_data["provenance"]
    mhr_data["format_version"] = np.asarray(1, dtype=np.int64)
    np.savez(model_path, **mhr_data)

    loaded = load_model_npz(model_path, MHR_SCHEMA)

    assert (str(loaded["format"]), int(loaded["format_version"])) == ("better_human.mhr", 2)
    assert str(loaded["provenance"]) == "unknown (v1)"


@pytest.mark.parametrize("version", [1, 2])
def test_an_excluded_header_is_still_read(tmp_path: Path, mhr_data: dict[str, np.ndarray], version: int) -> None:
    # The version selects the migrations, so excluding the header leaves it out of the result only.
    model_path = tmp_path / "mhr_lod6.npz"
    if version == 1:
        del mhr_data["format"], mhr_data["provenance"]
        mhr_data["format_version"] = np.asarray(1, dtype=np.int64)
    np.savez(model_path, **mhr_data)

    loaded = load_model_npz(model_path, MHR_SCHEMA, exclude_fields=("format", "format_version"))

    assert set(loaded) == set(MHR_SCHEMA.fields) - {"format", "format_version"}


def test_model_loader_rejects_unknown_field_exclusions(
    tmp_path: Path,
    mhr_data: dict[str, np.ndarray],
) -> None:
    model_path = tmp_path / "mhr_lod6.npz"
    np.savez(model_path, **mhr_data)

    with pytest.raises(ValueError, match=r"exclude_fields.*not_in_schema"):
        load_model_npz(model_path, MHR_SCHEMA, exclude_fields={"not_in_schema"})
