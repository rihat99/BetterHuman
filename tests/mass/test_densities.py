"""One ``density`` parameter (decision record 0007): a table name by default; unknown names and mismatches raise at load.

The load-level cases use a synthetic FLAME-shaped file (no weights), whose root part ``global`` has no density
category until FLAME's table ships (H3).
"""

from __future__ import annotations

import inspect
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from better_robot.exceptions import ArgumentError, DtypeMismatchError, UnknownNameError

import better_human as bh
from better_human.mass import part_densities

NUM_VERTICES = 50
TABLES = {"Light": {"head": 900.0, "neck": 950.0}, "Heavy": {"head": 1100.0, "neck": 1150.0}}


@pytest.fixture
def flame(tmp_path: Path) -> bh.FamilySpec:
    """A float64 FLAME spec from a random FLAME-shaped file: joints global, neck, jaw, left_eye, right_eye."""
    rng = np.random.default_rng(0)
    weights, regressor = rng.random((NUM_VERTICES, 5)), rng.random((5, NUM_VERTICES))
    path = tmp_path / "generic_model.npz"
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
    return bh.families.flame(path, dtype=torch.float64)


def test_the_default_density_is_the_dempster_table() -> None:
    for function in (bh.load, bh.HumanModel.from_spec):
        assert inspect.signature(function).parameters["density"].default == "Dempster"


def test_a_part_without_category_raises_at_load_naming_it(flame: bh.FamilySpec) -> None:
    with pytest.raises(ArgumentError, match=r"1 of 5 parts have no density category: \['global'\]\. Pass density=None"):
        bh.HumanModel.from_spec(flame)


def test_an_unknown_table_raises_at_load(flame: bh.FamilySpec) -> None:
    with pytest.raises(UnknownNameError, match=r"unknown density table 'Nope'; valid density tables: \['Chandler'"):
        bh.HumanModel.from_spec(flame, density="Nope")


def test_an_unknown_category_raises_at_load(flame: bh.FamilySpec) -> None:
    categories = ("pelvis", "neck", "head", "head", "eyebrow")
    spec = replace(flame, parts=replace(flame.parts, density_categories=categories))
    with pytest.raises(UnknownNameError, match=r"unknown Dempster category name 'eyebrow'; 24 Dempster category names"):
        bh.HumanModel.from_spec(spec)


def test_a_dtype_mismatch_raises_at_load(flame: bh.FamilySpec) -> None:
    with pytest.raises(
        DtypeMismatchError, match=r"density is torch.float32; pass it in the spec's dtype, torch.float64"
    ):
        bh.HumanModel.from_spec(flame, density=torch.full((5,), 1000.0, dtype=torch.float32))


def test_each_part_takes_its_category_in_the_named_table() -> None:
    densities = part_densities(
        TABLES,
        "Heavy",
        categories=("neck", "head", "head"),
        part_names=("a", "b", "c"),
        dtype=torch.float64,
        device=torch.device("cpu"),
    )
    torch.testing.assert_close(densities, torch.tensor([1150.0, 1100.0, 1100.0], dtype=torch.float64), rtol=0, atol=0)
