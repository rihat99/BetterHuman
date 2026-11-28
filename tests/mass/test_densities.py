"""One ``density`` parameter: a table name by default; unknown names and mismatches raise at load.

The load-level cases use a synthetic FLAME-shaped file (no weights) and replace its declared categories where a case
needs a part without one.
"""

from __future__ import annotations

import inspect
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from better_robot.exceptions import ArgumentError, DtypeMismatchError, UnknownNameError

import better_human as bh
from better_human.mass import part_densities
from tests.support.synthetic import flame_spec

TABLES = {"Light": {"head": 900.0, "neck": 950.0}, "Heavy": {"head": 1100.0, "neck": 1150.0}}


@pytest.fixture
def flame(tmp_path: Path) -> bh.FamilySpec:
    return flame_spec(tmp_path)


def test_the_default_density_is_the_dempster_table() -> None:
    for function in (bh.load, bh.HumanModel.from_spec):
        assert inspect.signature(function).parameters["density"].default == "Dempster"


def test_a_part_without_category_raises_at_load_naming_it(flame: bh.FamilySpec) -> None:
    categories = (None, "neck", "head", "head", "head")
    spec = replace(flame, parts=replace(flame.parts, density_categories=categories))
    with pytest.raises(ArgumentError, match=r"1 of 5 parts have no density category: \['global'\]\. Pass density=None"):
        bh.HumanModel.from_spec(spec)


def test_an_unknown_table_raises_at_load(flame: bh.FamilySpec) -> None:
    with pytest.raises(UnknownNameError, match=r"unknown density table 'Nope' \(valid names: \['Chandler'"):
        bh.HumanModel.from_spec(flame, density="Nope")


def test_an_unknown_category_raises_at_load(flame: bh.FamilySpec) -> None:
    categories = ("pelvis", "neck", "head", "head", "eyebrow")
    spec = replace(flame, parts=replace(flame.parts, density_categories=categories))
    with pytest.raises(UnknownNameError, match=r"unknown Dempster category name 'eyebrow' \(24 valid names\)"):
        bh.HumanModel.from_spec(spec)


def test_a_dtype_mismatch_raises_at_load(flame: bh.FamilySpec) -> None:
    with pytest.raises(
        DtypeMismatchError,
        match=r"density.dtype=torch.float32 != spec.dtype=torch.float64; pass density.to\(torch.float64\)",
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
