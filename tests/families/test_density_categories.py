"""Every family declares the density category of each part (decision record 0007), and every part has one.

The SMPL family declares one category per joint in ``src/better_human/families/_tables.py``; MHR and SOMA map their
part joints in their loader modules. Every declared category is a shipped table category: SMPL's 24 part names. Every
shipped density names its source.
"""

from __future__ import annotations

from collections.abc import Mapping
from importlib import resources
from pathlib import Path

import numpy as np
import pytest

import better_human as bh
from better_human.families import _tables
from better_human.families.mhr_family import _DENSITY_CATEGORIES as MHR_CATEGORIES
from better_human.families.soma_family import _DENSITY_CATEGORIES as SOMA_CATEGORIES
from better_human.io import weight_path
from better_human.io.mass_data import shipped_density_tables
from tests.support.synthetic import flame_spec

TABLES = (_tables.SMPL, _tables.SMPLH, _tables.SMPLX, _tables.MANO, _tables.FLAME)


def test_the_shipped_categories_are_smpl_joints() -> None:
    for table in shipped_density_tables().values():
        assert tuple(table) == _tables.SMPL_JOINT_NAMES


def test_every_shipped_density_names_its_source() -> None:
    """A number read in a publication names it; any other number says it is unverified or assumed (owner ruling)."""
    with np.load(resources.files("better_human.io") / "assets" / "density_tables.npz") as archive:
        sources, densities = archive["sources"], archive["densities"]
    assert sources.shape == densities.shape
    forms = ("Herman 2007, The Physics of the Human Body, Table 1.7", "unverified (cited from memory): ", "assumed: ")
    for source in map(str, sources.flat):
        assert any(source.startswith(form) and source != form for form in forms), source


@pytest.mark.parametrize("table", TABLES, ids=lambda table: table.name)
def test_every_smpl_family_joint_has_a_shipped_category(table: _tables.FamilyTable) -> None:
    assert len(table.density_categories) == len(table.joint_names)
    categories = {category.format(side=side) for side in ("left", "right") for category in table.density_categories}
    assert categories <= set(_tables.SMPL_JOINT_NAMES)


@pytest.mark.parametrize("categories", [MHR_CATEGORIES, SOMA_CATEGORIES], ids=["mhr", "soma"])
def test_every_mapped_mhr_and_soma_joint_has_a_shipped_category(categories: Mapping[str, str]) -> None:
    assert set(categories.values()) <= set(_tables.SMPL_JOINT_NAMES)


def test_the_flame_loader_declares_its_table_categories(tmp_path: Path) -> None:
    assert flame_spec(tmp_path).parts.density_categories == ("head", "neck", "head", "head", "head")


@pytest.mark.licensed
@pytest.mark.parametrize("family", ["smpl", "smplh", "smplx", "mhr", "soma"])
def test_every_part_of_a_loaded_family_has_its_category(family: str) -> None:
    paths = {
        "smpl": ("smpl", "SMPL_NEUTRAL.npz"),
        "smplh": ("smplh/smplh/neutral", "model.npz"),
        "smplx": ("smplx", "SMPLX_NEUTRAL.npz"),
        "mhr": ("MHR/converted", "mhr_lod1.npz"),
        "soma": ("SOMA", "soma_mid.npz"),
    }
    parts = getattr(bh.families, family)(weight_path(*paths[family])).parts
    assert len(parts.density_categories) == parts.count
    assert None not in parts.density_categories
