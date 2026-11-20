"""Every family declares the density category of each part, and every part has one.

The SMPL family declares one category per joint in ``src/better_human/families/_tables.py``; MHR and SOMA map their
part joints in their loader modules, whose loaders raise FormatError for a part the map lacks. Every
declared category is a category of every shipped table (full tier); every part of every MHR LOD and of SOMA is mapped
(release tier: the part joints come from the archives). Every shipped density names its source.
"""

from __future__ import annotations

from collections.abc import Callable
from importlib import resources
from pathlib import Path

import numpy as np
import pytest

import better_human as bh
from better_human.families import _tables, mhr_family, soma_family
from better_human.io import FormatError, weight_path
from better_human.io.mass_data import shipped_density_tables
from tests.support.synthetic import flame_spec

TABLES = (_tables.SMPL, _tables.SMPLH, _tables.SMPLX, _tables.MANO, _tables.FLAME)
#: ``family -> every category it declares``; MANO's ``{side}`` filled with both sides.
DECLARED = {
    **{
        table.name: {category.format(side=side) for side in ("left", "right") for category in table.density_categories}
        for table in TABLES
    },
    "mhr": set(mhr_family._DENSITY_CATEGORIES.values()),
    "soma": set(soma_family._DENSITY_CATEGORIES.values()),
}


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
def test_every_smpl_family_joint_declares_a_category(table: _tables.FamilyTable) -> None:
    assert len(table.density_categories) == len(table.joint_names)


@pytest.mark.parametrize("family", list(DECLARED))
def test_every_declared_category_is_in_every_shipped_table(family: str) -> None:
    for name, table in shipped_density_tables().items():
        assert DECLARED[family] <= set(table), f"{family}: {sorted(DECLARED[family] - set(table))} not in {name}"


def test_the_flame_loader_declares_its_table_categories(tmp_path: Path) -> None:
    assert flame_spec(tmp_path).parts.density_categories == ("head", "neck", "head", "head", "head")


LOADED: dict[str, Callable[[], bh.FamilySpec]] = {
    "smpl": lambda: bh.families.smpl(weight_path("smpl", "SMPL_NEUTRAL.npz")),
    "smplh": lambda: bh.families.smplh(weight_path("smplh/smplh/neutral", "model.npz")),
    "smplx": lambda: bh.families.smplx(weight_path("smplx", "SMPLX_NEUTRAL.npz")),
    **{
        f"mhr-lod{lod}": lambda lod=lod: bh.families.mhr(
            weight_path("MHR/converted", f"mhr_lod{lod}.npz"), lod=lod, expression=False, correctives=False
        )
        for lod in mhr_family.LODS
    },
    "soma": lambda: bh.families.soma(weight_path("SOMA", "soma_mid.npz"), correctives=False),
}


@pytest.mark.licensed
@pytest.mark.parametrize("family", list(LOADED))
def test_every_part_of_a_loaded_family_has_its_category(family: str) -> None:
    parts = LOADED[family]().parts
    assert len(parts.density_categories) == parts.count
    assert None not in parts.density_categories


@pytest.mark.licensed
@pytest.mark.parametrize(("family", "joint"), [("mhr-lod6", "c_spine0"), ("soma", "LeftHand")])
def test_a_part_the_map_lacks_raises_at_load(family: str, joint: str, monkeypatch: pytest.MonkeyPatch) -> None:
    module = mhr_family if family.startswith("mhr") else soma_family
    monkeypatch.delitem(module._DENSITY_CATEGORIES, joint)
    with pytest.raises(FormatError, match=rf"the part of joint '{joint}' has no (MHR|SOMA) density category"):
        LOADED[family]()
