"""Every family declares the density category of each part (decision record 0007): the keyword map and its coverage.

SOMA, MANO and FLAME have parts no category covers until their tables ship (H3): ``UNCOVERED`` pins them, and
``density="Dempster"`` raises for those families (``tests/mass/test_densities.py``).
"""

from __future__ import annotations

import pytest

import better_human as bh
from better_human.families import _tables as tables
from better_human.families.densities import density_categories
from better_human.io import weight_path
from better_human.io.mass_data import shipped_density_tables

#: Parts the keyword map leaves without a category, per family; H3 empties it with each family's table.
UNCOVERED = {
    "mano": list(tables.MANO.joint_names),
    "flame": ["global"],
    "soma": ["LeftArm", "RightArm", "LeftLeg", "RightLeg"],
}


def test_the_shipped_categories_are_smpl_joints() -> None:
    for table in shipped_density_tables().values():
        assert tuple(table) == tables.SMPL.joint_names


def test_smplx_names() -> None:
    names = ["pelvis", "spine1", "jaw", "left_eye_smplhf", "left_index1", "right_thumb3"]
    assert density_categories(names, shipped_density_tables()) == (
        "pelvis",  # exact match
        "spine1",  # exact match
        "head",  # jaw -> head
        "head",  # eye -> head (center category ignores the side token)
        "left_hand",  # finger -> hand
        "right_hand",
    )


def test_mhr_names() -> None:
    names = [
        "l_thumb2",
        "c_jaw",
        "l_upleg",
        "l_lowleg_twist2_proc",
        "l_ball",
        "r_uparm_twist1_proc",
        "c_spine0",
        "l_transversetarsal",
        "root",
    ]
    assert density_categories(names, shipped_density_tables()) == (
        "left_hand",  # thumb -> hand
        "head",  # jaw -> head
        "left_hip",  # upleg -> hip
        "left_knee",  # lowleg twist -> knee segment
        "left_foot",  # ball -> foot
        "right_shoulder",  # uparm twist -> shoulder segment
        "spine1",  # c_spine0 -> spine1 (digit + 1)
        "left_ankle",  # transversetarsal -> ankle
        "pelvis",  # root -> pelvis
    )


def test_a_name_the_map_misses_has_no_category() -> None:
    assert density_categories(["totally_unknown_gadget"], shipped_density_tables()) == (None,)


@pytest.mark.parametrize("family", ["smpl", "smplh", "smplx", "mano", "flame"])
def test_smpl_family_coverage_by_joint_names(family: str) -> None:
    """The SMPL-family loaders declare one category per joint from the table's joint names (no weights needed)."""
    names = getattr(tables, family.upper()).joint_names
    categories = density_categories(names, shipped_density_tables())
    uncovered = [name for name, category in zip(names, categories, strict=True) if category is None]
    assert uncovered == UNCOVERED.get(family, [])


@pytest.mark.licensed
@pytest.mark.parametrize("family", ["smpl", "smplh", "smplx", "mhr", "soma"])
def test_every_part_of_a_loaded_family_has_its_category(family: str) -> None:
    """``Parts.density_categories`` has one entry per part; ``None`` only for the parts ``UNCOVERED`` names."""
    paths = {
        "smpl": ("smpl", "SMPL_NEUTRAL.npz"),
        "smplh": ("smplh/smplh/neutral", "model.npz"),
        "smplx": ("smplx", "SMPLX_NEUTRAL.npz"),
        "mhr": ("MHR/converted", "mhr_lod1.npz"),
        "soma": ("SOMA", "soma_mid.npz"),
    }
    spec = getattr(bh.families, family)(weight_path(*paths[family]))
    parts = spec.parts
    assert len(parts.density_categories) == parts.count
    names = [spec.joint_names[joint] for joint in parts.joints]
    uncovered = [name for name, category in zip(names, parts.density_categories, strict=True) if category is None]
    assert uncovered == UNCOVERED.get(family, [])
