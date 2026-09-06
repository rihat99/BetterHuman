"""Name-keyword density mapping for SMPL-X and MHR parts."""

from __future__ import annotations

from pathlib import Path

import pytest

from better_human.bodies.smpl_family.config_smpl import SMPL_CONFIG
from better_human.bodies.assets import densities

MODELS_DIR = Path(__file__).parents[2] / "models"
SMPLX_PATH = MODELS_DIR / "smplx" / "SMPLX_NEUTRAL.npz"
MHR_PATH = MODELS_DIR / "MHR" / "converted" / "mhr_lod1.npz"


def test_categories_match_smpl_config() -> None:
    # The 24 density categories are exactly SMPL's native joints, in order.
    assert densities.SMPL_CATEGORY_NAMES == tuple(SMPL_CONFIG.joint_names)


def test_part_categories_smplx_names() -> None:
    names = ["pelvis", "spine1", "jaw", "left_eye_smplhf", "left_index1", "right_thumb3"]
    assert densities.part_categories(names) == [
        "pelvis",  # exact match
        "spine1",  # exact match
        "head",  # jaw -> head
        "head",  # eye -> head (center category ignores the side token)
        "left_hand",  # finger -> hand
        "right_hand",
    ]


def test_part_categories_mhr_names() -> None:
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
    assert densities.part_categories(names) == [
        "left_hand",  # thumb -> hand
        "head",  # jaw -> head
        "left_hip",  # upleg -> hip
        "left_knee",  # lowleg twist -> knee segment
        "left_foot",  # ball -> foot
        "right_shoulder",  # uparm twist -> shoulder segment
        "spine1",  # c_spine0 -> spine1 (digit + 1)
        "left_ankle",  # transversetarsal -> ankle
        "pelvis",  # root -> pelvis
    ]


def test_unmapped_part_falls_back_to_mean() -> None:
    table = {str(index): 1000.0 for index in range(24)}
    table["0"] = 2000.0  # pelvis
    tensor = densities.density_for_parts(["pelvis", "totally_unknown_gadget"], table)
    assert tensor.shape == (2,)
    assert float(tensor[0]) == pytest.approx(2000.0)
    # Unmapped -> table mean over all 24 category values.
    expected_mean = (2000.0 + 23 * 1000.0) / 24
    assert float(tensor[1]) == pytest.approx(expected_mean)
    assert densities.part_categories(["totally_unknown_gadget"]) == [None]


@pytest.mark.parametrize("path", [SMPLX_PATH, MHR_PATH], ids=["smplx", "mhr"])
def test_full_coverage_on_real_joint_names(path: Path) -> None:
    if not path.is_file():
        pytest.skip(f"licensed fixture is absent: {path}")
    import better_human as bh  # noqa: PLC0415 - only needed with weights present

    if "smplx" in path.name.lower():
        body = bh.SMPLX(str(path), compute_mass=False)
    else:
        body = bh.MHR(model_path=str(path), lod=1, compute_mass=False)
    names = list(body.structure.joint_names)
    categories = densities.part_categories(names)
    unmapped = [name for name, category in zip(names, categories, strict=True) if category is None]
    assert unmapped == []
    tensor = densities.density_for_parts(names, {str(index): 1000.0 for index in range(24)})
    assert tensor.shape == (len(names),)
