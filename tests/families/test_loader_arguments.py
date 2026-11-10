"""Every family loader checks its arguments before it reads a file (no weights needed).

A flag that is not a bool raises ArgumentError naming the flag (H3 review M3: ``smplx(hands="no")`` used to keep the
hands). A dtype other than float32 or float64 raises DtypeMismatchError in every loader, the one place that rule lives
(``bh.load`` passes ``dtype`` through), and an unknown gender raises UnknownNameError, FLAME's too (H3 review L6).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import torch
from better_robot.exceptions import ArgumentError, DtypeMismatchError, UnknownNameError

import better_human as bh

LOADERS = ["flame", "mano", "mhr", "smpl", "smplh", "smplx", "soma"]
GENDERS = {
    "smpl": ["female", "male", "neutral"],
    "smplh": ["female", "male", "neutral"],
    "smplx": ["female", "male", "neutral"],
    "flame": ["female", "generic", "male"],
}
FLAGS = [
    ("smplh", "hands"),
    ("smplx", "hands"),
    ("smplx", "face"),
    ("smplx", "flat_hand_mean"),
    ("mano", "flat_hand_mean"),
    ("mhr", "expression"),
    ("mhr", "correctives"),
    ("soma", "correctives"),
]


@pytest.mark.parametrize(("family", "flag"), FLAGS, ids=[f"{family}-{flag}" for family, flag in FLAGS])
@pytest.mark.parametrize("value", ["no", 0, None], ids=["str", "int", "none"])
def test_a_flag_that_is_not_a_bool_raises_naming_it(family: str, flag: str, value: object, tmp_path: Path) -> None:
    with pytest.raises(ArgumentError, match=rf"^{flag} must be bool \(True or False\), got {value!r}$"):
        getattr(bh.families, family)(tmp_path / "missing.npz", **{flag: value})


@pytest.mark.parametrize("family", LOADERS)
def test_a_dtype_that_is_not_float32_or_float64_raises_in_the_loader(family: str, tmp_path: Path) -> None:
    for dtype in (torch.float16, torch.int64):
        with pytest.raises(DtypeMismatchError, match=rf"dtype={dtype} is not torch.float32 or torch.float64"):
            getattr(bh.families, family)(tmp_path / "missing.npz", dtype=dtype)
        with pytest.raises(DtypeMismatchError, match=rf"dtype={dtype} is not torch.float32 or torch.float64"):
            bh.load(family, tmp_path / "missing.npz", dtype=dtype)


@pytest.mark.parametrize("family", list(GENDERS))
def test_an_unknown_gender_raises_listing_the_family_genders(family: str, tmp_path: Path) -> None:
    with pytest.raises(
        UnknownNameError, match=re.escape(f"unknown gender 'nuetral'; valid genders: {GENDERS[family]}")
    ):
        getattr(bh.families, family)(tmp_path / "missing.npz", gender="nuetral")
