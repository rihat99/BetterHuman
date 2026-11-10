"""Every family loader checks its arguments before it reads a file (no weights needed).

A flag that is not a bool raises ArgumentError naming the flag (H3 review M3: ``smplx(hands="no")`` used to keep the
hands).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from better_robot.exceptions import ArgumentError

import better_human as bh

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
