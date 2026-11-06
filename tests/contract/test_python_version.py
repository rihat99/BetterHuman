"""One Python version: the floor, the lint target, the pinned development version and the running interpreter agree.

Decision record: ``BetterRobot/docs/contracts/decisions/0034-python-312.md``. ``tomllib`` (3.11 and later) in this
module is itself the check that the declared floor can run the tests.
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

REPO = Path(__file__).parents[2]


def test_floor_lint_target_and_pin_agree() -> None:
    pyproject = tomllib.loads((REPO / "pyproject.toml").read_text())
    floor = pyproject["project"]["requires-python"]
    target = pyproject["tool"]["ruff"]["target-version"]
    pin = (REPO / ".python-version").read_text().strip()
    running = f"{sys.version_info.major}.{sys.version_info.minor}"
    assert pin == "3.12"
    assert floor == f">={pin}"
    assert target == "py312"
    assert running == pin
