"""One Python version: the floor, the lint target, the pinned development version and the running interpreter agree.

``tomllib`` (3.11 and later) in this
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
    rule = "one Python version, 3.12"
    assert pin == "3.12", f".python-version is {pin!r}; write 3.12 there ({rule})"
    assert floor == f">={pin}", f"requires-python is {floor!r}; set it to '>={pin}' in pyproject.toml ({rule})"
    assert target == "py312", f"ruff target-version is {target!r}; set it to 'py312' in pyproject.toml ({rule})"
    assert running == pin, (
        f"the tests run on Python {running}; run them through uv, which reads .python-version ({rule})"
    )
