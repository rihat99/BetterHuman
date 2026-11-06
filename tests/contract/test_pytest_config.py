"""pytest runs strictly, with exactly the markers the tiers select on.

Decision record: ``BetterRobot/docs/contracts/decisions/0033-test-tiers.md``. A mistyped marker or config key, a
passing xfail, an empty parameter set and a new warning each fail the run instead of passing silently. The markers
are the tier selectors: the fast and full tiers deselect ``licensed`` and ``cuda``, the fast tier also ``slow``.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

PYPROJECT = Path(__file__).parents[2] / "pyproject.toml"


@pytest.fixture(scope="module")
def ini() -> dict:
    return tomllib.loads(PYPROJECT.read_text())["tool"]["pytest"]["ini_options"]


def test_strict_options(ini: dict) -> None:
    assert {"--strict-markers", "--strict-config"} <= set(ini["addopts"])
    assert ini["xfail_strict"] is True
    assert ini["empty_parameter_set_mark"] == "fail_at_collect"
    assert ini["filterwarnings"][0] == "error"


def test_markers_are_the_tier_selectors(ini: dict) -> None:
    assert {marker.split(":")[0] for marker in ini["markers"]} == {"licensed", "cuda", "slow"}
