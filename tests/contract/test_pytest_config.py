"""pytest runs strictly, with exactly the markers the tiers select on.

A mistyped marker or config key, a passing xfail, an empty parameter set and a new warning each fail the run
instead of passing silently. Every warning filter after ``error`` names its message and a category narrower than
``Warning``, and sits under a comment giving the reason. The markers are the tier selectors: the fast and full tiers
deselect ``licensed`` and ``cuda``, the fast tier also ``slow``.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

PYPROJECT = Path(__file__).parents[2] / "pyproject.toml"


@pytest.fixture(scope="module")
def ini() -> dict:
    return tomllib.loads(PYPROJECT.read_text())["tool"]["pytest"]["ini_options"]


def test_strict_options(ini: dict) -> None:
    fix = "restore it in pyproject.toml's [tool.pytest.ini_options] (strict pytest)"
    assert {"--strict-markers", "--strict-config"} <= set(ini["addopts"]), f"addopts lacks a --strict-* flag; {fix}"
    assert ini["xfail_strict"] is True, f"xfail_strict is not true; {fix}"
    assert ini["empty_parameter_set_mark"] == "fail_at_collect", f"empty parameter sets do not fail; {fix}"
    assert ini["filterwarnings"][0] == "error", f'filterwarnings does not start with "error"; {fix}'


def test_markers_are_the_tier_selectors(ini: dict) -> None:
    markers = {marker.split(":")[0] for marker in ini["markers"]}
    assert markers == {"licensed", "cuda", "slow"}, (
        f"markers are {sorted(markers)}; the tier commands select on exactly licensed, cuda and slow, so a new "
        "marker comes with the tier commands that select on it"
    )


def test_each_later_warning_filter_names_its_warning_and_reason() -> None:
    """A filter after ``error`` has an action, a message and a category other than ``Warning``, under a comment."""
    block = re.search(r"^filterwarnings = \[\n(.*?)^\]", PYPROJECT.read_text(), re.MULTILINE | re.DOTALL)
    lines = [line.strip() for line in block[1].splitlines() if line.strip()]
    problems = []
    filters = [index for index, line in enumerate(lines) if not line.startswith("#")]
    for index in filters[1:]:
        line = lines[index]
        fields = line.rstrip(",").strip('"').split(":")
        if len(fields) < 3 or not fields[1] or fields[2] in ("", "Warning"):
            problems.append(f"{line}: name the message and a category narrower than Warning")
        previous = index - 1
        while previous > filters[0] and not lines[previous].startswith("#"):
            previous -= 1
        if not lines[previous].startswith("#"):
            problems.append(f"{line}: put a comment with the reason above it")
    assert problems == [], "pyproject.toml filterwarnings:\n  " + "\n  ".join(problems)
