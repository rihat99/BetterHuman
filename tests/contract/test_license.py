"""One licence: the ``LICENSE`` text, the ``pyproject.toml`` licence and the README line all say Apache-2.0.

Decision record 0018 (BetterHuman is licensed Apache-2.0, the same licence as BetterRobot). The README line also keeps
the rule that body-model weights and data derived from them have their own terms, which no code licence changes.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

REPO = Path(__file__).parents[2]
README_LINE = (
    "Code: Apache-2.0 ([LICENSE](LICENSE)). Body-model weights and data derived from them keep their own licences."
)


def test_license_text_pyproject_and_readme_agree() -> None:
    license_head = [line.strip() for line in (REPO / "LICENSE").read_text().splitlines()[:2]]
    project = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]
    assert license_head == ["Apache License", "Version 2.0, January 2004"]
    assert project["license"] == "Apache-2.0"
    assert project["license-files"] == ["LICENSE"]
    assert README_LINE in (REPO / "README.md").read_text().splitlines()
