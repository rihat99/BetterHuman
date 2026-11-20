"""One licence: the ``LICENSE`` text, the ``pyproject.toml`` licence and the README line all say Apache-2.0.

BetterHuman is licensed Apache-2.0, the same licence as BetterRobot. The README line also keeps
the rule that body-model weights and data derived from them have their own terms, which no code licence changes. No
other code licence is named in the README or ``pyproject.toml``.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

REPO = Path(__file__).parents[2]
README_LINE = (
    "Code: Apache-2.0 ([LICENSE](LICENSE)). Body-model weights and data derived from them keep their own licences."
)
#: The common code licences other than Apache-2.0, by SPDX identifier or name, and a "released under" claim.
OTHER_LICENCES = re.compile(
    r"\b(MIT|BSD|L?GPL|AGPL|MPL|ISC|EPL|CDDL|Unlicense|WTFPL|CC0|CC[- ]BY|Apache[- ]1(\.\d)?|"
    r"(released|licensed|available) under (?!(the )?Apache[- ]2))",
    re.IGNORECASE,
)


def test_license_text_pyproject_and_readme_agree() -> None:
    license_head = [line.strip() for line in (REPO / "LICENSE").read_text().splitlines()[:2]]
    project = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]
    assert license_head == ["Apache License", "Version 2.0, January 2004"]
    assert project["license"] == "Apache-2.0"
    assert project["license-files"] == ["LICENSE"]
    assert README_LINE in (REPO / "README.md").read_text().splitlines()


def test_no_other_licence_is_claimed() -> None:
    found = [
        f"{name}:{number}: {line.strip()}"
        for name in ("README.md", "pyproject.toml")
        for number, line in enumerate((REPO / name).read_text().splitlines(), 1)
        if OTHER_LICENCES.search(line)
    ]
    assert not found, "a second licence claim (Apache-2.0 only):\n" + "\n".join(found)


def test_a_planted_licence_claim_is_found() -> None:
    claims = ["This code is released under the MIT licence.", "License :: OSI Approved :: BSD License", "GPL-3.0"]
    assert all(OTHER_LICENCES.search(claim) for claim in claims)
    assert not OTHER_LICENCES.search(README_LINE)
    assert not OTHER_LICENCES.search("BetterHuman is released under the Apache-2.0 licence.")
