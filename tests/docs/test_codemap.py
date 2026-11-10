"""The package map of the architecture page equals the edge tables (principle 7.7).

``docs/concepts/architecture.md`` copies ``ALLOWED`` and ``BR_ALLOWED`` of ``tests/contract/test_layer_dependencies.py``
as one ``unit -> imports`` line per row, each table in a ``text`` fence whose first line names it; this test reads
the page and the tables and fails when they differ.
"""

from __future__ import annotations

import re
from pathlib import Path

from tests.contract.test_layer_dependencies import ALLOWED, BR_ALLOWED

PAGE = Path(__file__).parents[2] / "docs" / "concepts" / "architecture.md"
FENCE = re.compile(r"^```text\n# (?P<title>[^\n]+)\n(?P<body>.*?)^```$", re.MULTILINE | re.DOTALL)


def _maps() -> dict[str, dict[str, set[str]]]:
    """Title of each package-map fence to its rows, ``unit -> set of imports``."""
    maps = {}
    for fence in FENCE.finditer(PAGE.read_text()):
        rows = re.findall(r"^(\w+) +-> (.+)$", fence["body"], re.MULTILINE)
        maps[fence["title"]] = {unit: set() if imports == "nothing" else set(imports.split()) for unit, imports in rows}
    return maps


def test_the_package_map_is_the_edge_tables() -> None:
    """Every row of both tables is on the page with the same imports, and the page has no other row."""
    assert _maps() == {"among BetterHuman's units": ALLOWED, "into BetterRobot": BR_ALLOWED}
