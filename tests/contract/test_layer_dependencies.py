"""Imports follow decision record 0011's two edge tables: among BetterHuman's units, and into BetterRobot's.

A unit is a first-level subpackage or module of ``better_human``; the root ``__init__`` is the unit
``better_human``. ``better_robot.testing.imported_packages`` (BetterRobot decision record 0035) walks every import,
wherever it sits (module level, inside a function, under ``TYPE_CHECKING``), resolves ``from pkg import sub`` to the
unit ``sub``, and reports a BetterRobot target as its first-level unit (``better_robot.lie``), or ``better_robot``
for a name of BetterRobot's root. So the record's ``better_robot.viewer.tools`` row entry is part of
``better_robot.viewer``. Imports of other packages are not this test's subject. An edge outside the tables that
exists today sits in ``ratchets/edges.json`` (``"from -> to"`` -> the ``path::symbol`` that makes it and why it
goes); the ratchet only shrinks, and the fix that removes an edge removes its entry.
"""

from __future__ import annotations

import json
from pathlib import Path

from better_robot.testing import imported_packages

ROOT = "better_human"
SRC = Path(__file__).resolve().parents[2] / "src"
EDGE_RATCHET: dict[str, str] = json.loads((Path(__file__).parent / "ratchets" / "edges.json").read_text())

ALLOWED: dict[str, set[str]] = {
    "io": set(),
    "mass": set(),
    "core": {"mass"},
    "families": {"core", "io", "mass"},
    "terms": {"core"},
    "viz": {"core", "io"},  # the studio persists frame sets (io/frame_sets.py) while it still lives in viz
    "studio": {ROOT, "core", "families", "io", "mass", "viz"},
    ROOT: {"core", "families", "io", "mass", "terms", "viz"},
}
BR_ALLOWED: dict[str, set[str]] = {
    "io": {"better_robot.exceptions"},  # BetterRobot record 0031: io's errors sit under BetterRobotError
    "mass": {"better_robot.lie", "better_robot.spatial"},
    "core": {"better_robot", "better_robot.kinematics", "better_robot.lie"},
    "families": {"better_robot", "better_robot.io", "better_robot.kinematics", "better_robot.lie"},
    "terms": {"better_robot", "better_robot.dynamics", "better_robot.kinematics", "better_robot.optim"},
    "viz": {"better_robot.viewer"},
    "studio": {"better_robot.lie", "better_robot.viewer"},
    ROOT: set(),
}


def _forbidden_edges(src: Path) -> set[str]:
    """``"unit -> target"`` for every import of ``src / "better_human"`` outside the two tables."""
    edges = imported_packages(src, ROOT)
    new_units = sorted(edges.keys() - ALLOWED.keys())
    assert not new_units, f"new units {new_units}: a decision record adds their rows to ALLOWED and BR_ALLOWED"
    return {
        f"{unit} -> {target}"
        for unit, targets in edges.items()
        for target in targets
        if (target in edges and target not in ALLOWED[unit])
        or (target.split(".")[0] == "better_robot" and target not in BR_ALLOWED[unit])
    }


def test_imports_follow_the_edge_tables() -> None:
    """Every import between units, and into BetterRobot, is in a table or, until its fix lands, in the ratchet."""
    new = sorted(_forbidden_edges(SRC) - EDGE_RATCHET.keys())
    assert not new, (
        "imports outside decision record 0011's edge tables (tests/contract/test_layer_dependencies.py): move the "
        "code to a unit the importer may import, or change the tables with a decision record:\n  " + "\n  ".join(new)
    )


def test_edge_ratchet_has_no_stale_entries() -> None:
    """An edge that no longer exists leaves ``ratchets/edges.json`` in the commit that removes it."""
    stale = sorted(EDGE_RATCHET.keys() - _forbidden_edges(SRC))
    assert not stale, "remove these edges from tests/contract/ratchets/edges.json:\n  " + "\n  ".join(stale)


def test_a_planted_import_is_a_forbidden_edge(tmp_path: Path) -> None:
    """The tables are checked on the real walker: an import between units and one into BetterRobot each show up."""
    sources = {
        "__init__": "from . import core\n",
        "core/__init__": "from better_human.viz import layers\n",
        "viz/__init__": "",
        "viz/layers": "import better_robot.viewer\n",
        "io": "from better_robot.lie import so3\n",
    }
    for module, source in sources.items():
        path = tmp_path / ROOT / f"{module}.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
    assert _forbidden_edges(tmp_path) == {"core -> viz", "io -> better_robot.lie"}
