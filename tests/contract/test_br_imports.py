"""BetterHuman imports BetterRobot through its public surface only (principle 1.6; decision records 0011 and 0021).

Every ``better_robot`` path that a module of ``src`` or ``examples`` imports (``import a.b`` or ``from a import b``,
wherever the import sits) is a path of BetterRobot's ``__all__`` tree (``better_robot.testing.public_objects``), and
a first-level name of BetterRobot's root is imported from the root: ``from better_robot import Viewer``, never
``from better_robot.viewer import Viewer``, so each name has one import path.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path

from better_robot.testing import public_objects

REPO = Path(__file__).resolve().parents[2]
PUBLIC = {
    **public_objects("better_robot"),
    # BetterRobot documents its viewer tools as public, but better_robot.viewer's __all__ does not list them yet.
    **public_objects("better_robot.viewer.tools"),
}


def _imported(tree: ast.AST) -> Iterator[tuple[int, str]]:
    """``(line, dotted path)`` of every absolute import in ``tree``."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from ((node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            yield from ((node.lineno, f"{node.module}.{alias.name}") for alias in node.names)


def _off_the_surface(path: str) -> str | None:
    """Why ``path`` is not the public import path of what it names, or None when it is."""
    if path not in PUBLIC:
        return "not in BetterRobot's __all__ tree"
    root = "better_robot." + path.rsplit(".", 1)[-1]
    if path != root and PUBLIC.get(root) is PUBLIC[path]:
        return f"a first-level name: import it from better_robot ({root})"
    return None


def test_better_robot_is_imported_through_its_public_paths() -> None:
    found = [
        f"{path.relative_to(REPO)}:{line}: {imported}: {reason}"
        for folder in ("src", "examples")
        for path in sorted((REPO / folder).rglob("*.py"))
        for line, imported in _imported(ast.parse(path.read_text()))
        if imported.split(".")[0] == "better_robot" and (reason := _off_the_surface(imported))
    ]
    assert not found, "BetterRobot imports off its public surface (principle 1.6):\n" + "\n".join(found)
