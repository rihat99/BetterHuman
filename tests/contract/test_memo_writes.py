"""Principle 1.5: a memo is read and written only by the class that owns it.

``HumanState._memo`` (the outputs computed on first use) and ``Identity._frame_placements`` (the contact-frame
placements of each frame set) appear only as ``self.<memo>`` inside their own class, in
``src/better_human/core/human_model.py``. Every tracked Python file is parsed; any other reference to either name, as
an attribute or as the name argument of ``getattr``, ``setattr`` or ``delattr``, fails.
"""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OWNING_MODULE = "src/better_human/core/human_model.py"
MEMOS = {"_memo": "HumanState", "_frame_placements": "Identity"}
ACCESSORS = {"getattr", "setattr", "delattr"}


def _references(path: str, tree: ast.AST) -> list[str]:
    """``path::<class or <module>>: <memo>`` for each reference to a memo outside ``self.<memo>`` in its class."""
    found: list[str] = []

    def visit(node: ast.AST, owner: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                visit(child, child.name)
                continue
            if isinstance(child, ast.Attribute) and child.attr in MEMOS:
                by_self = isinstance(child.value, ast.Name) and child.value.id == "self"
                if not (path == OWNING_MODULE and owner == MEMOS[child.attr] and by_self):
                    found.append(f"{path}::{owner}: {child.attr}")
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Name) and child.func.id in ACCESSORS:
                names = [arg.value for arg in child.args if isinstance(arg, ast.Constant) and arg.value in MEMOS]
                found.extend(f"{path}::{owner}: {child.func.id}({name!r})" for name in names)
            visit(child, owner)

    visit(tree, "<module>")
    return found


def test_only_the_owning_class_touches_its_memo() -> None:
    files = subprocess.run(["git", "ls-files", "*.py"], cwd=REPO, capture_output=True, text=True, check=True).stdout
    found = [hit for path in files.split() for hit in _references(path, ast.parse((REPO / path).read_text()))]
    assert found == [], "a memo belongs to its class (HumanState._memo, Identity._frame_placements):\n" + "\n".join(
        found
    )
