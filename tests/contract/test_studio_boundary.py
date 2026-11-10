"""Decision record 0009: the studio is an application on the public BetterHuman and BetterRobot surface.

No module of ``better_human.studio`` reads a ``_``-prefixed attribute of another object (an attribute of ``self`` or
``cls`` is the object's own: a ``Layer`` subclass calls its base's ``_track``), imports a ``_``-prefixed name or
module, or reaches one through ``getattr``. No module under ``src/better_human`` writes ``os.environ``: the weights
directory is the user's ``$BETTERHUMAN_MODELS_DIR``, read through ``better_human.io.models_dir()`` (record 0017).
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src" / "better_human"
OWN = {"self", "cls"}
ENVIRON_WRITES = {"update", "setdefault", "pop", "popitem", "clear", "__setitem__", "__delitem__"}


def _private(name: str) -> bool:
    return name.startswith("_") and not (name.startswith("__") and name.endswith("__"))


def _private_reads(tree: ast.AST) -> Iterator[tuple[int, str]]:
    """``(line, what)`` of every private attribute, import or ``getattr`` that reaches past an object's surface."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and _private(node.attr):
            owner = node.value
            if not (isinstance(owner, ast.Name) and owner.id in OWN) and not (
                isinstance(owner, ast.Call) and isinstance(owner.func, ast.Name) and owner.func.id == "super"
            ):
                yield node.lineno, f".{node.attr}"
        elif isinstance(node, ast.ImportFrom) and node.module:
            parts = node.module.split(".")
            if any(_private(part) for part in parts):
                yield node.lineno, f"from {node.module}"
            yield from ((node.lineno, f"import {a.name}") for a in node.names if _private(a.name))
        elif isinstance(node, ast.Import):
            yield from ((node.lineno, f"import {a.name}") for a in node.names if any(map(_private, a.name.split("."))))
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ("getattr", "hasattr"):
            name = node.args[1] if len(node.args) > 1 else None
            if isinstance(name, ast.Constant) and isinstance(name.value, str) and _private(name.value):
                yield node.lineno, f"{node.func.id}(..., {name.value!r})"


def _is_environ(node: ast.AST) -> bool:
    return isinstance(node, ast.Attribute) and node.attr == "environ" and isinstance(node.value, ast.Name)


def _environ_writes(tree: ast.AST) -> Iterator[int]:
    """Lines that assign, delete or update an item of ``os.environ``, or call ``os.putenv`` / ``os.unsetenv``."""
    for node in ast.walk(tree):
        targets = []
        if isinstance(node, ast.Assign | ast.Delete):
            targets = node.targets
        elif isinstance(node, ast.AugAssign | ast.AnnAssign):
            targets = [node.target]
        if any(
            _is_environ(target) or (isinstance(target, ast.Subscript) and _is_environ(target.value))
            for target in targets
        ):
            yield node.lineno
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if (node.func.attr in ENVIRON_WRITES and _is_environ(node.func.value)) or node.func.attr in (
                "putenv",
                "unsetenv",
            ):
                yield node.lineno


def _scan(root: Path, finder) -> list[str]:  # noqa: ANN001 - one of the two finders above
    return [
        f"{path.relative_to(root.parent)}:{found[0] if isinstance(found, tuple) else found}"
        + (f": {found[1]}" if isinstance(found, tuple) else "")
        for path in sorted(root.rglob("*.py"))
        for found in finder(ast.parse(path.read_text()))
    ]


def test_the_studio_reads_no_private_name() -> None:
    found = _scan(SRC / "studio", _private_reads)
    assert not found, "the studio reads private names (record 0009: public names only):\n" + "\n".join(found)


def test_no_module_writes_the_environment() -> None:
    found = _scan(SRC, _environ_writes)
    assert not found, "writes to os.environ under src (record 0009; read weights through io.models_dir()):\n" + (
        "\n".join(found)
    )


def test_the_scans_see_every_planted_spelling() -> None:
    private = (
        "from better_human.mass.topology import _majority_face_parts\nimport better_robot.lie._checks\n"
        "from better_robot.lie._checks import check_dtype\nlayer._handles\ngetattr(tabs, '_tab_handles', ())\n"
        "self._track(handle)\nsuper()._fade(0.5)\nlayer.__class__\n"
    )
    assert sorted(what for _, what in _private_reads(ast.parse(private))) == [
        "._handles",
        "from better_robot.lie._checks",
        "getattr(..., '_tab_handles')",
        "import _majority_face_parts",
        "import better_robot.lie._checks",
    ]
    environ = (
        "os.environ['X'] = '1'\nos.environ.update(X='1')\nos.environ.setdefault('X', '1')\ndel os.environ['X']\n"
        "os.putenv('X', '1')\nos.environ['X'] += '1'\nvalue = os.environ.get('X')\n"
    )
    assert sorted(_environ_writes(ast.parse(environ))) == [1, 2, 3, 4, 5, 6]
