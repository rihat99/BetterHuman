"""Decision record 0009: the studio is an application on the public BetterHuman and BetterRobot surface.

Every BetterHuman or BetterRobot name a module of ``better_human.studio`` imports from outside the studio is a member
of a package ``__all__`` tree (``better_robot.testing.public_objects``). No studio module reads a ``_``-prefixed
attribute that its own class does not define: a studio ``Layer`` may call the hooks BetterRobot's ``Layer`` documents
for subclasses (``_track``, ``_store_opacity``, ``_fade``) and nothing else of its base. No studio module imports a
``_``-prefixed name or module, or reaches one through ``getattr`` or ``vars``. No module under ``src/better_human``
writes ``os.environ`` under any spelling: the weights directory is the user's ``$BETTERHUMAN_MODELS_DIR``, read
through ``better_human.io.models_dir()`` (record 0017).
"""

from __future__ import annotations

import ast
from collections.abc import Callable, Iterator
from functools import cache
from pathlib import Path

from better_robot.testing import public_objects

SRC = Path(__file__).resolve().parents[2] / "src" / "better_human"
STUDIO = "better_human.studio"
LIBRARIES = ("better_human", "better_robot")
OWN = {"self", "cls"}
SUBCLASS_HOOKS = {"_track", "_store_opacity", "_fade"}
ENVIRON_WRITES = {"update", "setdefault", "pop", "popitem", "clear", "__setitem__", "__delitem__"}


@cache
def _public_paths() -> frozenset[str]:
    packages = [
        "better_human",
        "better_robot",
        # BetterRobot documents its viewer tools as public, but better_robot.viewer's __all__ does not list them yet.
        "better_robot.viewer.tools",
    ]
    return frozenset(path for package in packages for path in public_objects(package))


def _private(name: str) -> bool:
    return name.startswith("_") and not (name.startswith("__") and name.endswith("__"))


def _imported(node: ast.Import | ast.ImportFrom) -> Iterator[str]:
    """The absolute dotted path of every name an import binds (a relative import resolves inside the studio)."""
    if isinstance(node, ast.Import):
        yield from (alias.name for alias in node.names)
        return
    package = STUDIO.split(".")[: len(STUDIO.split(".")) - max(node.level - 1, 0)] if node.level else []
    module = ".".join([*package, *([node.module] if node.module else [])])
    yield from (f"{module}.{alias.name}" for alias in node.names)


def _outside_the_surface(path: str) -> bool:
    parts = path.split(".")
    if any(map(_private, parts)):
        return True
    in_studio = path == STUDIO or path.startswith(STUDIO + ".")
    return parts[0] in LIBRARIES and not in_studio and path not in _public_paths()


def _own_names(cls: ast.ClassDef) -> set[str]:
    """What a class defines itself: its methods and class attributes, and every ``self.<name>`` it assigns."""
    own = {node.name for node in cls.body if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)}
    for statement in cls.body:
        if isinstance(statement, ast.Assign | ast.AnnAssign):
            targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
            own |= {target.id for target in targets if isinstance(target, ast.Name)}
    return own | {
        node.attr
        for node in ast.walk(cls)
        if isinstance(node, ast.Attribute)
        and isinstance(node.ctx, ast.Store)
        and isinstance(node.value, ast.Name)
        and node.value.id in OWN
    }


def _private_reads(tree: ast.AST) -> Iterator[tuple[int, str]]:
    """``(line, what)`` of every name or attribute that reaches past the public surface or the class's own state."""
    own_of: dict[int, set[str]] = {}
    for cls in (node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)):
        own = _own_names(cls) | SUBCLASS_HOOKS
        own_of.update((id(node), own) for node in ast.walk(cls))
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and _private(node.attr):
            owner = node.value
            if isinstance(owner, ast.Name) and owner.id in OWN:
                allowed = own_of.get(id(node), set())
            elif isinstance(owner, ast.Call) and isinstance(owner.func, ast.Name) and owner.func.id == "super":
                allowed = SUBCLASS_HOOKS
            else:
                allowed = set()
            if node.attr not in allowed:
                yield node.lineno, f".{node.attr}"
        elif isinstance(node, ast.Import | ast.ImportFrom):
            yield from ((node.lineno, f"import {path}") for path in _imported(node) if _outside_the_surface(path))
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "vars" and node.args:
                yield node.lineno, "vars(...)"
            elif node.func.id in ("getattr", "hasattr", "setattr", "delattr") and len(node.args) > 1:
                name = node.args[1]
                if isinstance(name, ast.Constant) and isinstance(name.value, str) and _private(name.value):
                    yield node.lineno, f"{node.func.id}(..., {name.value!r})"


def _environ_writes(tree: ast.AST) -> Iterator[int]:
    """Lines that assign, delete or update an item of ``os.environ``, or call ``os.putenv`` / ``os.unsetenv``, under
    any alias (``import os as x``, ``from os import environ as e``, ``e = os.environ``)."""
    aliases = {
        alias.asname or alias.name: alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "os"
        for alias in node.names
    }

    assigned: set[str] = set()

    def is_environ(node: ast.AST) -> bool:
        if isinstance(node, ast.Name):
            return aliases.get(node.id) == "environ" or node.id in assigned
        return isinstance(node, ast.Attribute) and node.attr == "environ" and isinstance(node.value, ast.Name)

    assignments = [node for node in ast.walk(tree) if isinstance(node, ast.Assign | ast.AnnAssign) and node.value]
    while new := {  # names assigned from os.environ, then from those names, until nothing new
        target.id
        for node in assignments
        if is_environ(node.value)
        for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
        if isinstance(target, ast.Name) and target.id not in assigned
    }:
        assigned |= new

    for node in ast.walk(tree):
        targets = []
        if isinstance(node, ast.Assign | ast.Delete):
            targets = node.targets
        elif isinstance(node, ast.AugAssign | ast.AnnAssign):
            targets = [node.target]
        if any(  # binding an alias (``_ENV = os.environ``) is no write; writing through it is
            (is_environ(target) and not (isinstance(target, ast.Name) and target.id in assigned))
            or (isinstance(target, ast.Subscript) and is_environ(target.value))
            for target in targets
        ):
            yield node.lineno
        elif isinstance(node, ast.Call):
            function = node.func
            if isinstance(function, ast.Attribute):
                writes = function.attr in ("putenv", "unsetenv") or (
                    function.attr in ENVIRON_WRITES and is_environ(function.value)
                )
            else:
                writes = isinstance(function, ast.Name) and aliases.get(function.id) in ("putenv", "unsetenv")
            if writes:
                yield node.lineno


def _scan(root: Path, finder: Callable[[ast.AST], Iterator[tuple[int, str] | int]]) -> list[str]:
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
        "from better_robot.lie._checks import check_dtype\nfrom better_human.core.human_model import Identity\n"
        "from better_robot.viewer.layers.mesh import MeshLayer\nfrom ..core import contact_frames\n"
        "from better_human.mass import PartTopology\nfrom better_robot.viewer.tools import PoseGizmo\n"
        "from better_human.studio.state import StudioSession\nfrom .state import StudioSession\nimport better_human\n"
        "layer._handles\ngetattr(tabs, '_tab_handles', ())\nvars(tabs)\n"
        "class Mesh(Layer):\n"
        "    def __init__(self):\n        self._meshes = []\n"
        "    def set_step(self, step):\n"
        "        self._track(self._meshes)\n        super()._fade(0.5)\n        self.__class__\n"
        "        self._visible\n        super()._handles\n"
    )
    assert sorted(what for _, what in _private_reads(ast.parse(private))) == [
        "._handles",
        "._handles",
        "._visible",
        "getattr(..., '_tab_handles')",
        "import better_human.core.contact_frames",
        "import better_human.core.human_model.Identity",
        "import better_human.mass.topology._majority_face_parts",
        "import better_robot.lie._checks",
        "import better_robot.lie._checks.check_dtype",
        "import better_robot.viewer.layers.mesh.MeshLayer",
        "vars(...)",
    ]
    environ = (
        "os.environ['X'] = '1'\nos.environ.update(X='1')\nos.environ.setdefault('X', '1')\ndel os.environ['X']\n"
        "os.putenv('X', '1')\nos.environ['X'] += '1'\nfrom os import environ as env, unsetenv\nenv['X'] = '1'\n"
        "unsetenv('X')\nvalue = os.environ.get('X')\nvalue = env.get('X')\n_ENV = os.environ\nalias = _ENV\n"
        "alias['X'] = '1'\n_ENV.update(X='1')\n"
    )
    assert sorted(_environ_writes(ast.parse(environ))) == [1, 2, 3, 4, 5, 6, 8, 9, 14, 15]
