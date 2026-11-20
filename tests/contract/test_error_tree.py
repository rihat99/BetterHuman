"""One error tree: BetterHuman raises BetterRobot's exception classes and its own subclasses.

Every exception class under ``better_human`` derives from ``BetterRobotError``; ``src`` raises a builtin
``ValueError``, ``TypeError``, ``RuntimeError``, ``KeyError``, ``IndexError`` or ``AssertionError`` (called or bare,
``raise ValueError``, or under another name: ``_Plain = ValueError``, ``from builtins import ValueError as _Plain``)
only where ``PLAIN_RAISES`` (permanent, reasoned) or the ratchet
``tests/contract/ratchets/plain_raises.json`` (shrinks as the classic, families and viz rewrites land) allows.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import json
import pkgutil
from pathlib import Path

import pytest
import torch
from better_robot.exceptions import BetterRobotError, UnknownNameError

import better_human as bh
from tests.support.synthetic import flame_spec

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src" / "better_human"
PLAIN_RAISES_RATCHET = REPO / "tests" / "contract" / "ratchets" / "plain_raises.json"
BUILTIN_RAISES = {"ValueError", "TypeError", "RuntimeError", "KeyError", "IndexError", "AssertionError"}
_NOT_A_TENSOR = "a value that is not a floating torch.Tensor is a Python TypeError; the tree has no node for it"
PLAIN_RAISES: dict[str, str] = {
    "src/better_human/core/human_model.py::_checked": _NOT_A_TENSOR,
    "src/better_human/core/human_model.py::_density": _NOT_A_TENSOR,
    "src/better_human/viz/layers.py::forces_layer": _NOT_A_TENSOR,
    "src/better_human/viz/layers.py::human_layers": _NOT_A_TENSOR,
}


def _exception_classes() -> list[type]:
    found = []
    for _, name, _ in pkgutil.walk_packages(bh.__path__, "better_human."):
        for value in vars(importlib.import_module(name)).values():
            if inspect.isclass(value) and value.__module__ == name and issubclass(value, BaseException):
                found.append(value)
    return found


def _name(node: ast.AST) -> str | None:
    return node.id if isinstance(node, ast.Name) else getattr(node, "attr", None)


def _builtin_aliases(tree: ast.AST) -> dict[str, str]:
    """Every other name a file gives a ``BUILTIN_RAISES`` class: ``_Plain = ValueError`` or an import from builtins."""
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "builtins":
            aliases.update({item.asname or item.name: item.name for item in node.names if item.name in BUILTIN_RAISES})
        elif isinstance(node, ast.Assign | ast.AnnAssign) and node.value is not None:
            builtin = _name(node.value)
            if builtin in BUILTIN_RAISES:
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                aliases.update({target.id: builtin for target in targets if isinstance(target, ast.Name)})
    return aliases


def _raised_builtin(node: ast.AST, aliases: dict[str, str]) -> str | None:
    """The ``BUILTIN_RAISES`` class ``node`` raises, called (``raise KeyError(x)``), bare (``raise ValueError``) or
    under one of the file's ``aliases``."""
    if not (isinstance(node, ast.Raise) and node.exc is not None):
        return None
    name = _name(node.exc.func if isinstance(node.exc, ast.Call) else node.exc)
    name = aliases.get(name, name)
    return name if name in BUILTIN_RAISES else None


def _builtin_raise_sites() -> set[str]:
    """``path::function`` of every raise of a ``BUILTIN_RAISES`` class under ``src``."""
    sites: set[str] = set()

    def visit(node: ast.AST, file: Path, owner: str, aliases: dict[str, str]) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            owner = node.name
        if _raised_builtin(node, aliases):
            sites.add(f"{file.relative_to(REPO).as_posix()}::{owner}")
        for child in ast.iter_child_nodes(node):
            visit(child, file, owner, aliases)

    for file in SRC.rglob("*.py"):
        tree = ast.parse(file.read_text())
        visit(tree, file, "<module>", _builtin_aliases(tree))
    return sites


def test_every_exception_class_derives_from_better_robot_error() -> None:
    classes = _exception_classes()
    assert classes, "the walk found no exception class (FormatError, AssetNotFoundError)"
    outside = sorted(cls.__qualname__ for cls in classes if not issubclass(cls, BetterRobotError))
    assert outside == [], f"derive from better_robot.exceptions.BetterRobotError: {outside}"


def test_src_raises_the_error_tree_only() -> None:
    """A builtin raise is allowed only by the permanent list or the ratchet, and every entry still has its raise."""
    sites = _builtin_raise_sites()
    ratchet = json.loads(PLAIN_RAISES_RATCHET.read_text())
    assert not set(PLAIN_RAISES) & set(ratchet), "an entry is either permanent or ratcheted"
    unexpected = sorted(sites - set(PLAIN_RAISES) - set(ratchet))
    assert unexpected == [], f"raise better_robot.exceptions classes instead of a builtin: {unexpected}"
    stale = sorted((set(PLAIN_RAISES) | set(ratchet)) - sites)
    assert stale == [], f"drop these entries with the fix that removed the raise: {stale}"


@pytest.mark.parametrize(
    ("statement", "builtin"),
    [
        ("raise ValueError", "ValueError"),
        ("raise KeyError(name)", "KeyError"),
        ("raise IndexError(index)", "IndexError"),
        ("raise AssertionError", "AssertionError"),
        ("raise builtins.RuntimeError", "RuntimeError"),
        ("_Plain = ValueError\nraise _Plain('x')", "ValueError"),
        ("from builtins import TypeError as _Plain\nraise _Plain", "TypeError"),
    ],
)
def test_every_form_of_a_builtin_raise_is_found(statement: str, builtin: str) -> None:
    tree = ast.parse(statement)
    aliases = _builtin_aliases(tree)
    assert [_raised_builtin(node, aliases) for node in ast.walk(tree) if _raised_builtin(node, aliases)] == [builtin]


@pytest.fixture
def human(tmp_path: Path) -> bh.HumanModel:
    return bh.HumanModel.from_spec(flame_spec(tmp_path), density=None)


@pytest.mark.slow
def test_a_misspelled_identity_group_raises_naming_the_groups(human: bh.HumanModel) -> None:
    with pytest.raises(UnknownNameError, match=r"unknown identity group 'beta' \(valid names: \['betas'\]\)"):
        human.bake(beta=torch.zeros(100, dtype=torch.float64))


@pytest.mark.slow
def test_a_misspelled_stream_group_raises_naming_the_groups(human: bh.HumanModel) -> None:
    streams = {"expresion": torch.zeros(50, dtype=torch.float64)}
    with pytest.raises(UnknownNameError, match=r"unknown stream group 'expresion' \(valid names: \['expression'\]\)"):
        human.bake().forward(human.model.q_neutral, streams)


@pytest.mark.slow
def test_a_value_that_is_not_a_tensor_is_a_type_error(human: bh.HumanModel) -> None:
    with pytest.raises(TypeError, match="betas must be a floating torch.Tensor"):
        human.bake(betas=[0.0] * 100)
