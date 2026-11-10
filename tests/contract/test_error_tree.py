"""One error tree (BR decision record 0031): BetterHuman raises BetterRobot's exception classes and its own subclasses.

Every exception class under ``better_human`` derives from ``BetterRobotError``; ``src`` raises a builtin
``ValueError``, ``TypeError``, ``RuntimeError`` or ``KeyError`` only where ``PLAIN_RAISES`` (permanent, reasoned) or the
ratchet ``tests/contract/ratchets/plain_raises.json`` (shrinks as the classic, families and viz rewrites land) allows.
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
BUILTIN_RAISES = {"ValueError", "TypeError", "RuntimeError", "KeyError"}
_NOT_A_TENSOR = "a value that is not a floating torch.Tensor is a Python TypeError; the tree has no node for it"
PLAIN_RAISES: dict[str, str] = {
    "src/better_human/core/human_model.py::_checked": _NOT_A_TENSOR,
    "src/better_human/core/human_model.py::_density": _NOT_A_TENSOR,
}


def _exception_classes() -> list[type]:
    found = []
    for _, name, _ in pkgutil.walk_packages(bh.__path__, "better_human."):
        for value in vars(importlib.import_module(name)).values():
            if inspect.isclass(value) and value.__module__ == name and issubclass(value, BaseException):
                found.append(value)
    return found


def _builtin_raise_sites() -> set[str]:
    """``path::function`` of every ``raise ValueError/TypeError/RuntimeError/KeyError(...)`` under ``src``."""
    sites: set[str] = set()

    def visit(node: ast.AST, file: Path, owner: str) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            owner = node.name
        if isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call):
            callee = node.exc.func
            name = callee.id if isinstance(callee, ast.Name) else getattr(callee, "attr", None)
            if name in BUILTIN_RAISES:
                sites.add(f"{file.relative_to(REPO).as_posix()}::{owner}")
        for child in ast.iter_child_nodes(node):
            visit(child, file, owner)

    for file in SRC.rglob("*.py"):
        visit(ast.parse(file.read_text()), file, "<module>")
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
