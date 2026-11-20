"""The declared caches and no other (principles 5.6 and 11.6).

``HumanState._memo`` (a state's outputs) and ``Identity._frame_placements`` (an identity's contact-frame placements)
are the only caches: both are ``init=False``, so ``dataclasses.replace`` starts them empty, each is reached by one
function, and each key carries the grad mode, so a value made under ``no_grad`` never serves a differentiable call.
A new cache fails here: a dataclass field with ``init=False``, a module-level empty container, a ``functools`` cache
outside ``FUNCTOOLS_CACHES`` (each entry with its reason; none today), or a declared cache reached from another
function.
"""

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path

import torch

import better_human as bh
from tests.support.random_archives import random_weight_spec

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src" / "better_human"
HUMAN_MODEL = "src/better_human/core/human_model.py"
CACHES = {
    "Identity._frame_placements": (HUMAN_MODEL, "_contact_placements"),
    "HumanState._memo": (HUMAN_MODEL, "_memoised"),
}
FUNCTOOLS_CACHE_NAMES = {"cache", "lru_cache", "cached_property"}
FUNCTOOLS_CACHES: dict[str, str] = {}
STORES = {"dict", "list", "set", "defaultdict", "OrderedDict", "Counter", "WeakKeyDictionary", "WeakValueDictionary"}
FRAMES = {"heel": (3458, None), "tip": (412, "neck")}


def _modules() -> list[tuple[str, ast.Module]]:
    return [(file.relative_to(REPO).as_posix(), ast.parse(file.read_text())) for file in sorted(SRC.rglob("*.py"))]


def _late_fields(tree: ast.Module) -> list[str]:
    """``Class.field`` of every class-body field declared with ``field(..., init=False)``."""
    return [
        f"{node.name}.{ast.unparse(item.target)}"
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef)
        for item in node.body
        if isinstance(item, ast.AnnAssign)
        and isinstance(item.value, ast.Call)
        and any(
            keyword.arg == "init" and isinstance(keyword.value, ast.Constant) and keyword.value.value is False
            for keyword in item.value.keywords
        )
    ]


def _module_stores(tree: ast.Module) -> list[str]:
    """Every module-level name bound to an empty container: a store a function could fill as a cache."""
    found = []
    for statement in tree.body:
        value = statement.value if isinstance(statement, ast.Assign | ast.AnnAssign) else None
        empty_literal = isinstance(value, ast.Dict | ast.List | ast.Set) and not ast.unparse(value).strip("{}[]")
        call = value.func if isinstance(value, ast.Call) else None
        name = call.id if isinstance(call, ast.Name) else call.attr if isinstance(call, ast.Attribute) else None
        if empty_literal or (name in STORES and not value.args):
            found.append(ast.unparse(statement))
    return found


def _reaches(tree: ast.Module) -> list[tuple[str, str | None]]:
    """``(cache attribute, enclosing function or None)`` for every reach of a cache by attribute or string."""
    names = {cache.split(".")[1] for cache in CACHES}
    reaches: list[tuple[str, str | None]] = []

    def visit(node: ast.AST, function: str | None) -> None:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            function = node.name
        name = node.attr if isinstance(node, ast.Attribute) else node.value if isinstance(node, ast.Constant) else None
        if isinstance(name, str) and name in names:
            reaches.append((name, function))
        for child in ast.iter_child_nodes(node):
            visit(child, function)

    visit(tree, None)
    return reaches


def _functools_caches(tree: ast.Module) -> list[str | None]:
    """The enclosing function (``None`` at module level) of every ``functools`` cache, by module or by name."""
    imported = {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "functools"
        for alias in node.names
        if alias.name in FUNCTOOLS_CACHE_NAMES
    }
    found: list[str | None] = []

    def visit(node: ast.AST, function: str | None) -> None:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            function = node.name
        through_module = (
            isinstance(node, ast.Attribute)
            and node.attr in FUNCTOOLS_CACHE_NAMES
            and isinstance(node.value, ast.Name)
            and node.value.id == "functools"
        )
        if through_module or (isinstance(node, ast.Name) and node.id in imported):
            found.append(function)
        for child in ast.iter_child_nodes(node):
            visit(child, function)

    visit(tree, None)
    return found


def _state(*, grad: bool) -> bh.HumanState:
    human = bh.HumanModel.from_spec(random_weight_spec("smpl", dtype=torch.float64), density=None)
    return human.bake().forward(human.model.q_neutral.clone().requires_grad_(grad))


def test_the_declared_caches_are_the_only_late_fields() -> None:
    """A field outside the constructor is a cache; the declared ones stay out of ``init`` and ``repr``, and their
    classes compare by identity, so a cache never enters ``==``."""
    found = {field for _, tree in _modules() for field in _late_fields(tree)}
    assert found == CACHES.keys(), f"a new cache is the owner's decision; declare it in CACHES: {sorted(found)}"
    for cls in (bh.Identity, bh.HumanState):
        assert cls.__eq__ is object.__eq__, cls.__name__
        for field in dataclasses.fields(cls):
            if f"{cls.__name__}.{field.name}" in CACHES:
                assert not field.init and not field.repr, f"{cls.__name__}.{field.name}"


def test_no_module_level_store_in_src() -> None:
    """No module binds a name to an empty container, which a function would fill as a process-wide cache."""
    found = [f"{path}: {store}" for path, tree in _modules() for store in _module_stores(tree)]
    assert found == [], "a store that outlives its call is a cache:\n  " + "\n  ".join(found)


def test_only_its_function_reaches_a_cache() -> None:
    """Every reach of a cache, read or write, by attribute or by string, sits in the one function that owns it."""
    owners = {cache.split(".")[1]: owner for cache, owner in CACHES.items()}
    found = [
        f"{path}: {function} reaches {name}; only {owners[name][1]} may"
        for path, tree in _modules()
        for name, function in _reaches(tree)
        if (path, function) != owners[name]
    ]
    assert found == []


def test_functools_caches_are_the_reasoned_ones() -> None:
    """A ``functools`` cache lives as long as the process: each one needs its reason in ``FUNCTOOLS_CACHES``."""
    found = {f"{path}::{function}" for path, tree in _modules() for function in _functools_caches(tree)}
    assert found == FUNCTOOLS_CACHES.keys(), (
        f"functools caches without a reason: {sorted(found - FUNCTOOLS_CACHES.keys())}; "
        f"entries whose cache is gone: {sorted(FUNCTOOLS_CACHES.keys() - found)}"
    )


def test_replace_starts_with_empty_caches() -> None:
    """``dataclasses.replace`` on an identity or a state never carries a cache over."""
    state = _state(grad=False)
    state.contact_frame_poses(FRAMES)
    state.vertices
    assert state.identity._frame_placements and state._memo
    assert dataclasses.replace(state.identity)._frame_placements == {}
    assert dataclasses.replace(state)._memo == {}


def test_a_value_made_under_no_grad_never_serves_a_grad_call() -> None:
    """Both caches key by the grad mode: a ``no_grad`` read first, then a differentiable read gets a graph."""
    state = _state(grad=True)
    with torch.no_grad():
        assert not state.vertices.requires_grad
        assert not state.contact_frame_poses(FRAMES).requires_grad
    vertices, frames = state.vertices, state.contact_frame_poses(FRAMES)
    assert vertices.requires_grad and frames.requires_grad
    torch.autograd.grad((vertices.sum() + frames.sum()), state.q)
    assert {key.rsplit("/", 1)[1] for key in state._memo} == {"False", "True"}
    assert {key[1] for key in state.identity._frame_placements} == {False, True}
