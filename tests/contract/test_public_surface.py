"""No module shares a name with an exported function, and no public function takes a
joint index (joints are named).

A module that shadows a function today sits in ``tests/contract/ratchets/module_names.json`` (module -> the fix that
renames it); the ratchet only shrinks.
"""

from __future__ import annotations

import importlib
import inspect
import json
import pkgutil
from collections.abc import Callable, Iterator
from pathlib import Path

import better_human

MODULE_NAME_RATCHET: dict[str, str] = json.loads((Path(__file__).parent / "ratchets" / "module_names.json").read_text())
JOINT_INDEX_PARAMETERS = {"joint_id", "joint_ids", "joint_index", "joint_indices"}


def _modules() -> list[str]:
    return ["better_human"] + [name for _, name, _ in pkgutil.walk_packages(better_human.__path__, "better_human.")]


def _exported() -> Iterator[tuple[str, str, object]]:
    """(package, exported name, object) for every name in a module's ``__all__``."""
    for name in _modules():
        module = importlib.import_module(name)
        for exported in getattr(module, "__all__", ()):
            yield name, exported, getattr(module, exported)


def _shadowing_modules() -> set[str]:
    """Submodules ``<package>.<f>`` of a package that exports a function ``f``."""
    shadowing = set()
    for package, exported, obj in _exported():
        module = importlib.import_module(package)
        if inspect.isfunction(obj) and hasattr(module, "__path__"):
            if exported in {sub for _, sub, _ in pkgutil.iter_modules(module.__path__)}:
                shadowing.add(f"{package}.{exported}")
    return shadowing


def _public_functions() -> Iterator[tuple[str, Callable]]:
    for _, exported, obj in _exported():
        if inspect.isfunction(obj):
            yield exported, obj
        elif inspect.isclass(obj):
            for attribute, member in vars(obj).items():
                if not attribute.startswith("_") and inspect.isfunction(member):
                    yield f"{exported}.{attribute}", member


def test_no_module_shares_a_name_with_an_exported_function() -> None:
    new = sorted(_shadowing_modules() - MODULE_NAME_RATCHET.keys())
    assert not new, f"rename these modules: {new}"


def test_module_name_ratchet_has_no_stale_entries() -> None:
    stale = sorted(MODULE_NAME_RATCHET.keys() - _shadowing_modules())
    assert not stale, f"fixed or renamed; remove from tests/contract/ratchets/module_names.json: {stale}"


def test_no_public_function_takes_a_joint_index() -> None:
    taking = sorted(
        f"{name}({parameter})"
        for name, function in _public_functions()
        for parameter in inspect.signature(function).parameters
        if parameter in JOINT_INDEX_PARAMETERS
    )
    assert taking == [], f"take joint names instead: {taking}"
