"""Decision record 0001: ``HumanModel``, ``Identity`` and ``HumanState`` are frozen and compare by identity.

Every dataclass a public ``__all__`` exports is frozen; the three tensor-holding records also set ``eq=False``,
because ``==`` on tensors is elementwise (their field tables are ``tests/contract/test_naming.py``'s).
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect

import pytest

import better_human as bh

PACKAGES = ["better_human", "better_human.core", "better_human.families", "better_human.io", "better_human.mass"]


def _public_dataclasses() -> list[tuple[str, type]]:
    found = {}
    for package in PACKAGES:
        module = importlib.import_module(package)
        for name in module.__all__:
            value = getattr(module, name)
            if inspect.isclass(value) and dataclasses.is_dataclass(value):
                found[f"{value.__module__}.{value.__qualname__}"] = value
    return sorted(found.items())


@pytest.mark.parametrize(("qualname", "cls"), _public_dataclasses(), ids=[name for name, _ in _public_dataclasses()])
def test_public_dataclass_is_frozen(qualname: str, cls: type) -> None:
    assert cls.__dataclass_params__.frozen, f"{qualname} must be @dataclass(frozen=True)"


@pytest.mark.parametrize("cls", [bh.HumanModel, bh.Identity, bh.HumanState], ids=lambda cls: cls.__name__)
def test_tensor_records_compare_by_identity(cls: type) -> None:
    assert not cls.__dataclass_params__.eq, f"{cls.__name__} must set eq=False"
    assert cls.__eq__ is object.__eq__ and cls.__hash__ is object.__hash__
