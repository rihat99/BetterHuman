"""Decision records 0002 and 0008 (BR C-EXT-2): every class in ``src`` is abstract or ``@final``.

The exceptions are the entries of the ratchet ``tests/contract/ratchets/abstract_or_final.json`` (classes whose owning
work package has not decided yet; it only shrinks). Exception classes are excluded: they are never final.
"""

from __future__ import annotations

import importlib
import inspect
import json
import pkgutil
from pathlib import Path

import pytest

import better_human

RATCHET = Path(__file__).resolve().parent / "ratchets" / "abstract_or_final.json"


def _classes() -> list[tuple[str, type]]:
    """Every class defined under ``better_human`` by qualified name, exceptions excluded."""
    found: dict[str, type] = {}
    for _, name, _ in pkgutil.walk_packages(better_human.__path__, "better_human."):
        module = importlib.import_module(name)
        for attribute, value in vars(module).items():
            if inspect.isclass(value) and value.__module__ == name and not issubclass(value, BaseException):
                found[f"{name}.{attribute}"] = value
    return sorted(found.items())


def _undecided(cls: type) -> bool:
    return not (inspect.isabstract(cls) or getattr(cls, "__final__", False))


@pytest.mark.parametrize(("qualname", "cls"), _classes(), ids=[qualname for qualname, _ in _classes()])
def test_class_is_abstract_or_final(qualname: str, cls: type) -> None:
    ratchet = json.loads(RATCHET.read_text())
    assert not _undecided(cls) or qualname in ratchet, f"{qualname} is neither abstract nor @final"


def test_ratchet_names_only_undecided_classes() -> None:
    """An entry whose class is gone or decided leaves with the change that decided it."""
    undecided = {qualname for qualname, cls in _classes() if _undecided(cls)}
    stale = sorted(set(json.loads(RATCHET.read_text())) - undecided)
    assert stale == [], f"drop from the ratchet: {stale}"
