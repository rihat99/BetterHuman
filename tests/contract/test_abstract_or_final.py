"""BetterRobot's C-EXT-2: every class in ``src`` is abstract or ``@final``, and none derives from
a ``@final`` class (rule H-API-1: a partial model is a loader option, never a subclass).

The exceptions are the entries of the ratchet ``tests/contract/ratchets/abstract_or_final.json`` (classes whose
abstract-or-final decision is still open; it only shrinks). Exception classes are excluded: they are never final.
"""

from __future__ import annotations

import importlib
import inspect
import json
import pkgutil
from pathlib import Path
from typing import final

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
    # The class's own __dict__: a subclass of a @final class inherits __final__ without being final itself.
    return not (inspect.isabstract(cls) or cls.__dict__.get("__final__", False))


def _final_bases(cls: type) -> list[str]:
    """The qualified names of the ``@final`` classes ``cls`` derives from."""
    return [f"{base.__module__}.{base.__qualname__}" for base in cls.__mro__[1:] if base.__dict__.get("__final__")]


@pytest.mark.parametrize(("qualname", "cls"), _classes(), ids=[qualname for qualname, _ in _classes()])
def test_no_class_derives_from_a_final_class(qualname: str, cls: type) -> None:
    assert not _final_bases(cls), f"{qualname} derives from the @final {_final_bases(cls)} (rule H-API-1)"


@pytest.mark.parametrize(("qualname", "cls"), _classes(), ids=[qualname for qualname, _ in _classes()])
def test_class_is_abstract_or_final(qualname: str, cls: type) -> None:
    ratchet = json.loads(RATCHET.read_text())
    assert not _undecided(cls) or qualname in ratchet, f"{qualname} is neither abstract nor @final"


def test_a_subclass_of_a_final_class_is_undecided() -> None:
    @final
    class Sealed:
        pass

    class Inheritor(Sealed):  # type: ignore[misc]
        pass

    @final
    class SealedInheritor(Sealed):  # type: ignore[misc]
        pass

    assert not _undecided(Sealed)
    assert _undecided(Inheritor)
    assert _final_bases(Inheritor) == _final_bases(SealedInheritor) == [f"{__name__}.{Sealed.__qualname__}"]


def test_ratchet_names_only_undecided_classes() -> None:
    """An entry whose class is gone or decided leaves with the change that decided it."""
    undecided = {qualname for qualname, cls in _classes() if _undecided(cls)}
    stale = sorted(set(json.loads(RATCHET.read_text())) - undecided)
    assert stale == [], f"drop from the ratchet: {stale}"
