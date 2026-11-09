"""H-API-2 / BR C-API-2: public callables take at most two positional parameters.

Every function in a public ``__all__`` and every public method of a public class is inspected; parameters after
the second (``self`` excluded) must be keyword-only. The violations that exist today sit in
``tests/contract/ratchets/keyword_only.json`` (qualified name -> the fix that removes it); the ratchet only shrinks,
and the fix that resolves a violation removes its entry.
"""

from __future__ import annotations

import importlib
import inspect
import json
from pathlib import Path

import pytest
from better_robot.exceptions import UnknownNameError

import better_human as bh

PACKAGES = [
    "better_human",
    "better_human.core",
    "better_human.families",
    "better_human.io",
    "better_human.terms",
    "better_human.viz",
    "better_human.viz.studio",
]
MAX_POSITIONAL = 2
KEYWORD_ONLY_RATCHET: dict[str, str] = json.loads(
    (Path(__file__).parent / "ratchets" / "keyword_only.json").read_text()
)


def _positional_count(fn: object, *, bound: bool) -> int:
    params = list(inspect.signature(fn).parameters.values())
    if bound and params:
        params = params[1:]
    kinds = (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    return sum(param.kind in kinds for param in params)


def _public_callables() -> dict[str, int]:
    counts: dict[str, int] = {}
    for package in PACKAGES:
        module = importlib.import_module(package)
        for name in getattr(module, "__all__", ()):
            obj = getattr(module, name)
            if inspect.isclass(obj):
                for attr, member in vars(obj).items():
                    if attr.startswith("_") or not inspect.isfunction(member):
                        continue
                    counts[f"{obj.__module__}.{obj.__qualname__}.{attr}"] = _positional_count(member, bound=True)
            elif inspect.isfunction(obj):
                counts[f"{obj.__module__}.{obj.__qualname__}"] = _positional_count(obj, bound=False)
    return counts


def _violations() -> set[str]:
    return {name for name, count in _public_callables().items() if count > MAX_POSITIONAL}


def test_public_callables_are_keyword_only_after_second_argument() -> None:
    new = sorted(_violations() - KEYWORD_ONLY_RATCHET.keys())
    assert not new, f"positional parameters beyond {MAX_POSITIONAL} (make them keyword-only): {new}"


def test_keyword_only_ratchet_has_no_stale_entries() -> None:
    stale = sorted(KEYWORD_ONLY_RATCHET.keys() - _violations())
    assert not stale, f"fixed or renamed; remove from tests/contract/ratchets/keyword_only.json: {stale}"


def test_load_shows_every_option_and_dispatches_to_every_family_loader() -> None:
    """Record 0006: ``bh.load`` takes no ``**kwargs``, and its literal table is the loaders in ``families.__all__``."""
    parameters = inspect.signature(bh.load).parameters.values()
    assert all(param.kind is not inspect.Parameter.VAR_KEYWORD for param in parameters)
    loaders = {
        name: getattr(bh.families, name)
        for name in bh.families.__all__
        if inspect.isfunction(getattr(bh.families, name))
        and next(iter(inspect.signature(getattr(bh.families, name)).parameters)) == "path"
    }
    assert bh._LOADERS == loaders
    with pytest.raises(UnknownNameError, match="smplxx"):
        bh.load("smplxx")
