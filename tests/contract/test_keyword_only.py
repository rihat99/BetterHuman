"""BetterRobot C-API-2: public callables take at most two positional parameters.

Every function, class constructor (its ``__init__``, a dataclass's fields included), method, class method and
static method that the root's ``__all__`` tree lists (``better_robot.testing.public_objects``, each object once,
under the first path that lists it) is inspected; parameters after the second (``self`` and ``cls`` excluded) must be
keyword-only. A ``*args`` parameter takes any number of arguments, so it counts as three on its own.
``tests/contract/ratchets/positional_parameters.json`` holds today's violations (path -> its positional count and
the reason); it only shrinks, and a listed callable may not take one more positional parameter than its entry says.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable
from pathlib import Path

import pytest
from better_robot.exceptions import UnknownNameError
from better_robot.testing import public_objects

import better_human as bh

MAX_POSITIONAL = 2
ALLOWED: dict[str, int] = {
    path: entry["count"]
    for path, entry in json.loads(
        (Path(__file__).parent / "ratchets" / "positional_parameters.json").read_text()
    ).items()
}


def _callables() -> dict[str, Callable[..., object]]:
    """Each public function, class (its constructor) and method once, under the first path that lists it."""
    found: dict[int, tuple[str, Callable[..., object]]] = {}
    for path, listed in public_objects("better_human").items():
        value = listed.__func__ if isinstance(listed, staticmethod | classmethod) else listed
        if inspect.isfunction(value) or (inspect.isclass(value) and inspect.isfunction(value.__init__)):
            found.setdefault(id(value), (path, value))
    return dict(found.values())


def _positional_count(function: Callable[..., object]) -> int:
    parameters = list(inspect.signature(function).parameters.values())
    if parameters and parameters[0].name in {"self", "cls"}:
        parameters = parameters[1:]
    named = sum(
        parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD) for parameter in parameters
    )
    variadic = any(parameter.kind is parameter.VAR_POSITIONAL for parameter in parameters)
    return named + (MAX_POSITIONAL + 1) * variadic


CALLABLES = _callables()


@pytest.mark.parametrize("path", sorted(CALLABLES))
def test_public_callable_is_keyword_only_after_its_second_argument(path: str) -> None:
    count = _positional_count(CALLABLES[path])
    assert count <= ALLOWED.get(path, MAX_POSITIONAL), (
        f"{path} takes {count} positional parameters; make the ones after the second keyword-only (`*`, or "
        "`kw_only=True` on a dataclass)"
    )


def test_ratchet_names_only_violations() -> None:
    """A fixed entry leaves the ratchet with its fix, and an entry states the positional count it allows."""
    stale = sorted(
        f"{path}: entry {allowed}, found {_positional_count(CALLABLES[path]) if path in CALLABLES else 'no callable'}"
        for path, allowed in ALLOWED.items()
        if path not in CALLABLES or _positional_count(CALLABLES[path]) != allowed
    )
    assert stale == []


def test_load_shows_every_option_and_dispatches_to_every_family_loader() -> None:
    """``bh.load`` takes no ``**kwargs``, and its literal table is the loaders in ``families.__all__``."""
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
