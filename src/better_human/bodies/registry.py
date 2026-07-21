"""Decorator-backed registry for concrete parametric bodies."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from better_human.core import ParametricBody

BodyFactory = Callable[..., ParametricBody]
FactoryType = TypeVar("FactoryType", bound=BodyFactory)

_BODY_FACTORIES: dict[str, BodyFactory] = {}


def register(name: str) -> Callable[[FactoryType], FactoryType]:
    """Register a concrete body factory under a lowercase public name."""
    normalized = name.lower()

    def decorator(factory: FactoryType) -> FactoryType:
        if normalized in _BODY_FACTORIES:
            raise ValueError(f"A body factory is already registered as {normalized!r}")
        _BODY_FACTORIES[normalized] = factory
        return factory

    return decorator


def load(name: str, **configuration: Any) -> ParametricBody:
    """Construct the body registered as ``name`` with ``configuration``."""
    normalized = name.lower()
    try:
        factory = _BODY_FACTORIES[normalized]
    except KeyError as error:
        supported = ", ".join(sorted(_BODY_FACTORIES))
        raise ValueError(f"Unknown body {name!r}; choose one of: {supported}") from error
    return factory(**configuration)


__all__ = ["load", "register"]
