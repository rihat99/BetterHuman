"""Argument checks every family loader runs first, before it reads a file."""

from __future__ import annotations

from better_robot.exceptions import ArgumentError


def check_flags(**flags: object) -> None:
    """Raise :class:`~better_robot.exceptions.ArgumentError` naming the first of ``flags`` that is not a bool."""
    for name, value in flags.items():
        if not isinstance(value, bool):
            raise ArgumentError(f"{name} must be bool (True or False), got {value!r}")


__all__ = ["check_flags"]
