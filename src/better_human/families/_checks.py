"""Argument checks every family loader runs first, before it reads a file."""

from __future__ import annotations

import torch
from better_robot.exceptions import ArgumentError, DtypeMismatchError


def check_dtype(dtype: torch.dtype) -> None:
    """Raise :class:`~better_robot.exceptions.DtypeMismatchError` for a ``dtype`` other than float32 or float64.

    The one dtype rule of the library's loaders (``better_human.load`` passes ``dtype`` to them).
    """
    if dtype not in (torch.float32, torch.float64):
        raise DtypeMismatchError(
            f"dtype={dtype!r} is not torch.float32 or torch.float64, the floating dtypes BetterRobot computes in; "
            "pass dtype=torch.float32 or dtype=torch.float64"
        )


def check_flags(**flags: object) -> None:
    """Raise :class:`~better_robot.exceptions.ArgumentError` naming the first of ``flags`` that is not a bool."""
    for name, value in flags.items():
        if not isinstance(value, bool):
            raise ArgumentError(f"{name} must be bool (True or False), got {value!r}")


__all__ = ["check_dtype", "check_flags"]
