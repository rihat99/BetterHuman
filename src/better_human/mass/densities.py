"""Part densities from a named density table and the density category of each part."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from better_robot.exceptions import ArgumentError, UnknownNameError


def part_densities(
    tables: Mapping[str, Mapping[str, float]],
    name: str,
    *,
    categories: Sequence[str | None],
    part_names: Sequence[str],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """``(P,)`` kg/m³: part ``p`` takes table ``name``'s density of its category ``categories[p]``.

    :param tables: table name -> density category -> kg/m³ (``Parts.density_tables``).
    :param name: the table, e.g. ``"Dempster"``.
    :param categories: the category of each part, ``None`` for a part no category covers (``Parts.density_categories``).
    :param part_names: the name of each part, for the messages.
    Raises :class:`~better_robot.exceptions.UnknownNameError` for a table not in ``tables`` or a category the table
    lacks, and :class:`~better_robot.exceptions.ArgumentError` naming the parts without a category.
    """
    if name not in tables:
        raise UnknownNameError("density table", name, sorted(tables))
    uncovered = [part for part, category in zip(part_names, categories, strict=True) if category is None]
    if uncovered:
        raise ArgumentError(
            f"density={name!r}: {len(uncovered)} of {len(part_names)} parts have no density category: {uncovered}. "
            "Pass density=None (no inertias), a uniform float in kg/m^3 or a (P,) tensor, until the family declares "
            "a category for every part."
        )
    table = tables[name]
    for category in categories:
        if category not in table:
            raise UnknownNameError(f"{name} category name", category, list(table))
    return torch.tensor([table[category] for category in categories], dtype=dtype, device=device)
