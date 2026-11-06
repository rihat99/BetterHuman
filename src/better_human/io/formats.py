"""File formats: the name and version every BetterHuman file carries, their one check, and the migrations.

A file names its ``format`` (``"better_human.mhr"``) and its ``format_version``. A reader calls :func:`upgrade`, which
migrates an older version in memory one step at a time, so files on disk stay valid, and then :func:`check_format`.
This module is the one place that compares ``format_version`` (decision record 0012).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
from better_robot.exceptions import BetterRobotError

MHR = "better_human.mhr"
SOMA = "better_human.soma"
FRAME_SET = "better_human.frame_set"

#: The version each format's readers read and its writers write.
LATEST = {MHR: 2, SOMA: 2, FRAME_SET: 1}


class FormatError(BetterRobotError, ValueError):
    """A file is not the format its reader expects, is a later version than this BetterHuman reads, or is malformed."""


def check_format(data: Mapping[str, Any], *, format: str, version: int, source: str = "<mapping>") -> None:
    """Raise :class:`FormatError` unless ``data`` names ``format`` at ``format_version`` ``version`` or earlier.

    :param data: the fields of one file; ``data["format"]`` and ``data["format_version"]`` are read.
    :param format: the format the reader expects, e.g. ``"better_human.mhr"``.
    :param version: the latest version the reader reads.
    :param source: the file named in the message.
    """
    found = data.get("format")
    if found is None or str(found) != format:
        raise FormatError(
            f"{source}: expected a {format} file, got format {None if found is None else str(found)!r}. "
            f"Pass a {format} file, or reconvert it with this version's converter."
        )
    found_version = int(data.get("format_version", 0))
    if found_version > version:
        raise FormatError(
            f"{source}: {format} format_version {found_version} is later than {version}, the latest this BetterHuman "
            "reads. Upgrade better-human, or reconvert the file with this version's converter."
        )


def upgrade(fields: Mapping[str, Any], *, format: str, source: str = "<mapping>") -> dict[str, Any]:
    """``fields`` migrated in memory to the latest version of ``format``, then checked by :func:`check_format`.

    A missing ``format_version`` counts as 0. Raises :class:`FormatError` for a version no migration starts from.
    """
    latest = LATEST[format]
    upgraded = dict(fields)
    while (version := int(upgraded.get("format_version", 0))) < latest:
        migrate = MIGRATIONS.get((format, version))
        if migrate is None:
            raise FormatError(
                f"{source}: {format} format_version {version} has no migration to version {latest}. "
                f"Pass a {format} file, or reconvert it with this version's converter."
            )
        upgraded = migrate(upgraded)
    check_format(upgraded, format=format, version=latest, source=source)
    return upgraded


def _mhr_v1_to_v2(fields: Mapping[str, Any]) -> dict[str, Any]:
    """v2 adds ``format`` and ``provenance``; a v1 archive keeps its arrays and gains provenance ``"unknown (v1)"``."""
    return {**fields, "format": np.str_(MHR), "format_version": np.int64(2), "provenance": np.str_("unknown (v1)")}


def _soma_v1_to_v2(fields: Mapping[str, Any]) -> dict[str, Any]:
    """v2 adds ``format`` and ``provenance``; a v1 archive keeps its arrays and gains provenance ``"unknown (v1)"``."""
    return {**fields, "format": np.str_(SOMA), "format_version": np.int64(2), "provenance": np.str_("unknown (v1)")}


def _frame_set_v0_to_v1(fields: Mapping[str, Any]) -> dict[str, Any]:
    """v1 wraps the frames in a header and writes every entry as ``{vertex, joint}``; a bare vertex id gets joint null."""
    frames = {
        name: value if isinstance(value, Mapping) else {"vertex": value, "joint": None}
        for name, value in fields.items()
    }
    return {"format": FRAME_SET, "format_version": 1, "frames": frames}


#: ``(format, version) -> migration`` to ``version + 1``.
MIGRATIONS: dict[tuple[str, int], Callable[[Mapping[str, Any]], dict[str, Any]]] = {
    (MHR, 1): _mhr_v1_to_v2,
    (SOMA, 1): _soma_v1_to_v2,
    (FRAME_SET, 0): _frame_set_v0_to_v1,
}
