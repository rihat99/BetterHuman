"""File formats: the name and version every BetterHuman file carries, their one check, and the migrations.

A file names its ``format`` (``"better_human.mhr"``) and its ``format_version``. A reader calls :func:`upgrade`, which
migrates an older version in memory one step at a time, so files on disk stay valid, and then :func:`check_format`; a
writer writes :func:`header`. Inside the package, this module is the one place that reads or writes
``format_version`` (decision record 0012; ``src/better_human/io/schema.py`` lists it among the archive fields). The
converters under ``tools/convert`` run without BetterHuman and spell the header out; ``tests/contract/test_formats.py``
checks it against :func:`header`. Each format's spec is ``docs/reference/formats/<name>.md``.
"""

from __future__ import annotations

import operator
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
from better_robot.exceptions import BetterRobotError

MHR = "better_human.mhr"
SOMA = "better_human.soma"
FRAME_SET = "better_human.frame_set"
SEGMENTATION = "better_human.segmentation"
DENSITY_TABLES = "better_human.density_tables"

#: The version each format's readers read and its writers write.
LATEST = {MHR: 2, SOMA: 2, FRAME_SET: 1, SEGMENTATION: 1, DENSITY_TABLES: 1}


def header(format: str) -> dict[str, Any]:
    """The ``format`` and ``format_version`` fields a writer of ``format`` writes: its latest version."""
    return {"format": format, "format_version": LATEST[format]}


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
    if "format_version" not in data:
        raise FormatError(
            f"{source}: a {format} file without format_version. Write it with this version's writer or converter."
        )
    found_version = _version(data, format=format, source=source)
    if found_version > version:
        raise FormatError(
            f"{source}: {format} format_version {found_version} is later than {version}, the latest this BetterHuman "
            "reads. Upgrade better-human, or reconvert the file with this version's converter."
        )


def upgrade(fields: Mapping[str, Any], *, format: str, source: str = "<mapping>") -> dict[str, Any]:
    """``fields`` migrated in memory to the latest version of ``format``, then checked by :func:`check_format`.

    A file that names its ``format`` is checked before any migration, so a file of another format is refused rather
    than migrated into this one; a file without ``format`` or ``format_version`` is version 0. Raises
    :class:`FormatError` for another format, a later version, or a version no migration starts from.
    """
    latest = LATEST[format]
    if "format" in fields:
        check_format(fields, format=format, version=latest, source=source)
    upgraded = dict(fields)
    while (version := _version(upgraded, format=format, source=source)) < latest:
        migrate = MIGRATIONS.get((format, version))
        if migrate is None:
            raise FormatError(
                f"{source}: {format} format_version {version} has no migration to version {latest}. "
                f"Pass a {format} file, or reconvert it with this version's converter."
            )
        missing = [name for name in PAST_FIELDS.get((format, version), ()) if name not in upgraded]
        if missing:
            raise FormatError(
                f"{source}: a {format} format_version {version} file without its field(s) {missing}, which the "
                f"migration to version {version + 1} reads. Reconvert it with this version's converter."
            )
        upgraded = migrate(upgraded)
    check_format(upgraded, format=format, version=latest, source=source)
    return upgraded


def _mhr_v1_to_v2(fields: Mapping[str, Any]) -> dict[str, Any]:
    """v2 adds ``format`` and ``provenance``; a v1 archive keeps its arrays and gains provenance ``"unknown (v1)"``."""
    return {**fields, "format": np.str_(MHR), "format_version": np.int64(2), "provenance": np.str_("unknown (v1)")}


def _soma_v1_to_v2(fields: Mapping[str, Any]) -> dict[str, Any]:
    """v2 adds ``format``, and v1's provenance JSON ``source`` (asset commit, tag, sha256) becomes ``provenance``."""
    kept = {name: value for name, value in fields.items() if name != "source"}
    return {**kept, "format": np.str_(SOMA), "format_version": np.int64(2), "provenance": np.str_(fields["source"])}


def _frame_set_v0_to_v1(fields: Mapping[str, Any]) -> dict[str, Any]:
    """v1 wraps the frames in a header and writes every entry as ``{vertex_id, joint_name}``; a bare vertex id gets
    joint_name null."""
    frames = {
        name: {"vertex_id": value["vertex"], "joint_name": value.get("joint")}
        if isinstance(value, Mapping)
        else {"vertex_id": value, "joint_name": None}
        for name, value in fields.items()
    }
    return {"format": FRAME_SET, "format_version": 1, "frames": frames}


#: ``(format, version) -> migration`` to ``version + 1``.
MIGRATIONS: dict[tuple[str, int], Callable[[Mapping[str, Any]], dict[str, Any]]] = {
    (MHR, 1): _mhr_v1_to_v2,
    (SOMA, 1): _soma_v1_to_v2,
    (FRAME_SET, 0): _frame_set_v0_to_v1,
}

#: ``(format, version) -> fields`` that only that version carries and its migration reads.
PAST_FIELDS: dict[tuple[str, int], tuple[str, ...]] = {(SOMA, 1): ("source",)}


def past_fields(fields: Mapping[str, Any], *, format: str, source: str = "<mapping>") -> tuple[str, ...]:
    """The fields a reader reads besides the latest schema's, for the version ``fields`` names: v1 SOMA's ``source``."""
    return PAST_FIELDS.get((format, _version(fields, format=format, source=source)), ())


def _version(fields: Mapping[str, Any], *, format: str, source: str) -> int:
    """``fields["format_version"]`` as an int, 0 when absent; :class:`FormatError` when it is not an integer."""
    value = fields.get("format_version", 0)
    try:
        return operator.index(value)
    except TypeError:
        raise FormatError(
            f"{source}: {format} format_version {value!r} is not an integer. "
            "Write the file with this version's writer or converter."
        ) from None
