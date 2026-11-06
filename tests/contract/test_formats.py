"""Every BetterHuman file format names itself and its version, and old files keep loading (records 0012, 0013).

For each format in ``io.formats.LATEST``: the latest version round-trips, a fixture of every past version in
``tests/support/formats/`` reads through its migration and equals the latest fixture, a later version and another
format are refused naming both, and ``docs/reference/formats/<name>.md`` specifies it. ``format_version`` is compared
only in ``io/formats.py``. The fixtures are synthetic and frozen: every field a distinct constant; the v1 archives are
written as the converters wrote them before record 0012 (no ``format``, no ``provenance``; MHR's ``coordinate_axis``,
SOMA's ``lod`` and ``source``).
"""

from __future__ import annotations

import ast
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from better_human.io import FormatError, read_frame_set, write_frame_set
from better_human.io.formats import FRAME_SET, LATEST, MHR, MIGRATIONS, SOMA
from better_human.io.load import load_model_npz
from better_human.io.schema import MHR_SCHEMA, SOMA_SCHEMA

REPO = Path(__file__).resolve().parents[2]
FIXTURES = REPO / "tests" / "support" / "formats"
READERS: dict[str, tuple[str, Callable[[Path], Any]]] = {
    MHR: ("npz", lambda path: load_model_npz(path, MHR_SCHEMA)),
    SOMA: ("npz", lambda path: load_model_npz(path, SOMA_SCHEMA)),
    FRAME_SET: ("json", read_frame_set),
}


def _fixture(format: str, version: int) -> Path:
    return FIXTURES / f"{format.removeprefix('better_human.')}_v{version}.{READERS[format][0]}"


def _read(format: str, path: Path) -> Any:
    """What the reader returns; archive fields as NumPy arrays."""
    value = READERS[format][1](path)
    if format == FRAME_SET:
        return value
    return {name: array if isinstance(array, np.ndarray) else array.numpy() for name, array in value.items()}


def _comparable(format: str, value: Any) -> Any:
    """Archive fields as ``name -> (dtype, shape, bytes)``; a frame set as it is."""
    if format == FRAME_SET:
        return value
    return {name: (array.dtype, array.shape, array.tobytes()) for name, array in value.items()}


def _rewrite(source: Path, destination: Path, changes: dict[str, Any]) -> Path:
    """``source`` with the ``changes`` header fields, written to ``destination``."""
    if source.suffix == ".json":
        destination.write_text(json.dumps({**json.loads(source.read_text()), **changes}))
    else:
        with np.load(source, allow_pickle=False) as archive:
            np.savez_compressed(destination, **{**{name: archive[name] for name in archive.files}, **changes})
    return destination


@pytest.mark.parametrize("format", LATEST)
def test_latest_version_round_trips(format: str, tmp_path: Path) -> None:
    """What the reader returns, written back, is the file again (the archive writers are the converters)."""
    fixture = _fixture(format, LATEST[format])
    latest = _read(format, fixture)
    copy = tmp_path / fixture.name
    if format == FRAME_SET:
        write_frame_set(copy, latest)
        assert copy.read_text() == fixture.read_text()
    else:
        np.savez_compressed(copy, **latest)
    assert _comparable(format, _read(format, copy)) == _comparable(format, latest)


@pytest.mark.parametrize(
    ("format", "version"), sorted(MIGRATIONS), ids=[f"{name}_v{version}" for name, version in sorted(MIGRATIONS)]
)
def test_past_version_reads_through_migration_as_the_latest(format: str, version: int) -> None:
    past = _comparable(format, _read(format, _fixture(format, version)))
    latest = _comparable(format, _read(format, _fixture(format, LATEST[format])))
    if format != FRAME_SET:
        # A v1 archive's provenance is unknown; everything else is the v2 archive's.
        assert past.pop("provenance")[2] == np.asarray("unknown (v1)").tobytes()
        latest.pop("provenance")
    assert past == latest


@pytest.mark.parametrize("format", LATEST)
def test_later_version_is_refused_naming_both(format: str, tmp_path: Path) -> None:
    later = LATEST[format] + 1
    fixture = _fixture(format, LATEST[format])
    path = _rewrite(fixture, tmp_path / fixture.name, {"format_version": later})
    with pytest.raises(FormatError, match=rf"{format} format_version {later} is later than {LATEST[format]}"):
        _read(format, path)


@pytest.mark.parametrize("format", LATEST)
def test_another_format_is_refused_naming_both(format: str, tmp_path: Path) -> None:
    fixture = _fixture(format, LATEST[format])
    path = _rewrite(fixture, tmp_path / fixture.name, {"format": "x.y"})
    with pytest.raises(FormatError, match=rf"expected a {format} file, got format 'x.y'"):
        _read(format, path)


@pytest.mark.parametrize("format", LATEST)
def test_every_format_has_a_spec_page(format: str) -> None:
    page = REPO / "docs" / "reference" / "formats" / f"{format.removeprefix('better_human.')}.md"
    assert page.is_file(), f"{format} has no spec page {page.relative_to(REPO)}"


def _version_comparisons(src: Path) -> list[str]:
    """``"<path>, line <n>"`` of every comparison that mentions ``format_version`` outside ``io/formats.py``."""
    found = []
    for path in sorted(src.rglob("*.py")):
        if path.relative_to(src).as_posix() == "better_human/io/formats.py":
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Compare) and "format_version" in ast.unparse(node):
                found.append(f"{path.relative_to(src).as_posix()}, line {node.lineno}")
    return found


def test_format_version_is_compared_only_in_io_formats() -> None:
    offences = _version_comparisons(REPO / "src")
    assert not offences, "compare format_version through better_human.io.check_format:\n  " + "\n  ".join(offences)


def test_a_planted_comparison_is_found(tmp_path: Path) -> None:
    (tmp_path / "better_human" / "io").mkdir(parents=True)
    (tmp_path / "better_human" / "io" / "formats.py").write_text("ok = data['format_version'] < 2\n")
    (tmp_path / "better_human" / "reader.py").write_text("if int(data['format_version']) != 1:\n    pass\n")
    assert _version_comparisons(tmp_path) == ["better_human/reader.py, line 1"]
