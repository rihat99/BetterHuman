"""Every BetterHuman file format names itself and its version, and old files keep loading (records 0012, 0013).

For each format in ``io.formats.LATEST``: the latest version round-trips, a fixture of every past version in
``tests/support/formats/`` reads through its migration and equals the latest fixture, a later version and another
format are refused naming both, and ``docs/reference/formats/<name>.md`` specifies it. No module but ``io/formats.py``
and ``io/schema.py`` names ``format_version`` (8.8). The fixtures are synthetic and frozen: every field a distinct
constant; the v1 archives are written as the converters wrote them before record 0012 (no ``format``, no
``provenance``; MHR's ``coordinate_axis``, SOMA's ``lod`` and ``source``).
"""

from __future__ import annotations

import ast
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from better_human.io import FormatError, check_format, read_frame_set, write_frame_set
from better_human.io.formats import FRAME_SET, LATEST, MHR, MIGRATIONS, SOMA
from better_human.io.load import load_model_npz
from better_human.io.schema import MHR_SCHEMA, SOMA_SCHEMA

REPO = Path(__file__).resolve().parents[2]
PACKAGE = REPO / "src" / "better_human"
FIXTURES = REPO / "tests" / "support" / "formats"
#: The modules that may name ``format_version``: the one check with its migrations, and the archive schemas.
FORMAT_VERSION_MODULES = ("io/formats.py", "io/schema.py")
READERS: dict[str, tuple[str, Callable[[Path], Any]]] = {
    MHR: ("npz", lambda path: load_model_npz(path, MHR_SCHEMA)),
    SOMA: ("npz", lambda path: load_model_npz(path, SOMA_SCHEMA)),
    FRAME_SET: ("json", read_frame_set),
}
#: Every format; SOMA's write cases run in the full tier: rewriting its archive compresses the schema's fixed sizes
#: (18056 vertices, 40 MB), seconds on this host. Its migration case, a read, stays in the fast tier.
WRITTEN_FORMATS = [pytest.param(name, marks=pytest.mark.slow) if name == SOMA else name for name in LATEST]


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


@pytest.mark.parametrize("format", WRITTEN_FORMATS)
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


@pytest.mark.parametrize("format", WRITTEN_FORMATS)
def test_later_version_is_refused_naming_both(format: str, tmp_path: Path) -> None:
    later = LATEST[format] + 1
    fixture = _fixture(format, LATEST[format])
    path = _rewrite(fixture, tmp_path / fixture.name, {"format_version": later})
    with pytest.raises(FormatError, match=rf"{format} format_version {later} is later than {LATEST[format]}"):
        _read(format, path)


@pytest.mark.parametrize("format", WRITTEN_FORMATS)
def test_another_format_is_refused_naming_both(format: str, tmp_path: Path) -> None:
    fixture = _fixture(format, LATEST[format])
    path = _rewrite(fixture, tmp_path / fixture.name, {"format": "x.y"})
    with pytest.raises(FormatError, match=rf"expected a {format} file, got format 'x.y'"):
        _read(format, path)


def test_another_format_is_refused_before_any_migration(tmp_path: Path) -> None:
    """A v1 archive that names another format is refused, not migrated into an MHR archive."""
    path = _rewrite(_fixture(MHR, 1), tmp_path / "mhr_v1.npz", {"format": SOMA})
    with pytest.raises(FormatError, match=rf"expected a {MHR} file, got format '{SOMA}'"):
        _read(MHR, path)


def test_a_format_without_its_version_is_refused() -> None:
    with pytest.raises(FormatError, match=rf"a {FRAME_SET} file without format_version"):
        check_format({"format": FRAME_SET, "frames": {}}, format=FRAME_SET, version=LATEST[FRAME_SET])


@pytest.mark.parametrize("format", LATEST)
def test_every_format_has_a_spec_page(format: str) -> None:
    page = REPO / "docs" / "reference" / "formats" / f"{format.removeprefix('better_human.')}.md"
    assert page.is_file(), f"{format} has no spec page {page.relative_to(REPO)}"


def _format_version_names(package: Path) -> list[str]:
    """``"<path>, line <n>"`` of every ``"format_version"`` string constant outside ``FORMAT_VERSION_MODULES``.

    Any constant, not only a comparison: a version check can hide in a dict compared whole, a variable or a match.
    """
    found = []
    for path in sorted(package.rglob("*.py")):
        relative = path.relative_to(package).as_posix()
        if relative in FORMAT_VERSION_MODULES:
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Constant) and node.value == "format_version":
                found.append(f"{relative}, line {node.lineno}")
    return found


def test_format_version_is_named_only_in_io_formats_and_schema() -> None:
    offences = _format_version_names(PACKAGE)
    assert not offences, (
        "read and write format_version through better_human.io.formats (upgrade, check_format, header):\n  "
        + "\n  ".join(offences)
    )


def test_a_planted_format_version_is_found(tmp_path: Path) -> None:
    """A comparison, the whole-metadata equality ``families/mhr.py`` had before record 0012, and a match."""
    planted = {
        "io/formats.py": "ok = data['format_version'] < 2\n",
        "io/schema.py": "FIELDS = {'format_version': ()}\n",
        "compare.py": "found = int(data.get('format_version', 0))\nif found > 2:\n    raise ValueError\n",
        "metadata.py": "METADATA = {'format_version': 1, 'up_axis': '+y'}\nif header != METADATA:\n    raise ValueError\n",
        "match.py": "match data['format_version']:\n    case 1:\n        pass\n",
    }
    for name, text in planted.items():
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text(text)
    assert _format_version_names(tmp_path) == ["compare.py, line 1", "match.py, line 1", "metadata.py, line 1"]
