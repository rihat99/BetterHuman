"""Every BetterHuman file format names itself and its version, and old files keep loading (records 0012, 0013).

For each format in ``io.formats.LATEST``: the latest version round-trips, a fixture of every past version in
``tests/support/formats/`` reads through its migration and equals the latest fixture, a later version and another
format are refused naming both, and ``docs/reference/formats/<name>.md`` specifies it. No module but
``src/better_human/io/formats.py`` and ``src/better_human/io/schema.py`` names ``format_version`` (8.8), and none loads a pickle but the listed exception (record 0013).
The fixtures are synthetic and frozen: ``tests/support/formats/make_fixtures.py`` wrote them (every field a distinct
constant; the v1 archives as the converters wrote them before record 0012) and ``SHA256SUMS`` there lists them. Each
spec page's field tables equal the reader's schema, field by field.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from better_human.io import FormatError, check_format, read_frame_set, write_frame_set
from better_human.io.formats import (
    DENSITY_TABLES,
    FRAME_SET,
    LATEST,
    MHR,
    MIGRATIONS,
    SEGMENTATION,
    SOMA,
    header,
    upgrade,
)
from better_human.io.load import load_model_npz
from better_human.io.schema import DENSITY_TABLES_SCHEMA, MHR_SCHEMA, SEGMENTATION_SCHEMA, SOMA_SCHEMA, ModelSchema

REPO = Path(__file__).resolve().parents[2]
PACKAGE = REPO / "src" / "better_human"
FIXTURES = REPO / "tests" / "support" / "formats"
CONVERTERS = REPO / "tools" / "convert"
#: The modules that may name ``format_version``: the one check with its migrations, and the archive schemas.
FORMAT_VERSION_MODULES = ("io/formats.py", "io/schema.py")
#: ``path -> reason`` for each module that still loads a pickle; an entry leaves with its fix (record 0013).
ALLOW_PICKLE_EXCEPTIONS: dict[str, str] = {}
READERS: dict[str, tuple[str, Callable[[Path], Any]]] = {
    MHR: ("npz", lambda path: load_model_npz(path, MHR_SCHEMA)),
    SOMA: ("npz", lambda path: load_model_npz(path, SOMA_SCHEMA)),
    FRAME_SET: ("json", read_frame_set),
    SEGMENTATION: ("npz", lambda path: load_model_npz(path, SEGMENTATION_SCHEMA)),
    DENSITY_TABLES: ("npz", lambda path: load_model_npz(path, DENSITY_TABLES_SCHEMA)),
}
#: The provenance a v1 archive migrates to: an MHR v1 archive has none, a SOMA v1 archive carries it as ``source``.
V1_PROVENANCE = {MHR: "unknown (v1)", SOMA: '{"package": "py-soma-x"}'}
#: Every format; SOMA's write cases run in the full tier: rewriting its archive compresses the schema's fixed sizes
#: (18056 vertices, 40 MB), seconds on this host.
WRITTEN_FORMATS = [pytest.param(name, marks=pytest.mark.slow) if name == SOMA else name for name in LATEST]
#: Every migration; SOMA's is full tier: its 40 MB fixtures stall on hosts with transparent huge pages (fast: ~1 s).
MIGRATION_CASES = [
    pytest.param(name, version, id=f"{name}_v{version}", marks=pytest.mark.slow if name == SOMA else ())
    for name, version in sorted(MIGRATIONS)
]


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


@pytest.mark.parametrize(("format", "version"), MIGRATION_CASES)
def test_past_version_reads_through_migration_as_the_latest(format: str, version: int) -> None:
    past = _comparable(format, _read(format, _fixture(format, version)))
    latest = _comparable(format, _read(format, _fixture(format, LATEST[format])))
    if format != FRAME_SET:
        # A v1 archive's provenance is its own; everything else is the v2 archive's.
        assert past.pop("provenance")[2] == np.asarray(V1_PROVENANCE[format]).tobytes()
        latest.pop("provenance")
    assert past == latest


@pytest.mark.parametrize("format", WRITTEN_FORMATS)
def test_an_added_field_is_ignored(format: str, tmp_path: Path) -> None:
    """A file may carry a field its reader does not know (Law 7): the reader returns what it returns without it."""
    fixture = _fixture(format, LATEST[format])
    added = np.asarray([1.0, 2.0]) if READERS[format][0] == "npz" else {"note": "written by a later tool"}
    path = _rewrite(fixture, tmp_path / fixture.name, {"added_field": added})
    assert _comparable(format, _read(format, path)) == _comparable(format, _read(format, fixture))


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


@pytest.mark.parametrize("version", ["1", 1.0, None])
def test_a_version_that_is_not_an_integer_is_refused(version: object) -> None:
    with pytest.raises(FormatError, match=rf"format_version {version!r} is not an integer\. Write the file"):
        check_format({"format": FRAME_SET, "format_version": version}, format=FRAME_SET, version=LATEST[FRAME_SET])


def test_a_past_version_without_the_field_its_migration_reads_is_refused() -> None:
    v1 = {"format": SOMA, "format_version": np.int64(1)}
    with pytest.raises(FormatError, match=r"v1\.npz: a better_human\.soma format_version 1 file without .*'source'"):
        upgrade(v1, format=SOMA, source="v1.npz")


def _spec_page(format: str) -> Path:
    return REPO / "docs" / "reference" / "formats" / f"{format.removeprefix('better_human.')}.md"


@pytest.mark.parametrize("format", LATEST)
def test_every_format_has_a_spec_page(format: str) -> None:
    page = _spec_page(format)
    assert page.is_file(), f"{format} has no spec page {page.relative_to(REPO)}"


def _spec_fields(page: Path) -> dict[str, tuple[str, str]]:
    """``name -> (dtype, shape)`` from the page's ``| field | dtype | shape | meaning |`` tables.

    A row may list several fields, with one dtype and one shape each or one for all of them.
    """
    fields = {}
    in_table = False
    for line in page.read_text().splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if cells == ["field", "dtype", "shape", "meaning"]:
            in_table = True
            continue
        in_table = in_table and line.startswith("|")
        if not in_table or set(cells[0]) == {"-"}:
            continue
        names = re.findall(r"`([^`]+)`", cells[0])
        dtypes = [dtype.strip().strip("`") for dtype in cells[1].split(",")]
        shapes = [shape.replace(" ", "") for shape in re.findall(r"`([^`]+)`", cells[2])]
        for index, name in enumerate(names):
            fields[name] = (dtypes[index if len(dtypes) > 1 else 0], shapes[index if len(shapes) > 1 else 0])
    return fields


SCHEMAS = [MHR_SCHEMA, SOMA_SCHEMA, SEGMENTATION_SCHEMA, DENSITY_TABLES_SCHEMA]


@pytest.mark.parametrize("schema", SCHEMAS, ids=[schema.format for schema in SCHEMAS])
def test_spec_page_fields_are_the_schema(schema: ModelSchema) -> None:
    """Every field the reader reads has a row with its name, dtype and shape, and no row names another field."""
    expected = {
        name: (
            spec.dtype,
            "(" + ",".join(str(axis) for axis in spec.shape) + ("," if len(spec.shape) == 1 else "") + ")",
        )
        for name, spec in schema.fields.items()
    }
    assert _spec_fields(_spec_page(schema.format)) == expected


def test_fixtures_are_the_listed_files() -> None:
    """The fixtures are frozen: ``SHA256SUMS`` (written with them by ``make_fixtures.py``) lists each one's hash."""
    listed = dict(reversed(line.split("  ")) for line in (FIXTURES / "SHA256SUMS").read_text().splitlines())
    on_disk = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in FIXTURES.iterdir()
        if path.suffix in {".npz", ".json"}
    }
    assert on_disk == listed


def _nodes(package: Path) -> Iterator[tuple[str, ast.AST]]:
    """``(path relative to package, node)`` for every syntax node of every module under ``package``."""
    for path in sorted(package.rglob("*.py")):
        relative = path.relative_to(package).as_posix()
        for node in ast.walk(ast.parse(path.read_text())):
            yield relative, node


def _format_version_names(package: Path) -> list[str]:
    """``"<path>, line <n>"`` of every ``"format_version"`` string constant outside ``FORMAT_VERSION_MODULES``.

    Any constant, not only a comparison: a version check can hide in a dict compared whole, a variable or a match.
    """
    return [
        f"{path}, line {node.lineno}"
        for path, node in _nodes(package)
        if path not in FORMAT_VERSION_MODULES and isinstance(node, ast.Constant) and node.value == "format_version"
    ]


def _allow_pickle_arguments(package: Path) -> list[str]:
    """``"<path>, line <n>"`` of every ``allow_pickle=`` argument that is not the constant ``False``."""
    return [
        f"{path}, line {node.value.lineno}"
        for path, node in _nodes(package)
        if isinstance(node, ast.keyword)
        and node.arg == "allow_pickle"
        and not (isinstance(node.value, ast.Constant) and node.value.value is False)
    ]


def _converter_headers() -> dict[str, dict[str, Any]]:
    """``path -> {format, format_version}`` of each dict literal under ``tools/convert`` that holds both header keys.

    A value is a literal or ``np.asarray(<literal>, ...)``; anything else fails ``ast.literal_eval``.
    """
    found = {}
    for path in sorted(CONVERTERS.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.Dict):
                continue
            fields = {
                key.value: ast.literal_eval(value.args[0] if isinstance(value, ast.Call) else value)
                for key, value in zip(node.keys, node.values)
                if isinstance(key, ast.Constant) and key.value in ("format", "format_version")
            }
            if len(fields) == 2:
                found[str(path.relative_to(REPO))] = fields
    return found


def test_each_converter_writes_the_header_of_its_format() -> None:
    """The converters run without BetterHuman installed, so each spells the header out; it equals ``header(format)``."""
    found = _converter_headers()
    assert {fields["format"] for fields in found.values()} == {MHR, SOMA}, found
    for path, fields in found.items():
        assert fields == header(fields["format"]), f"{path} writes {fields}; header() gives {header(fields['format'])}"


def test_format_version_is_named_only_in_io_formats_and_schema() -> None:
    offences = _format_version_names(PACKAGE)
    assert not offences, (
        "read and write format_version through better_human.io.formats (upgrade, check_format, header):\n  "
        + "\n  ".join(offences)
    )


def test_a_planted_format_version_is_found(tmp_path: Path) -> None:
    """A comparison, the whole-metadata equality the MHR loader (``families.mhr``) had before record 0012, and a match."""
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


def test_no_module_loads_a_pickle() -> None:
    """Record 0013: no shipped file executes pickle; ``ALLOW_PICKLE_EXCEPTIONS`` names the modules that still do."""
    found = _allow_pickle_arguments(PACKAGE)
    offences = [hit for hit in found if hit.split(",")[0] not in ALLOW_PICKLE_EXCEPTIONS]
    stale = sorted(ALLOW_PICKLE_EXCEPTIONS.keys() - {hit.split(",")[0] for hit in found})
    assert not offences, "load without allow_pickle=True (decision record 0013):\n  " + "\n  ".join(offences)
    assert not stale, f"these modules no longer load a pickle; remove them from ALLOW_PICKLE_EXCEPTIONS: {stale}"


def test_a_planted_allow_pickle_is_found(tmp_path: Path) -> None:
    (tmp_path / "reader.py").write_text(
        "np.load(path, allow_pickle=True)\nnp.load(path, allow_pickle=False)\nnp.load(path)\nnp.load(path, allow_pickle=ok)\n"
    )
    assert _allow_pickle_arguments(tmp_path) == ["reader.py, line 1", "reader.py, line 4"]
