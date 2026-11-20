"""The docs text says what holds now, and its hand-written tables equal their source (principles 7.7, 7.9).

What changed, and the code of the plan that changed it, belong in the commit messages. No page of ``docs/`` and not
the README names a plan code (a work package, lane, ruling, review finding or phase such as ``H9`` or ``P10-D``) or
tells history (``no longer``, ``was moved``, ``was added ... on <date>``).

Three tables are typed by hand and read here against what they copy: the README's family table against each
family's ``FamilySpec`` (its identity and stream groups; MHR and SOMA need the licensed weights), the weights layout
of ``docs/guides/load-a-family.md`` against the folder and file each loader asks ``io.weight_path`` for, and the
test tables of ``docs/reference/tests.md`` against the listing of ``tests/contract/`` and ``tests/docs/``.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

import better_human as bh
from better_human.families import mhr_family, smpl_family, soma_family
from tests.support.random_archives import random_weight_spec
from tests.support.synthetic import write_family

REPO = Path(__file__).resolve().parents[2]
HISTORY = re.compile(
    r"\bruling\b|\bwp/[A-Z]|\blane [A-Z]\b|\bwork package\b|\breview (?:finding|[HML]\d)|\b[GHRX]\d+(?:fix)?\b"
    r"|\bP\d+-[A-Z]|(?i:\bno\s+longer\b|\bwas\s+moved\b|\bwas\s+added\b[^.]*?\bon\s+\d{4}-\d{2}-\d{2})"
)


def test_docs_pages_tell_no_history() -> None:
    """No plan code and no "no longer", "was moved" or "was added ... on <date>" on any page."""
    found = []
    for page in [REPO / "README.md", *sorted((REPO / "docs").rglob("*.md"))]:
        relative = page.relative_to(REPO).as_posix()
        found += [f"{relative}: {' '.join(match[0].split())}" for match in HISTORY.finditer(page.read_text())]
    assert found == [], "state what holds now instead:\n  " + "\n  ".join(found)


def _table(text: str, header: str) -> list[list[str]]:
    """The body rows of the Markdown table whose header row starts with ``header``, as stripped cells."""
    lines = text.splitlines()
    start = next(index for index, line in enumerate(lines) if line.startswith(header))
    rows = []
    for line in lines[start + 2 :]:
        if not line.startswith("|"):
            break
        rows.append([cell.strip() for cell in line.strip().strip("|").split("|")])
    return rows


def _ticked(cell: str) -> list[str]:
    return re.findall(r"`([^`]+)`", cell)


#: How to build each family of the README's table without the user's weights; MHR and SOMA read them.
README_FAMILIES: dict[str, Callable[[Path], bh.FamilySpec]] = {
    "SMPL": lambda _: random_weight_spec("smpl"),
    "SMPL-H": lambda _: random_weight_spec("smplh"),
    "SMPL-X": lambda _: random_weight_spec("smplx"),
    "MANO": lambda folder: bh.families.mano(write_family(folder, "mano")),
    "FLAME": lambda folder: bh.families.flame(write_family(folder, "flame")),
    "MHR": lambda _: bh.families.mhr(),
    "SOMA": lambda _: bh.families.soma(),
}


def test_the_readme_family_table_lists_every_family() -> None:
    families = [row[0] for row in _table((REPO / "README.md").read_text(), "| family | loader |")]
    assert families == list(README_FAMILIES)


@pytest.mark.parametrize(
    "family",
    [pytest.param(name, marks=pytest.mark.licensed) if name in ("MHR", "SOMA") else name for name in README_FAMILIES],
)
def test_the_readme_family_table_gives_each_familys_groups(family: str, tmp_path: Path) -> None:
    """The identity and stream groups of a row are those of the spec the family's default loader gives."""
    rows = {row[0]: row for row in _table((REPO / "README.md").read_text(), "| family | loader |")}
    spec = README_FAMILIES[family](tmp_path)
    assert (_ticked(rows[family][2]), _ticked(rows[family][3])) == (
        list(spec.identity_groups),
        list(spec.stream_groups),
    )


class _Asked(Exception):
    """Raised by the stand-in ``weight_path``: the folder and file a loader asked for."""


#: Each loader and the options that choose its file: every file of the layout table is one of these calls.
LAYOUT_LOADERS: dict[str, tuple[Callable[..., Any], list[dict[str, Any]]]] = {
    "SMPL": (bh.families.smpl, [{"gender": gender} for gender in ("neutral", "male", "female")]),
    "SMPL-H": (bh.families.smplh, [{"gender": gender} for gender in ("neutral", "male", "female")]),
    "SMPL-X": (bh.families.smplx, [{"gender": gender} for gender in ("neutral", "male", "female")]),
    "MANO": (bh.families.mano, [{"side": side} for side in ("right", "left")]),
    "FLAME": (bh.families.flame, [{"gender": gender} for gender in ("generic", "male", "female")]),
    "MHR": (bh.families.mhr, [{"lod": lod} for lod in range(7)]),
    "SOMA": (bh.families.soma, [{}]),
}


def _expand(pattern: str) -> list[str]:
    """``a{x,y}b`` -> ``axb``, ``ayb``; ``a{0..2}`` -> ``a0``, ``a1``, ``a2``."""
    match = re.search(r"\{([^}]*)\}", pattern)
    if match is None:
        return [pattern]
    first, _, last = match[1].partition("..")
    choices = [str(value) for value in range(int(first), int(last) + 1)] if last else match[1].split(",")
    return [path for choice in choices for path in _expand(pattern[: match.start()] + choice + pattern[match.end() :])]


def test_the_weights_layout_is_where_each_loader_looks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each row's folder and files are the ``io.weight_path(folder, file)`` calls of its loader, over its options."""

    def asked(folder: str, filename: str, *, path: object = None) -> Path:
        raise _Asked(folder, filename)

    for module in (smpl_family, mhr_family, soma_family):
        monkeypatch.setattr(module, "weight_path", asked)
    rows = _table((REPO / "docs" / "guides" / "load-a-family.md").read_text(), "| Family | Folder | File |")
    assert [row[0] for row in rows] == list(LAYOUT_LOADERS)
    for family, folder, files in rows:
        loader, options = LAYOUT_LOADERS[family]
        calls = set()
        for option in options:
            with pytest.raises(_Asked) as call:
                loader(**option)
            calls.add(call.value.args)
        listed = {(_ticked(folder)[0], path) for pattern in _ticked(files) for path in _expand(pattern)}
        assert listed == calls, family


@pytest.mark.parametrize(("heading", "folder"), [("## Contract tests", "contract"), ("## Docs tests", "docs")])
def test_the_test_tables_list_every_test_module(heading: str, folder: str) -> None:
    """``docs/reference/tests.md`` has one row per test module of ``tests/contract/`` and ``tests/docs/``."""
    text = (REPO / "docs" / "reference" / "tests.md").read_text()
    section = text[text.index(heading) :]
    rows = [_ticked(row[0])[0] for row in _table(section, "| Test | Guards |")]
    modules = sorted(path.name for path in (REPO / "tests" / folder).glob("test_*.py"))
    assert sorted(rows) == modules and len(rows) == len(set(rows))
