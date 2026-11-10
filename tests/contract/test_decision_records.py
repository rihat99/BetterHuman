"""Every decision record follows the template, carries an allowed status and an index row, and keeps its text (8.2).

BetterRobot's test of the same name, on BetterHuman's records. A record is a file ``NNNN-kebab-title.md`` in the
decisions folder; the numbers run from 0001 without a gap. Its Status line is, in full,
``**Status:** WORD[ (WHO)][; amended by NNNN, ...][; amendment proposed in NNNN, ...]`` then ``.`` or `` ·`` and the
``**Date:**``: WORD is ``Proposed``, ``Accepted`` or ``Superseded by NNNN``, WHO is ``owner``, ``integrator, awaiting
owner`` or ``integrator ruling R-n, YYYY-MM-DD``, and every number it points at is a later record: one that amends is
not Proposed, one that proposes an amendment is. Accepted is the owner's word (principle 8.2): a record whose status
was owner-accepted or superseded in any commit (no ``integrator`` in it) never returns to Proposed and keeps the text
of that first commit, Status line aside. A record accepted now keeps the text it had when its current acceptance began
(the first commit after its last Proposed one), Status line aside: a change is a new record. Every BetterHuman record
follows the template, so none is exempt; the resolver test (``tests/contract/test_citations.py``) checks the paths
every record cites.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DECISIONS = REPO / "docs" / "contracts" / "decisions"
INDEX = DECISIONS / "index.md"
TEMPLATE = (
    "Context",
    "Decision",
    "Example",
    "Consequences",
    "Alternatives",
    "Compatibility",
    "Enforcement",
    "Open questions",
)
STATUS = re.compile(
    r"^\*\*Status:\*\* (?P<word>Proposed|Accepted|Superseded by (?P<by>\d{4}))"
    r"(?: \((?:owner|integrator, awaiting owner|integrator ruling R-\d+, \d{4}-\d{2}-\d{2})\))?"
    r"(?:; amended by (?P<amended>\d{4}(?:, \d{4})*))?"
    r"(?:; amendment proposed in (?P<proposed>\d{4}(?:, \d{4})*))?"
    r"(?:\.| ·) \*\*Date:\*\* \d{4}-\d{2}-\d{2}(?: \(recorded; decision taken earlier\))?$"
)


def _records() -> dict[str, Path]:
    return {path.name[:4]: path for path in sorted(DECISIONS.glob("[0-9][0-9][0-9][0-9]-*.md"))}


def _status_line(text: str) -> str:
    return next(line for line in text.splitlines() if line.startswith("**Status:**"))


def test_record_files_are_numbered_without_a_gap_and_titled_by_their_number() -> None:
    """File names are ``NNNN-kebab-title.md`` from 0001 on, and each title line starts with the file's number."""
    records = _records()
    assert list(records) == [f"{number:04d}" for number in range(1, len(records) + 1)]
    for number, path in records.items():
        assert re.fullmatch(r"\d{4}-[a-z0-9]+(?:-[a-z0-9]+)*\.md", path.name), path.name
        assert path.read_text().startswith(f"# {number} "), f"{path.name}: the title line is '# {number} <Title>'"


def test_every_status_is_allowed_and_points_at_later_records() -> None:
    """The status is Proposed, Accepted (optionally amended by later records) or Superseded by a later record; an
    amendment still Proposed is named as proposed, not as made."""
    records = _records()
    for number, path in records.items():
        match = STATUS.match(_status_line(path.read_text()))
        assert match, f"{path.name}: status must be Proposed, Accepted[; amended by NNNN] or Superseded by NNNN"
        made = match["amended"].split(", ") if match["amended"] else []
        proposed = match["proposed"].split(", ") if match["proposed"] else []
        for target in ([match["by"]] if match["by"] else []) + made + proposed:
            assert target in records and target > number, f"{path.name}: {target} is not a later record"
        for target in made + proposed:
            is_proposed = _proposed(records[target].read_text())
            assert is_proposed == (target in proposed), (
                f"{path.name}: {target} is {'' if is_proposed else 'not '}Proposed"
            )


def test_every_record_has_exactly_the_template_sections() -> None:
    """Every record has the eight template sections in order, and an accepted one has no open question."""
    for path in _records().values():
        text = path.read_text()
        assert tuple(re.findall(r"^## (.+)$", text, re.M)) == TEMPLATE, f"{path.name}: sections must be {TEMPLATE}"
        open_questions = text.split("\n## Open questions\n", 1)[1].strip()
        if not _status_line(text).startswith("**Status:** Proposed"):
            assert open_questions in ("", "None."), f"{path.name}: an accepted record has no open question"


def test_the_index_lists_every_record_with_its_status() -> None:
    """``index.md`` has one row per record, linking the file and naming the record's status word."""
    index = INDEX.read_text()
    for number, path in _records().items():
        row = re.search(rf"^\| \[{number}\]\({re.escape(path.name)}\) \|.*\| (?P<status>[^|]+) \|$", index, re.M)
        assert row, f"index.md has no row for {path.name}"
        word = STATUS.match(_status_line(path.read_text()))["word"]
        assert row["status"].startswith(word.split(" by ")[0]), f"index.md: {path.name} is {word}"


def _proposed(text: str) -> bool:
    return _status_line(text).startswith("**Status:** Proposed")


def _owner_decided(text: str) -> bool:
    """Accepted or superseded with no integrator in the Status line: the owner's word (principle 8.2)."""
    return not _proposed(text) and "integrator" not in _status_line(text)


def _history(path: Path) -> list[str]:
    """The record's committed texts, oldest first, then its text now."""
    relative = path.relative_to(REPO).as_posix()
    log = subprocess.run(
        ["git", "log", "--reverse", "--format=%H", "--", relative], cwd=REPO, capture_output=True, text=True, check=True
    )
    texts = []
    for commit in log.stdout.split():
        shown = subprocess.run(
            ["git", "show", f"{commit}:{relative}"], cwd=REPO, capture_output=True, text=True, check=False
        )
        if shown.returncode == 0:
            texts.append(shown.stdout)
    return [*texts, path.read_text()]


def _body(text: str) -> str:
    return text.replace(_status_line(text), "")


def test_an_accepted_record_changes_only_in_its_status_line() -> None:
    """A record accepted now keeps the text its current acceptance began with, Status line aside: amend or supersede
    with a new record."""
    changed = []
    for path in _records().values():
        accepted = None
        for text in _history(path):
            accepted = None if _proposed(text) else accepted or text
        if accepted is not None and _body(path.read_text()) != _body(accepted):
            changed.append(path.name)
    assert not changed, "write a new record that amends these instead of editing them:\n  " + "\n  ".join(changed)


def test_an_owner_accepted_record_never_returns_to_proposed_or_changes() -> None:
    """Once the owner accepted or superseded a record, it is never Proposed again and keeps that first text."""
    reopened = []
    for path in _records().values():
        history = _history(path)
        decided = next((text for text in history if _owner_decided(text)), None)
        if decided is None:
            continue
        if _proposed(history[-1]) or _body(history[-1]) != _body(decided):
            reopened.append(path.name)
    assert not reopened, "the owner accepted these; write a new record instead:\n  " + "\n  ".join(reopened)
