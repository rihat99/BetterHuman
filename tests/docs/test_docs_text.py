"""A docs page says what holds for a reader with no history (principle 7.9).

What changed, and the code of the plan that changed it, belong in the changelog, the migration pages and the
decision records. Every other page of ``docs/`` and the README names no plan code (a work package, lane, ruling,
review finding or phase such as ``H9`` or ``P10-D``) and no history (``no longer``, ``was moved``, ``was added ... on
<date>``).
"""

from __future__ import annotations

import re
from fnmatch import fnmatch
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HISTORY_PAGES = ("docs/reference/changelog.md", "docs/reference/migration-*.md", "docs/contracts/decisions/[0-9]*.md")
HISTORY = re.compile(
    r"\bruling\b|\bwp/[A-Z]|\blane [A-Z]\b|\bwork package\b|\breview (?:finding|[HML]\d)|\b[GHRX]\d+(?:fix)?\b"
    r"|\bP\d+-[A-Z]|(?i:\bno\s+longer\b|\bwas\s+moved\b|\bwas\s+added\b[^.]*?\bon\s+\d{4}-\d{2}-\d{2})"
)


def test_docs_pages_tell_no_history() -> None:
    """No plan code and no "no longer", "was moved" or "was added ... on <date>" outside the history pages."""
    found = []
    for page in [REPO / "README.md", *sorted((REPO / "docs").rglob("*.md"))]:
        relative = page.relative_to(REPO).as_posix()
        if not any(fnmatch(relative, pattern) for pattern in HISTORY_PAGES):
            found += [f"{relative}: {' '.join(match[0].split())}" for match in HISTORY.finditer(page.read_text())]
    assert found == [], "state what holds now instead:\n  " + "\n  ".join(found)
