"""Every repository path cited in tracked text resolves, and no line number is cited (principle 7.8).

Citations are found by ``better_robot.testing.dead_citations`` (the shared testing module): a ``src/``, ``tests/``,
``docs/``, ``tools/``, ``examples/`` or ``benchmarks/`` path must be tracked, a ``path::symbol`` must name a symbol
the file defines, and a ``.py:<line>`` citation is always dead. The dead citations that exist today sit in
``tests/contract/ratchets/dead_citations.json`` (``"<file>: <citation>"`` to the reason); the ratchet only shrinks,
and the fix that resolves a citation removes its entry. Tracked text states a reason, never the code of a plan, a
review or its finding (``PLAN_CODE``).
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from better_robot.testing import dead_citations

REPO = Path(__file__).resolve().parents[2]
THIS = Path(__file__).resolve().relative_to(REPO).as_posix()
PLAN_CODE = re.compile(
    r"\bwp/[A-Z]|\blane [A-Z]\b|\bR-\d+\b|\(P-\d+\)|\bP\d+-D\d+\b|\bPhase \d+ pass|\bowning WP\b|\bwork package\b"
    r"|\bruling R-|\breview finding\b|\b[A-Z]\d{1,2}(?:fix)? (?:review|finding)\b"
)
TEXT_SUFFIXES = (".md", ".py", ".toml", ".yml", ".yaml", ".json", ".sh", ".txt")
CITATION_RATCHET: dict[str, str] = json.loads((Path(__file__).parent / "ratchets" / "dead_citations.json").read_text())


def test_cited_paths_resolve() -> None:
    """A new dead citation fails: cite a tracked path or ``path::symbol``, never a line number."""
    new = sorted(set(dead_citations(REPO)) - CITATION_RATCHET.keys())
    assert not new, "cite a tracked path or path::symbol, never a line number:\n  " + "\n  ".join(new)


def test_citation_ratchet_has_no_stale_entries() -> None:
    """A citation that resolves now leaves ``tests/contract/ratchets/dead_citations.json`` in the commit that fixes
    it."""
    stale = sorted(CITATION_RATCHET.keys() - set(dead_citations(REPO)))
    assert not stale, "remove these entries from tests/contract/ratchets/dead_citations.json:\n  " + "\n  ".join(stale)


def test_no_plan_code_in_tracked_text() -> None:
    """Tracked text states a reason, never a plan code such as ``wp/H9``, ``H3 review L7``, ``ruling R-3`` or a
    review finding; this module and the docs-text lint are not read, since their patterns spell the codes."""
    listing = subprocess.run(["git", "ls-files", "-z"], cwd=REPO, capture_output=True, text=True, check=True).stdout
    found = [
        f"{path}: {match[0]}"
        for path in listing.split("\0")
        if path.endswith(TEXT_SUFFIXES) and path not in (THIS, "tests/docs/test_docs_text.py")
        for match in PLAN_CODE.finditer((REPO / path).read_text())
    ]
    assert not found, "write the reason instead of the plan code:\n  " + "\n  ".join(found)
