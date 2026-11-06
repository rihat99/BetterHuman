"""Every repository path cited in tracked text resolves, and no line number is cited (principle 7.8).

Citations are found by ``better_robot.testing.dead_citations`` (BetterRobot decision record 0035, the shared testing
module): a ``src/``, ``tests/``, ``docs/``, ``tools/``, ``examples/`` or ``benchmarks/`` path must be tracked, a
``path::symbol`` must name a symbol the file defines, and a ``.py:<line>`` citation is always dead. The dead
citations that exist today sit in ``ratchets/citations.json`` (``"<file>: <citation>"`` to the reason); the ratchet
only shrinks, and the fix that resolves a citation removes its entry.
"""

from __future__ import annotations

import json
from pathlib import Path

from better_robot.testing import dead_citations

REPO = Path(__file__).resolve().parents[2]
CITATION_RATCHET: dict[str, str] = json.loads((Path(__file__).parent / "ratchets" / "citations.json").read_text())


def test_cited_paths_resolve() -> None:
    """A new dead citation fails: cite a tracked path or ``path::symbol``, never a line number."""
    new = sorted(set(dead_citations(REPO)) - CITATION_RATCHET.keys())
    assert not new, "cite a tracked path or path::symbol, never a line number:\n  " + "\n  ".join(new)


def test_citation_ratchet_has_no_stale_entries() -> None:
    """A citation that resolves now leaves ``ratchets/citations.json`` in the commit that fixes it."""
    stale = sorted(CITATION_RATCHET.keys() - set(dead_citations(REPO)))
    assert not stale, "remove these entries from tests/contract/ratchets/citations.json:\n  " + "\n  ".join(stale)
