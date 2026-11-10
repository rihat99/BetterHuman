"""The public API equals its snapshot, ``tests/contract/public_api.json`` (BetterRobot decision record 0024).

Every path of the ``__all__`` tree (``better_robot.testing.public_api``; the root lists every subpackage) with its
kind and signature is tracked, so a change to the public surface is a reviewed edit of the snapshot, never a
side effect. After an intended change: ``uv run python tools/api_snapshot.py --check`` lists it, ``--write``
records it, and the changelog says it.
"""

from __future__ import annotations

import json
from pathlib import Path

from better_robot.testing import public_api

SNAPSHOT = Path(__file__).resolve().parent / "public_api.json"


def test_public_api_matches_the_snapshot() -> None:
    """A path added, removed or with a changed signature fails until the snapshot records it."""
    snapshot: dict[str, str] = json.loads(SNAPSHOT.read_text())
    current = public_api("better_human")
    changed = sorted(path for path in snapshot.keys() | current.keys() if snapshot.get(path) != current.get(path))
    assert not changed, (
        "the public API differs from tests/contract/public_api.json at these paths; review them with "
        "`uv run python tools/api_snapshot.py --check`, then record them with `--write`:\n  " + "\n  ".join(changed)
    )
