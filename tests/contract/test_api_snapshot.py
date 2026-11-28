"""The public API equals its snapshot, ``tests/contract/public_api.json``.

Every path of the ``__all__`` tree (``better_robot.testing.public_api``; the root lists every subpackage) with its
kind and signature is tracked, so a change to the public surface is a reviewed edit of the snapshot, never a
side effect. After an intended change:
``uv run python -m better_robot.testing.api_snapshot better_human tests/contract/public_api.json --check`` lists it,
``--write`` records it, and the commit message says it.
"""

from __future__ import annotations

from pathlib import Path

from better_robot.testing import snapshot_differences

SNAPSHOT = Path(__file__).resolve().parent / "public_api.json"


def test_public_api_matches_the_snapshot() -> None:
    """A path added, removed or with a changed signature fails until the snapshot records it."""
    changed = snapshot_differences("better_human", SNAPSHOT)
    assert not changed, (
        "the public API differs from tests/contract/public_api.json; review it with `uv run python -m "
        "better_robot.testing.api_snapshot better_human tests/contract/public_api.json --check`, then record it with "
        "`--write`:\n  " + "\n  ".join(changed)
    )
