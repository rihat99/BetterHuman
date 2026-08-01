"""Frame parent-joint resolution and notification in the studio session."""

from __future__ import annotations

from pathlib import Path

import pytest

import better_human as bh
from better_human.viz.studio.state import FrameEntry, StudioSession

MODELS_DIR = Path(__file__).parents[2] / "models"
SMPL_PATH = MODELS_DIR / "smpl" / "SMPL_NEUTRAL.npz"


@pytest.fixture(scope="module")
def session() -> StudioSession:
    if not SMPL_PATH.is_file():
        pytest.skip(f"licensed SMPL fixture is absent: {SMPL_PATH}")
    body = bh.SMPL(str(SMPL_PATH), compute_mass=False)
    return StudioSession(body, "smpl_neutral")


def test_resolved_joint_argmax_default(session: StudioSession) -> None:
    vertex = 331
    entry = FrameEntry(vertex, None)
    assert session.resolved_joint(entry) == int(session.vertex_native_joints[vertex])


def test_resolved_joint_override(session: StudioSession) -> None:
    assert session.resolved_joint(FrameEntry(331, 7)) == 7


def test_set_frame_joint_updates_entry_and_notifies(session: StudioSession) -> None:
    kinds: list[str] = []
    session.subscribe(kinds.append)
    session.set_frame_set({"nose": FrameEntry(331, None)})
    kinds.clear()
    session.set_frame_joint("nose", 5)
    assert session.frame_set["nose"] == FrameEntry(331, 5)
    assert kinds == ["frames"]
    # Clearing back to argmax.
    session.set_frame_joint("nose", None)
    assert session.frame_set["nose"].joint is None


def test_set_frame_joint_ignores_unknown_frame(session: StudioSession) -> None:
    session.set_frame_set({"nose": FrameEntry(331, None)})
    kinds: list[str] = []
    session.subscribe(kinds.append)
    session.set_frame_joint("does_not_exist", 3)
    assert kinds == []
    assert "does_not_exist" not in session.frame_set
