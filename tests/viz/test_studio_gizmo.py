"""GizmoController on the fake viewer server: options per joint kind, drags, status and lifecycle."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from better_robot.viewer import Viewer

import better_human as bh
from better_human.viz.studio.gizmo import GizmoController
from better_human.viz.studio.state import StudioSession

from .fake_server import FakeServer

MODELS = Path(__file__).parents[2] / "models"
SMPL_PATH = MODELS / "smpl" / "SMPL_NEUTRAL.npz"
MHR_PATH = MODELS / "MHR" / "converted" / "mhr_lod1.npz"
_TURN = [0.0, 0.3827, 0.0, 0.9239]


@pytest.fixture(scope="module")
def smpl() -> bh.Body:
    if not SMPL_PATH.is_file():
        pytest.skip(f"licensed SMPL fixture is absent: {SMPL_PATH}")
    return bh.load("smpl", SMPL_PATH)


@pytest.fixture(scope="module")
def mhr() -> bh.Body:
    if not MHR_PATH.is_file():
        pytest.skip(f"licensed MHR fixture is absent: {MHR_PATH}")
    return bh.load("mhr", MHR_PATH, lod=1)


def _setup(body: bh.Body, key: str) -> tuple[StudioSession, GizmoController, list[str]]:
    session = StudioSession(body, key)
    status: list[str] = []
    return session, GizmoController(Viewer(server=FakeServer()), session, on_status=status.append), status


def _angle(quat_a: np.ndarray, quat_b: np.ndarray) -> float:
    return float(2 * np.arccos(min(abs(float(np.dot(quat_a, quat_b))), 1.0)))


def test_non_root_is_rotation_only(smpl: bh.Body) -> None:
    session, gizmo, status = _setup(smpl, "smpl")
    gizmo.select(18)
    props = gizmo.gizmo.handle.props
    assert props["disable_sliders"] is True and props["disable_axes"] is True
    assert status[-1] == f"posing {smpl.joint_names[18]!r} (spherical)"
    np.testing.assert_allclose(gizmo.gizmo.pose, session.joint_poses[18].numpy(), atol=1e-6)


def test_root_is_six_dof(smpl: bh.Body) -> None:
    _, gizmo, status = _setup(smpl, "smpl")
    gizmo.select(0)
    props = gizmo.gizmo.handle.props
    assert "disable_sliders" not in props and "disable_axes" not in props
    assert status[-1].endswith("(free_flyer)")


def test_drag_non_root_rotates_only_its_slot(smpl: bh.Body) -> None:
    session, gizmo, _ = _setup(smpl, "smpl")
    gizmo.select(18)
    handle = gizmo.handle_for(18)
    q_old = session.q.clone()
    pose = session.joint_poses[18].numpy().copy()
    pose[3:] = _TURN
    gizmo._on_drag(handle, pose)
    changed = set(torch.nonzero(session.q != q_old).flatten().tolist())
    assert changed and changed <= set(range(handle.idx_q, handle.idx_q + 4))
    assert _angle(session.joint_poses[18, 3:].numpy(), np.array(_TURN)) < 1e-4


def test_drag_root_translates_and_follows(smpl: bh.Body) -> None:
    session, gizmo, _ = _setup(smpl, "smpl")
    gizmo.select(0)
    pose = session.joint_poses[0].numpy().copy()
    pose[:3] += [0.1, -0.2, 0.3]
    gizmo._on_drag(gizmo.handle_for(0), pose)
    np.testing.assert_allclose(session.joint_poses[0, :3].numpy(), pose[:3], atol=1e-5)
    np.testing.assert_allclose(gizmo.gizmo.pose[:3], pose[:3], atol=1e-5)


def test_clear_on_body_change_and_select_none(smpl: bh.Body) -> None:
    session, gizmo, _ = _setup(smpl, "smpl")
    gizmo.select(5)
    assert gizmo.gizmo is not None and gizmo.joint == 5
    session.load_body(smpl, "smpl")
    assert gizmo.gizmo is None
    gizmo.select(5)
    gizmo.select(None)
    assert gizmo.gizmo is None and gizmo.joint is None


def test_fixed_joint_reports_status(mhr: bh.Body) -> None:
    session, gizmo, status = _setup(mhr, "mhr")
    posed = {h.native_index for h in session.handles}
    fixed = next(j for j in range(len(mhr.joint_names)) if j not in posed)
    assert gizmo.handle_for(fixed) is None
    gizmo.select(fixed)
    assert gizmo.gizmo is None
    assert status[-1] == f"joint {mhr.joint_names[fixed]!r} has no degree of freedom"


def test_mhr_channel_drag(mhr: bh.Body) -> None:
    session, gizmo, _ = _setup(mhr, "mhr")
    handle = next(h for h in session.handles if h.kind == "channels" and len(h.channels) == 3)
    gizmo.select(handle.native_index)
    assert gizmo.gizmo.handle.props["disable_sliders"] is True
    q_old = session.q.clone()
    pose = session.joint_poses[handle.native_index].numpy().copy()
    pose[3:] = _TURN
    gizmo._on_drag(handle, pose)
    changed = set(torch.nonzero(session.q != q_old).flatten().tolist())
    assert changed and changed <= {idx for _, idx in handle.channels}
