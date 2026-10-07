"""StudioSession, posing, gizmo and layers on the new core and the fake viewer server."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from better_robot.viewer import Viewer

import better_human as bh
from better_human.viz.studio.gizmo import GizmoController
from better_human.viz.studio.layers import StudioLayers
from better_human.viz.studio.state import FrameEntry, StudioSession

from .fake_server import FakeServer

MODEL_PATH = Path(__file__).parents[2] / "models" / "smpl" / "SMPL_NEUTRAL.npz"


@pytest.fixture(scope="module")
def body() -> bh.Body:
    if not MODEL_PATH.is_file():
        pytest.skip(f"licensed SMPL fixture is absent: {MODEL_PATH}")
    return bh.load("smpl", MODEL_PATH)


def test_session_pose_shape_and_notify(body: bh.Body) -> None:
    session = StudioSession(body, "smpl")
    kinds: list[str] = []
    session.subscribe(kinds.append)
    assert session.vertices.shape == (body.num_vertices, 3) and session.total_mass > 40.0
    elbow = next(handle for handle in session.handles if handle.name == "left_elbow")
    old = session.joint_poses[elbow.native_index].clone()
    new = old.clone()
    new[3:] = torch.tensor([0.0, 0.0, 0.3827, 0.9239], dtype=old.dtype)
    session.rotate_joint(elbow, old, new)
    assert not torch.equal(session.q, session.identity.q_neutral)
    session.set_shape(betas=session.shape_tensor("betas", [2.0] + [0.0] * (body.identity_groups["betas"] - 1)))
    session.reset_pose()
    assert torch.equal(session.q, session.identity.q_neutral)
    assert kinds == ["pose", "shape", "pose"]
    session.set_frame_set({"heel": FrameEntry(10)})
    assert session.resolved_joint(session.frame_set["heel"]) == int(session.vertex_native_joints[10])


def test_gizmo_and_layers_follow_the_session(body: bh.Body) -> None:
    server = FakeServer()
    viewer = Viewer(server=server)
    session = StudioSession(body, "smpl")
    layers = StudioLayers(viewer, session)
    status: list[str] = []
    gizmo = GizmoController(viewer, session, on_status=status.append)
    gizmo.select(18)
    assert gizmo.gizmo is not None and status[-1].startswith("posing")
    pose = session.joint_poses[18].numpy().copy()
    pose[3:] = [0.0, 0.3827, 0.0, 0.9239]
    gizmo._on_drag(gizmo.handle_for(18), pose)
    np.testing.assert_allclose(layers.segmentation.vertices[0], session.vertices.numpy(), atol=1e-5)
    layers.set_highlight(18)
    assert layers.skeleton.colors[0, 18].tolist() == [255, 200, 40]
    session.reset_segmentation()
    layers.set_explode(0.5)
    session.load_body(body, "smpl")
    assert gizmo.gizmo is None
