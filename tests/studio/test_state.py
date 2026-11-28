"""StudioSession: pose, shape, frames and notifications; full tier, the random-weight SMPL archive."""

from __future__ import annotations

import pytest
import torch

import better_human as bh
from better_human.studio.state import FrameEntry, StudioSession
from tests.support.random_archives import random_weight_spec


@pytest.fixture(scope="module")
def body() -> bh.HumanModel:
    return bh.HumanModel.from_spec(random_weight_spec("smpl"))


def test_session_pose_shape_and_notify(body: bh.HumanModel) -> None:
    session = StudioSession(body, "smpl")
    kinds: list[str] = []
    session.subscribe(kinds.append)
    assert session.vertices.shape == (body.spec.num_vertices, 3) and session.total_mass > 40.0
    elbow = next(handle for handle in session.handles if handle.name == "left_elbow")
    old = session.joint_poses[elbow.native_index].clone()
    new = old.clone()
    new[3:] = torch.tensor([0.0, 0.0, 0.3827, 0.9239], dtype=old.dtype)
    session.rotate_joint(elbow, old, new)
    assert not torch.equal(session.q, session.identity.model.q_neutral)
    session.set_shape(betas=session.shape_tensor("betas", [2.0] + [0.0] * (body.spec.identity_groups["betas"] - 1)))
    session.reset_pose()
    assert torch.equal(session.q, session.identity.model.q_neutral)
    assert kinds == ["pose", "shape", "pose"]
    session.set_frame_set({"heel": FrameEntry(10)})
    assert kinds[-1] == "frames" and session.frame_poses.shape == (1, 7)
