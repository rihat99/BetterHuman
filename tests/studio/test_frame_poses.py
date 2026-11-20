"""The studio draws the frames the model attaches: one frame rule, ``HumanState.contact_frame_poses``.

Full tier: the random-weight SMPL-X archive (read once per session) with the shipped SMPL-X contact set.
"""

from __future__ import annotations

import numpy as np
import torch
from better_robot import kinematics
from better_robot.lie import so3

import better_human as bh
from better_human.io import shipped_frame_set
from better_human.studio.layers import frame_poses
from better_human.studio.state import FrameEntry, StudioSession
from tests.support.random_archives import random_weight_spec


def test_the_studio_draws_the_contact_frames_of_the_model() -> None:
    """The frame set is checked once; later pose and shape edits reuse the identity's placements, never stale ones."""
    spec = random_weight_spec("smplx")
    frames = shipped_frame_set("smplx_contacts")
    model = bh.HumanModel.from_spec(spec, density=None, contact_frames=frames)
    session = StudioSession(bh.HumanModel.from_spec(spec, density=None), "smplx_neutral")
    generator = torch.Generator().manual_seed(0)
    betas = 0.5 * torch.randn(spec.identity_groups["betas"], generator=generator, dtype=spec.dtype)
    motion = 0.4 * torch.randn(2, model.model.nv, generator=generator, dtype=spec.dtype)
    q, moved = kinematics.integrate(model.model, model.model.q_neutral, motion)
    names = spec.joint_names
    session.set_frame_set(
        {
            name: FrameEntry(vertex, None if joint is None else names.index(joint))
            for name, (vertex, joint) in frames.items()
        }
    )

    frame_ids = torch.tensor(model.contact_frames.frame_ids)
    for edit, shape, pose in (("shape", betas, q), ("pose", betas, moved), ("second shape", -betas, moved)):
        session.set_shape(betas=shape)
        session.set_q(pose)
        expected = model.bake(betas=shape).forward(pose).frame_poses.index_select(-2, frame_ids)
        drawn = torch.from_numpy(frame_poses(session)[0]).to(spec.dtype)
        np.testing.assert_allclose(
            drawn[:, :3],
            expected[:, :3],
            rtol=0,
            atol=1e-6,
            err_msg=f"studio frame origins vs the model's contact frames after the {edit} edit",
        )
        angles = so3.log(so3.compose(so3.inverse(expected[:, 3:]), drawn[:, 3:])).norm(dim=-1)
        np.testing.assert_allclose(
            angles,
            np.zeros(len(frames)),
            rtol=0,
            atol=1e-5,
            err_msg=f"studio frame rotations vs the model's contact frames after the {edit} edit",
        )
