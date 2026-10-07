"""Human layers on the BetterRobot viewer: shapes, in-place updates, offline geometry."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

import better_human as bh
from better_human.viz import HumanView, human_layers

ROOT = Path(__file__).parents[2]
SMPL_PATH = ROOT / "models" / "smpl" / "SMPL_NEUTRAL.npz"


@pytest.fixture(scope="module")
def clip():
    if not SMPL_PATH.is_file():
        pytest.skip(f"licensed SMPL fixture is absent: {SMPL_PATH}")
    body = bh.load("smpl", SMPL_PATH, num_betas=10, mass=False, contact_frames={"heel": (3387, None)})
    identity = body.bake()
    frames = 5
    tangent = torch.zeros(frames, identity.nv)
    tangent[:, 6 + 3 * 15] = torch.linspace(0.0, 1.0, frames)
    q = identity.model.integrate(identity.q_neutral.expand(frames, -1), tangent)
    return identity, identity.forward(q)


def test_layers_have_one_frame_per_pose(clip) -> None:
    identity, state = clip
    layers = human_layers(state, frames=True, forces={"heel": torch.ones(5, 3)})
    assert [layer.name for layer in layers] == ["mesh", "skeleton", "frames", "forces"]
    assert all(layer.nsteps == 5 for layer in layers)
    assert layers[0].vertices.shape == (5, identity.body.num_vertices, 3)
    assert layers[1].joints.shape == (5, 24, 3)
    np.testing.assert_allclose(layers[3].origins[2, 0], state.kinematics.frame_translation[2, -1].numpy(), atol=1e-6)


def test_skinned_layer_matches_exact_vertices_without_correctives(clip) -> None:
    identity, state = clip
    skinned = human_layers(state, mesh="skinned", skeleton=False)[0]
    exact = state.vertices()[3].numpy()
    corrective = identity.spec.corrective.apply(state.corrective_feature)[3].abs().max().item()
    assert np.abs(skinned.skin(3) - exact).max() < corrective + 1e-3


def test_human_view_updates_in_place(clip) -> None:
    identity, state = clip
    single = identity.forward(identity.q_neutral)
    view = HumanView(single, frames=True)
    mesh, skeleton, frames = view.layers
    before = mesh.vertices.copy()
    moved = identity.forward(state.q[-1])
    view.update(moved)
    np.testing.assert_allclose(mesh.vertices[0], moved.vertices().numpy(), atol=1e-6)
    assert np.abs(mesh.vertices - before).max() > 0.05
    np.testing.assert_allclose(skeleton.joints[0], moved.joints.numpy(), atol=1e-6)
    assert frames.poses.shape == (1, identity.model.nframes, 7)


def test_world_points_for_camera_fitting(clip) -> None:
    _, state = clip
    for layer in human_layers(state):
        points = layer.world_points(2)
        assert points is not None and points.ndim == 2 and points.shape[-1] == 3 and len(points) > 0
