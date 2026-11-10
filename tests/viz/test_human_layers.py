"""Human layers on the BetterRobot viewer: state-driven layers, frames by name, skinned binds, in-place updates.

SMPL runs on the random-weight archive (record 0016), so these tests run in the full tier; SOMA reads its licensed
weights. Scenes are built on BetterRobot's stand-in server (``better_robot.testing.FakeServer``).
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from better_robot.exceptions import ArgumentError, ShapeError, UnknownNameError
from better_robot.kinematics import integrate
from better_robot.lie import se3
from better_robot.testing import FakeServer
from better_robot.viewer import SkinnedMeshLayer, Viewer

import better_human as bh
from better_human.io import weight_path
from better_human.viz import HumanView, forces_layer, human_layers, skinned_layer

SMPL_RANDOM = Path(__file__).parents[1] / "support" / "random_weights" / "smpl_random.npz"
SKINNED_TOLERANCE_M = 1e-6


@pytest.fixture(scope="module")
def clip():
    body = bh.HumanModel.from_spec(bh.families.smpl(SMPL_RANDOM), density=None, contact_frames={"heel": (3387, None)})
    identity = body.bake()
    frames = 5
    tangent = torch.zeros(frames, identity.model.nv)
    tangent[:, 6 + 3 * 15] = torch.linspace(0.0, 1.0, frames)
    q = integrate(identity.model, identity.model.q_neutral.expand(frames, -1), tangent)
    return identity, identity.forward(q)


def _dense_skin(layer: SkinnedMeshLayer, weights: torch.Tensor, step: int) -> torch.Tensor:
    """``(V, 3)`` blend of the layer's bind and bone poses with every weight (viser keeps the four largest)."""
    bind = torch.from_numpy(layer.bind_poses).double()
    posed = torch.from_numpy(layer.bone_poses[step]).double()
    moved = se3.act(se3.compose(posed, se3.inverse(bind)), torch.from_numpy(layer.vertices).double()[:, None])
    return (weights.double()[..., None] * moved).sum(-2)


def test_layers_have_one_frame_per_pose(clip) -> None:
    identity, state = clip
    layers = human_layers(state, frames=True, forces={"heel": torch.ones(5, 3)})
    assert [layer.name for layer in layers] == ["mesh", "skeleton", "frames", "forces"]
    assert all(layer.nsteps == 5 for layer in layers)
    assert layers[0].vertices.shape == (5, identity.human.spec.num_vertices, 3)
    assert layers[1].joints.shape == (5, 24, 3)
    np.testing.assert_allclose(layers[3].origins[2, 0], state.kinematics.frame_translations[2, -1].numpy(), atol=1e-6)


def test_forces_layer_takes_frame_names(clip) -> None:
    _, state = clip
    layer = forces_layer(state, ["heel"], forces=torch.ones(5, 1, 3))
    np.testing.assert_allclose(layer.origins[:, 0], state.kinematics.frame_translations[:, -1].numpy(), atol=1e-6)
    with pytest.raises(UnknownNameError):
        forces_layer(state, ["toe"], forces=torch.ones(5, 1, 3))
    with pytest.raises(ShapeError, match="len\\(frame_names\\) = 1"):
        forces_layer(state, ["heel"], forces=torch.ones(5, 2, 3))


@pytest.mark.parametrize(
    "family",
    [
        "smpl",
        pytest.param("soma", marks=pytest.mark.licensed),
    ],
)
def test_skinned_layer_matches_state_vertices(family: str) -> None:
    """Records 0003 and 0010: the binds invert ``inverse_bind_affines``, so the layer's skinning gives the state's
    vertices; SOMA's binds rotate (0.292 m off with identity-rotation binds). At the neutral pose, without SOMA's
    correctives (SMPL's vanish there), compared with every skin weight."""
    spec = (
        bh.families.smpl(SMPL_RANDOM)
        if family == "smpl"
        else bh.families.soma(weight_path("SOMA", "soma_mid.npz"), correctives=False)
    )
    human = bh.HumanModel.from_spec(spec, density=None)
    identity = human.bake()
    state = identity.forward(identity.model.q_neutral)
    layer = skinned_layer(state)
    error = (_dense_skin(layer, human.skin_matrix, 0) - state.vertices.double()).abs().max().item()
    assert error <= SKINNED_TOLERANCE_M, f"{family} skinned layer vs state vertices: {error:.3e} m"


def test_skinned_layer_follows_the_pose_up_to_correctives(clip) -> None:
    identity, state = clip
    skinned = human_layers(state, mesh="skinned", skeleton=False)[0]
    exact = state.vertices[3].numpy()
    corrective = identity.human.spec.corrective.apply(state.corrective_feature)[3].norm(dim=-1).max().item()
    assert np.abs(skinned.skin(3) - exact).max() <= corrective + 1e-5


def test_skinned_layer_refuses_scaled_binds_and_identity_batches(clip) -> None:
    identity, _ = clip
    scaled = replace(identity, inverse_bind_affines=1.1 * identity.inverse_bind_affines)
    with pytest.raises(ArgumentError, match="rigid binds"):
        skinned_layer(scaled.forward(identity.model.q_neutral))
    batch = identity.human.bake(betas=torch.zeros(2, 10))
    with pytest.raises(ShapeError, match="one identity"):
        skinned_layer(batch.forward(batch.model.q_neutral))


def test_human_view_updates_in_place(clip) -> None:
    identity, state = clip
    single = identity.forward(identity.model.q_neutral)
    view = HumanView(single, frames=True)
    mesh, skeleton, frames = view.layers
    before = mesh.vertices.copy()
    moved = identity.forward(state.q[-1])
    view.update(moved)
    np.testing.assert_allclose(mesh.vertices[0], moved.vertices.numpy(), atol=1e-6)
    assert np.abs(mesh.vertices - before).max() > 0.05
    np.testing.assert_allclose(skeleton.joints[0], moved.joint_translations.numpy(), atol=1e-6)
    assert frames.poses.shape == (1, identity.model.nframes, 7)
    with pytest.raises(ShapeError, match="single pose"):
        HumanView(state)


def test_world_points_for_camera_fitting(clip) -> None:
    _, state = clip
    for layer in human_layers(state):
        points = layer.world_points(2)
        assert points is not None and points.ndim == 2 and points.shape[-1] == 3 and len(points) > 0


def test_layers_reach_the_stand_in_server(clip) -> None:
    _, state = clip
    server = FakeServer()
    viewer = Viewer(server=server)
    layers = human_layers(state, mesh="skinned", frames=True, forces={"heel": torch.ones(5, 3)})
    viewer.add(layers, group="human")
    viewer.show(2)
    skinned = next(node for node in server.scene.nodes.values() if node.kind == "MeshSkinnedHandle")
    positions = np.stack([bone.position for bone in skinned.bones[: layers[0].bind_poses.shape[0]]])
    np.testing.assert_allclose(positions, state.joint_poses[2, :, :3].numpy(), atol=1e-6)
    assert {"MeshSkinnedHandle", "BatchedAxesHandle"} <= {node.kind for node in server.scene.nodes.values()}
