"""Human layers on the BetterRobot viewer: state-driven layers, frames by name, skinned binds, in-place updates.

SMPL runs on the random-weight archive (record 0016), so these tests run in the full tier; SOMA and the real SMPL
case read their licensed weights. Each layer's arrays are checked against the state at step ``K`` (not the first, so
a layer that draws step 0 or a neighbouring step fails) and read back from BetterRobot's stand-in server
(``better_robot.testing.FakeServer``) as the browser would receive them.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch
from better_robot import Viewer
from better_robot.exceptions import ArgumentError, ShapeError, UnknownNameError
from better_robot.kinematics import integrate
from better_robot.lie import se3
from better_robot.testing import FakeServer
from better_robot.viewer import SkinnedMeshLayer

import better_human as bh
from better_human.io import weight_path
from better_human.viz import (
    HUMAN_THEME,
    HumanView,
    forces_layer,
    frames_layer,
    human_layers,
    mesh_layer,
    skeleton_layer,
    skinned_layer,
)
from tests.support.random_archives import random_weight_spec

SKINNED_TOLERANCE_M = 1e-6
#: The step every layer is checked at: not the first, so drawing step 0 or a neighbouring step fails.
K = 3
#: A colour and an opacity no default has, so a layer that drops either fails.
COLOR = (0.1, 0.5, 0.9)
OPACITY = 0.4


def _clip(spec) -> tuple:
    """``(identity, state)``: five steps that bend one shoulder from 0 to 1 rad, with a heel contact frame."""
    body = bh.HumanModel.from_spec(spec, density=None, contact_frames={"heel": (3387, None)})
    identity = body.bake()
    frames = 5
    tangent = torch.zeros(frames, identity.model.nv)
    tangent[:, 6 + 3 * 15] = torch.linspace(0.0, 1.0, frames)
    q = integrate(identity.model, identity.model.q_neutral.expand(frames, -1), tangent)
    return identity, identity.forward(q)


@pytest.fixture(scope="module")
def clip():
    return _clip(random_weight_spec("smpl"))


def _uint8(color) -> np.ndarray:
    """An RGB in [0, 1] as the uint8 triple viser receives."""
    return (np.asarray(color, dtype=np.float32) * 255.0 + 0.5).astype(np.uint8)


def _shown(layers, step: int) -> FakeServer:
    """The stand-in server after ``layers`` are added and the viewer shows ``step``."""
    server = FakeServer()
    viewer = Viewer(server=server)
    viewer.add(layers, group="human")
    viewer.show(step)
    return server


def _node(server: FakeServer, path: str):
    return server.scene.nodes[f"/human/{path}"]


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


def test_mesh_layer_draws_the_state(clip) -> None:
    identity, state = clip
    layer = mesh_layer(state, color=COLOR, opacity=OPACITY)
    faces = identity.human.spec.faces.numpy()
    np.testing.assert_allclose(layer.vertices[K], state.vertices[K].numpy(), atol=1e-6)
    np.testing.assert_array_equal(layer.faces, faces)
    assert (layer.color, layer.opacity) == (COLOR, OPACITY)
    node = _node(_shown([layer], K), "mesh/mesh")
    np.testing.assert_allclose(node.vertices, state.vertices[K].numpy(), atol=1e-6)
    np.testing.assert_array_equal(node.props["faces"], faces)
    np.testing.assert_array_equal(node.props["color"], _uint8(COLOR))
    assert node.props["opacity"] == pytest.approx(OPACITY)


def test_skinned_layer_draws_the_state(clip) -> None:
    identity, state = clip
    layer = skinned_layer(state, color=COLOR, opacity=OPACITY)
    faces = identity.human.spec.faces.numpy()
    np.testing.assert_allclose(layer.vertices, identity.rest_vertices.numpy(), atol=1e-6)
    np.testing.assert_allclose(layer.bone_poses[K], state.joint_poses[K].numpy(), atol=1e-6)
    np.testing.assert_array_equal(layer.faces, faces)
    assert (layer.color, layer.opacity) == (COLOR, OPACITY)
    node = _node(_shown([layer], K), "mesh/mesh")
    np.testing.assert_array_equal(node.props["faces"], faces)
    np.testing.assert_array_equal(node.props["color"], _uint8(COLOR))
    assert node.props["opacity"] == pytest.approx(OPACITY)
    positions = np.stack([bone.position for bone in node.bones[: layer.bind_poses.shape[0]]])
    np.testing.assert_allclose(positions, state.joint_poses[K, :, :3].numpy(), atol=1e-6)


def test_skeleton_layer_draws_the_native_skeleton(clip) -> None:
    identity, state = clip
    layer = skeleton_layer(state)
    np.testing.assert_allclose(layer.joints[K], state.joint_translations[K].numpy(), atol=1e-6)
    assert layer.parent_ids.tolist() == list(identity.human.spec.parents)
    np.testing.assert_array_equal(layer.colors, np.broadcast_to(_uint8(HUMAN_THEME.joint), (24, 3)))
    assert layer.bone_color == HUMAN_THEME.link
    server = _shown([layer], K)
    np.testing.assert_allclose(_node(server, "skeleton/joints").batched_positions, layer.joints[K], atol=1e-6)
    np.testing.assert_array_equal(_node(server, "skeleton/joints").props["batched_colors"], layer.colors)
    np.testing.assert_array_equal(_node(server, "skeleton/bones").props["batched_colors"], _uint8(HUMAN_THEME.link))


def test_frames_layer_draws_every_model_frame(clip) -> None:
    identity, state = clip
    layer = frames_layer(state)
    assert layer.poses.shape == (5, identity.model.nframes, 7)
    np.testing.assert_allclose(layer.poses[K], state.frame_poses[K].numpy(), atol=1e-6)


def test_forces_layer_draws_scaled_forces_at_the_frames(clip) -> None:
    _, state = clip
    forces = torch.arange(1.0, 16.0).reshape(5, 1, 3)
    layer = forces_layer(state, ["heel"], forces=forces, scale=0.01)
    heel = state.model.frame_id("heel")
    np.testing.assert_allclose(layer.origins[K, 0], state.kinematics.frame_translations[K, heel].numpy(), atol=1e-6)
    np.testing.assert_allclose(layer.vectors[K, 0], 0.01 * forces[K, 0].numpy(), atol=1e-6)


def test_forces_layer_takes_frame_names(clip) -> None:
    _, state = clip
    layer = forces_layer(state, ["heel"], forces=torch.ones(5, 1, 3))
    np.testing.assert_allclose(layer.origins[:, 0], state.kinematics.frame_translations[:, -1].numpy(), atol=1e-6)
    with pytest.raises(UnknownNameError):
        forces_layer(state, ["toe"], forces=torch.ones(5, 1, 3))
    with pytest.raises(ShapeError, match="len\\(frame_names\\) = 1"):
        forces_layer(state, ["heel"], forces=torch.ones(5, 2, 3))
    with pytest.raises(TypeError, match="torch.Tensor"):
        forces_layer(state, ["heel"], forces=np.ones((5, 1, 3)))


def test_human_layers_hold_a_constant_force_at_every_step(clip) -> None:
    _, state = clip
    forces = {"heel": torch.arange(1.0, 16.0).reshape(5, 3), "pelvis": torch.tensor([0.0, 9.0, 0.0])}
    arrows = human_layers(state, mesh=None, skeleton=False, forces=forces)[0]
    np.testing.assert_allclose(arrows.vectors[K], 1e-3 * np.stack([forces["heel"][K], forces["pelvis"]]), atol=1e-9)
    with pytest.raises(ShapeError, match="one T"):
        human_layers(state, forces={"heel": torch.ones(5, 3), "pelvis": torch.ones(4, 3)})
    with pytest.raises(TypeError, match="torch.Tensor"):
        human_layers(state, forces={"heel": np.ones(3)})


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
        random_weight_spec("smpl")
        if family == "smpl"
        else bh.families.soma(weight_path("SOMA", "soma_mid.npz"), correctives=False)
    )
    human = bh.HumanModel.from_spec(spec, density=None)
    identity = human.bake()
    state = identity.forward(identity.model.q_neutral)
    layer = skinned_layer(state)
    error = (_dense_skin(layer, human.skin_matrix, 0) - state.vertices.double()).abs().max().item()
    assert error <= SKINNED_TOLERANCE_M, f"{family} skinned layer vs state vertices: {error:.3e} m"


@pytest.mark.parametrize("weights", ["random", pytest.param("real", marks=pytest.mark.licensed)])
def test_skinned_layer_matches_exact_vertices_without_correctives(clip, weights: str) -> None:
    """Up to the largest pose corrective + 1 mm: 27.5 mm on the random archive, 11.0 mm on the real SMPL weights."""
    identity, state = (
        clip if weights == "random" else _clip(bh.families.smpl(weight_path("smpl", "SMPL_NEUTRAL.npz"), num_betas=10))
    )
    skinned = human_layers(state, mesh="skinned", skeleton=False)[0]
    exact = state.vertices[3].numpy()
    corrective = identity.human.spec.corrective.apply(state.corrective_feature)[3].abs().max().item()
    assert np.abs(skinned.skin(3) - exact).max() < corrective + 1e-3


def test_skinned_layer_binds_follow_streams_that_move_joints() -> None:
    """SMPL-X expression streams move the rest joints (0.10 m on the random archive): the layer's binds are the
    state's, so its skinning equals the state's skin affines applied to the identity's rest mesh."""
    identity, clip_state = _clip(random_weight_spec("smplx"))
    state = identity.forward(clip_state.q[K], {"expression": 2.0 * torch.ones(10)})
    weights, affines = identity.human.skin_matrix.double(), state.skin_affines.double()
    rest = identity.rest_vertices.double()
    expected = torch.einsum("vj,jab,vb->va", weights, affines[..., :3], rest) + weights @ affines[..., 3]
    error = (_dense_skin(skinned_layer(state), weights, 0) - expected).abs().max().item()
    assert error <= SKINNED_TOLERANCE_M, f"skinned layer vs the state's binds with streams: {error:.3e} m"
    streamed = identity.forward(clip_state.q, {"expression": torch.linspace(0.0, 2.0, 5)[:, None].expand(5, 10)})
    with pytest.raises(ShapeError, match="one bind set"):
        skinned_layer(streamed)


def test_skinned_layer_refuses_scaled_binds_and_identity_batches(clip) -> None:
    identity, _ = clip
    scaled = replace(identity, inverse_bind_affines=1.1 * identity.inverse_bind_affines)
    with pytest.raises(ArgumentError, match="rigid binds"):
        skinned_layer(scaled.forward(identity.model.q_neutral))
    batch = identity.human.bake(betas=torch.zeros(2, 10))
    with pytest.raises(ShapeError, match="one identity"):
        skinned_layer(batch.forward(batch.model.q_neutral))


@pytest.mark.parametrize("mesh", ["exact", "skinned"])
def test_human_view_updates_every_layer_in_place(clip, mesh: str) -> None:
    identity, state = clip
    view = HumanView(identity.forward(identity.model.q_neutral), mesh=mesh, frames=True)
    body, skeleton, frames = view.layers
    arrays = {"exact": "vertices", "skinned": "bone_poses"}[mesh]
    moved = identity.forward(state.q[K])
    expected = {
        (body, arrays): moved.vertices if mesh == "exact" else moved.joint_poses,
        (skeleton, "joints"): moved.joint_translations,
        (frames, "poses"): moved.frame_poses,
    }
    before = {key: getattr(*key).copy() for key in expected}
    view.update(moved)
    for (layer, field), value in expected.items():
        drawn = getattr(layer, field)
        np.testing.assert_allclose(drawn[0], value.numpy(), atol=1e-6, err_msg=f"{layer.name}.{field}")
        assert np.abs(drawn - before[layer, field]).max() > 1e-3, f"{layer.name}.{field} did not move"
    with pytest.raises(ShapeError, match="single pose"):
        HumanView(state)


def test_human_view_refuses_a_clip_or_another_body_and_writes_nothing(clip) -> None:
    identity, state = clip
    view = HumanView(identity.forward(identity.model.q_neutral), mesh="skinned", frames=True)
    other = bh.HumanModel.from_spec(random_weight_spec("smplx"), density=None).bake()
    before = [array.copy() for array in (view.layers[0].bone_poses, view.layers[1].joints, view.layers[2].poses)]
    for refused in (state, other.forward(other.model.q_neutral)):
        with pytest.raises(ShapeError, match="single pose of the drawn body"):
            view.update(refused)
    after = (view.layers[0].bone_poses, view.layers[1].joints, view.layers[2].poses)
    assert all(np.array_equal(old, new) for old, new in zip(before, after, strict=True))


def test_world_points_for_camera_fitting(clip) -> None:
    _, state = clip
    for layer in human_layers(state):
        points = layer.world_points(2)
        assert points is not None and points.ndim == 2 and points.shape[-1] == 3 and len(points) > 0


def test_layers_reach_the_stand_in_server(clip) -> None:
    _, state = clip
    server = _shown(human_layers(state, mesh="skinned", frames=True, forces={"heel": torch.ones(5, 3)}), K)
    assert {"MeshSkinnedHandle", "BatchedAxesHandle"} <= {node.kind for node in server.scene.nodes.values()}
