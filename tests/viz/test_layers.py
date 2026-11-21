"""Human layers on the BetterRobot viewer: state-driven layers, frames by name, the skin of RobotVisual, live updates.

SMPL runs on the random-weight archive, so these tests run in the full tier; SOMA, MHR and the real
SMPL-X cases read their licensed weights. Each layer's arrays are checked against the state at step ``K`` (not the
first, so a layer that draws step 0 or a neighbouring step fails) and read back from BetterRobot's stand-in server
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
from better_robot.viewer import RobotVisual

import better_human as bh
from better_human.io import weight_path
from better_human.viz import (
    HUMAN_THEME,
    HumanView,
    forces_layer,
    human_layers,
    mesh_layer,
    skeleton_layers,
    skin,
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
    """The stand-in server after ``layers`` are added under the tag ``human`` and the viewer shows ``step``."""
    server = FakeServer()
    viewer = Viewer(server=server)
    viewer.add(layers, tags="human")
    viewer.show(step)
    return server


def _array(value) -> np.ndarray:
    """A copy of a layer's array, or of a ``RobotVisual``'s ``q`` tensor."""
    return value.numpy().copy() if isinstance(value, torch.Tensor) else np.array(value, copy=True)


def _node(server: FakeServer, path: str):
    return server.scene.nodes[f"/human/{path}"]


def _drawn_skin(server: FakeServer, identity) -> torch.Tensor:
    """``(V, 3)``: the rest mesh skinned with the bind and current bone poses of the skinned node on ``server``, with
    every skin weight (viser keeps the four largest)."""
    (node,) = [node for node in server.scene.nodes.values() if node.kind == "MeshSkinnedHandle"]
    bones = identity.model.njoints
    wxyz = np.asarray(node.props["bone_wxyzs"])[:bones]
    bind = np.concatenate([np.asarray(node.props["bone_positions"])[:bones], wxyz[:, [1, 2, 3, 0]]], axis=-1)
    posed = np.stack([np.concatenate([bone.position, np.asarray(bone.wxyz)[[1, 2, 3, 0]]]) for bone in node.bones])
    moved = se3.compose(torch.from_numpy(posed[:bones]).double(), se3.inverse(torch.from_numpy(bind).double()))
    weights = torch.zeros(len(identity.rest_vertices), bones, dtype=torch.float64)
    weights[:, identity.human.joint_ids] = identity.human.skin_matrix.double()
    vertices = torch.from_numpy(np.asarray(node.props["vertices"])).double()
    return (weights[..., None] * se3.act(moved, vertices[:, None])).sum(-2)


def test_layers_have_one_frame_per_pose(clip) -> None:
    identity, state = clip
    layers = human_layers(state, frames=True, forces={"heel": torch.ones(5, 3)})
    assert [layer.name for layer in layers] == ["mesh", "joints", "bones", "frames", "forces"]
    assert all(layer.nsteps == 5 for layer in layers)
    assert layers[0].vertices.shape == (5, identity.human.spec.num_vertices, 3)
    assert layers[1].centers.shape == (5, 24, 3)
    assert layers[2].points.shape == (5, 23, 2, 3)
    np.testing.assert_allclose(layers[4].origins[2, 0], state.kinematics.frame_translations[2, -1].numpy(), atol=1e-6)


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


def test_skin_draws_the_state_in_a_robot_visual(clip) -> None:
    """The skinned node holds the identity's rest mesh and faces, the body colour and opacity, and its bones follow
    the model's joints at step ``K``."""
    identity, state = clip
    visual = RobotVisual(identity.model, state.q, skin=skin(identity), hue=COLOR, opacity=OPACITY)
    server = _shown([visual], K)
    (node,) = [node for node in server.scene.nodes.values() if node.kind == "MeshSkinnedHandle"]
    np.testing.assert_allclose(node.props["vertices"], identity.rest_vertices.numpy(), atol=1e-6)
    np.testing.assert_array_equal(node.props["faces"], identity.human.spec.faces.numpy())
    np.testing.assert_array_equal(node.props["color"], _uint8(COLOR))
    assert node.props["opacity"] == pytest.approx(OPACITY)
    positions = np.stack([bone.position for bone in node.bones[: identity.model.njoints]])
    np.testing.assert_allclose(positions, state.kinematics.joint_poses[K, :, :3].numpy(), atol=1e-6)


@pytest.mark.parametrize(
    "family",
    [
        "smpl",
        "smplh",
        "smplx",
        pytest.param("smplx_real", marks=pytest.mark.licensed),
        pytest.param("mhr", marks=pytest.mark.licensed),
    ],
)
def test_skinned_layer_matches_state_vertices(family: str) -> None:
    """What the browser skins (bind and bone poses of the skinned node, every skin weight) equals BetterHuman's
    linear blend skinning without correctives, and the exact vertices up to the largest corrective + 1 mm."""
    spec = {
        "smplx_real": lambda: bh.families.smplx(weight_path("smplx", "SMPLX_NEUTRAL.npz")),
        "mhr": lambda: bh.families.mhr(lod=1),
    }.get(family, lambda: random_weight_spec(family))()
    identity = bh.HumanModel.from_spec(spec, density=None).bake()
    q = torch.stack([identity.model.random_configuration(generator=torch.Generator().manual_seed(k)) for k in range(4)])
    state = identity.forward(q)
    drawn = _drawn_skin(_shown([RobotVisual(identity.model, q, skin=skin(identity))], K), identity)
    weights, affines = identity.human.skin_matrix.double(), state.skin_affines[K].double()
    rest = identity.rest_vertices.double()
    expected = torch.einsum("vj,jab,vb->va", weights, affines[..., :3], rest) + weights @ affines[..., 3]
    error = (drawn - expected).abs().max().item()
    assert error <= SKINNED_TOLERANCE_M, f"{family}: drawn skin vs linear blend skinning {error:.3e} m"
    corrective = (state.rest_vertices()[K] - identity.rest_vertices).abs().max().item()
    assert (drawn - state.vertices[K].double()).abs().max().item() < corrective + 1e-3


def test_skin_refuses_binds_off_q_neutral_and_identity_batches(clip) -> None:
    identity, _ = clip
    scaled = replace(identity, inverse_bind_affines=1.1 * identity.inverse_bind_affines)
    with pytest.raises(ArgumentError, match="does not follow its model's joints"):
        skin(scaled)
    batch = identity.human.bake(betas=torch.zeros(2, 10))
    with pytest.raises(ShapeError, match="one identity"):
        skin(batch)


@pytest.mark.licensed
def test_skin_refuses_soma_whose_joints_turn_away_from_the_model() -> None:
    """SOMA's native joints are not its model's joints (drawn with them, its skin landed 0.95 m off): exact only."""
    spec = bh.families.soma(weight_path("SOMA", "soma_mid.npz"))
    with pytest.raises(ArgumentError, match="does not follow its model's joints"):
        skin(bh.HumanModel.from_spec(spec, density=None).bake())


def test_skeleton_layers_draw_the_native_skeleton(clip) -> None:
    identity, state = clip
    joints, bones = skeleton_layers(state, radius=0.02)
    parents = identity.human.spec.parents
    translations = state.joint_translations[K].numpy()
    np.testing.assert_allclose(joints.centers[K], translations, atol=1e-6)
    assert np.all(joints.radii == np.float32(0.02))
    expected = np.stack([[translations[j], translations[parents[j]]] for j in range(1, 24)])
    np.testing.assert_allclose(bones.points[K], expected, atol=1e-6)
    server = _shown([joints, bones], K)
    spheres = _node(server, "joints/spheres")
    np.testing.assert_allclose(spheres.batched_positions, translations, atol=1e-6)
    srgb = (np.asarray(HUMAN_THEME.accent) * 255 + 0.5).astype(np.uint8) / 255  # viser reads these as linear
    accent = np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4) * 255 + 0.5
    np.testing.assert_array_equal(spheres.props["batched_colors"][0], accent.astype(np.uint8))
    np.testing.assert_allclose(_node(server, "bones/lines").points, expected, atol=1e-6)


def test_frames_draw_every_model_frame(clip) -> None:
    identity, state = clip
    (frames,) = human_layers(state, mesh=None, skeleton=False, frames=True)
    assert frames.poses.shape == (5, identity.model.nframes, 7)
    np.testing.assert_allclose(frames.poses[K], state.frame_poses[K].numpy(), atol=1e-6)


def test_forces_layer_draws_scaled_forces_at_the_frames(clip) -> None:
    _, state = clip
    forces = torch.arange(1.0, 16.0).reshape(5, 1, 3)
    layer = forces_layer(state, ["heel"], forces=forces, scale=0.01)
    heel = state.model.frame_id("heel")
    origin = state.kinematics.frame_translations[K, heel].numpy()
    np.testing.assert_allclose(layer.origins[K, 0], origin, atol=1e-6)
    np.testing.assert_allclose(layer.vectors[K, 0], forces[K, 0].numpy(), atol=1e-6)
    tail, tip = _node(_shown([layer], K), "forces/arrows").points[0]
    np.testing.assert_allclose(tip - tail, 0.01 * forces[K, 0].numpy(), atol=1e-6)


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
    np.testing.assert_allclose(arrows.vectors[K], np.stack([forces["heel"][K], forces["pelvis"]]), atol=1e-9)
    assert arrows.scale == pytest.approx(1e-3)
    with pytest.raises(ShapeError, match="one T"):
        human_layers(state, forces={"heel": torch.ones(5, 3), "pelvis": torch.ones(4, 3)})
    with pytest.raises(TypeError, match="torch.Tensor"):
        human_layers(state, forces={"heel": np.ones(3)})


@pytest.mark.parametrize("mesh", ["exact", "skinned"])
def test_human_view_updates_every_layer_and_the_viewer(clip, mesh: str) -> None:
    identity, state = clip
    view = HumanView(identity.forward(identity.model.q_neutral), mesh=mesh, frames=True)
    server = _shown(view, 0)
    body, joints, bones, frames = view.layers
    moved = identity.forward(state.q[K])
    expected = {
        (joints, "centers"): moved.joint_translations,
        (frames, "poses"): moved.frame_poses,
        **({(body, "vertices"): moved.vertices} if mesh == "exact" else {(body, "q"): moved.q}),
    }
    before = {key: _array(getattr(*key)) for key in expected}
    view.update(moved)
    for (layer, field), value in expected.items():
        drawn = _array(getattr(layer, field))
        np.testing.assert_allclose(drawn[0], value.numpy(), atol=1e-6, err_msg=f"{layer.name}.{field}")
        assert np.abs(drawn - before[layer, field]).max() > 1e-3, f"{layer.name}.{field} did not move"
    assert bones.points.shape == (1, 23, 2, 3)
    # The view pushed the pose itself: the browser's joints moved without a viewer.show().
    np.testing.assert_allclose(_node(server, "joints/spheres").batched_positions, moved.joint_translations, atol=1e-6)
    with pytest.raises(ShapeError, match="single pose"):
        HumanView(state)


def test_human_view_refuses_a_clip_or_another_body_and_writes_nothing(clip) -> None:
    identity, state = clip
    view = HumanView(identity.forward(identity.model.q_neutral), mesh="skinned", frames=True)
    other = bh.HumanModel.from_spec(random_weight_spec("smplx"), density=None).bake()
    before = [_array(array) for array in (view.layers[0].q, view.layers[1].centers, view.layers[3].poses)]
    for refused in (state, other.forward(other.model.q_neutral)):
        with pytest.raises(ShapeError, match="single pose of the drawn body"):
            view.update(refused)
    after = [_array(array) for array in (view.layers[0].q, view.layers[1].centers, view.layers[3].poses)]
    assert all(np.array_equal(old, new) for old, new in zip(before, after, strict=True))


def test_bounds_for_camera_fitting(clip) -> None:
    _, state = clip
    for layer in [*human_layers(state, frames=True), *human_layers(state, mesh="skinned", skeleton=False)]:
        low, high = layer.bounds(2)
        assert low.shape[-1] == high.shape[-1] == 3 and np.all(low <= high), layer.name


def test_layers_reach_the_stand_in_server(clip) -> None:
    _, state = clip
    server = _shown(human_layers(state, mesh="skinned", frames=True, forces={"heel": torch.ones(5, 3)}), K)
    kinds = {node.kind for node in server.scene.nodes.values()}
    assert {"MeshSkinnedHandle", "BatchedAxesHandle", "LineSegmentsHandle", "ArrowsHandle"} <= kinds, kinds
