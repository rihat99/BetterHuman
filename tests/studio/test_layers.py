"""Studio layers on the fake viewer server: segmentation, closed parts, overlays, visibility and reloads."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import pytest
import torch
from better_robot import Viewer
from better_robot.exceptions import ArgumentError
from better_robot.testing import FakeServer

import better_human as bh
from better_human.io import weight_path
from better_human.studio import layers as studio_layers
from better_human.studio.layers import StudioLayers
from better_human.studio.state import StudioSession

pytestmark = pytest.mark.licensed


@pytest.fixture(scope="module")
def smpl() -> bh.HumanModel:
    return bh.load("smpl", weight_path("smpl", "SMPL_NEUTRAL.npz"))


def _setup(body: bh.HumanModel) -> tuple[StudioSession, StudioLayers, FakeServer]:
    server = FakeServer()
    session = StudioSession(body, "smpl")
    return session, StudioLayers(Viewer(server=server, ground=False), session), server


def _mesh_nodes(server: FakeServer) -> list[Any]:
    """The live nodes of the body layer (tag ``body``, layer ``mesh``)."""
    return [node for name, node in server.scene.nodes.items() if name.startswith("/body/mesh/") and not node.removed]


def test_segmentation_colours_every_vertex_by_its_part(smpl: bh.HumanModel) -> None:
    session, layers, server = _setup(smpl)
    layers.set_mode("segmentation")
    colors = layers.mesh.colors  # (V, 3) uint8, sent as one GLB node
    assert colors.shape == (smpl.spec.num_vertices, 3)
    assert len(np.unique(colors, axis=0)) == len(np.unique(session.part_ids.numpy()))
    assert [node.kind for node in _mesh_nodes(server)] == ["GlbHandle"]


def test_segmentation_replaces_the_mesh_on_edit(smpl: bh.HumanModel) -> None:
    session, layers, _ = _setup(smpl)
    layers.set_mode("segmentation")
    before = layers.mesh
    ids = (session.vertices[:, 0] > 0).to(torch.long)
    session.set_segmentation(("right", "left"), ids)
    assert layers.mesh is not before and len(np.unique(layers.mesh.colors, axis=0)) == 2
    session.reset_segmentation()
    assert len(np.unique(layers.mesh.colors, axis=0)) == len(session.part_names)


def test_parts_are_closed_meshes(smpl: bh.HumanModel) -> None:
    session, layers, server = _setup(smpl)
    layers.set_mode("parts")
    assert layers.closed
    rest = session.identity.forward(session.identity.model.q_neutral).vertices
    meshes, closed = studio_layers.part_meshes(session, rest)
    assert closed and len(meshes) == len(_mesh_nodes(server)) == len(session.part_names)
    for _, _, vertices, faces in meshes:
        edges = Counter(tuple(sorted(edge)) for face in faces.tolist() for edge in zip(face, face[1:] + face[:1]))
        assert set(edges.values()) == {2}, "every edge of a closed part is shared by exactly two faces"
        assert faces.max() < len(vertices)


def test_parts_fall_back_to_open_submeshes(smpl: bh.HumanModel, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*_args: object, **_kwargs: object) -> None:
        raise ArgumentError("part 3 owns no surface faces")

    monkeypatch.setattr(studio_layers.PartTopology, "from_part_ids", fail)
    session, layers, server = _setup(smpl)
    layers.set_mode("parts")
    assert not layers.closed
    assert len(_mesh_nodes(server)) == len(session.part_names)


def test_parts_explode_moves_parts_apart(smpl: bh.HumanModel) -> None:
    _, layers, server = _setup(smpl)
    layers.set_mode("parts")
    assembled = np.stack([np.asarray(node.position) for node in _mesh_nodes(server)])
    layers.set_explode(0.5)
    exploded = np.stack([np.asarray(node.position) for node in _mesh_nodes(server)])
    assert layers.explode == 0.5
    assert np.linalg.norm(exploded - assembled, axis=1).max() > 0.05


def test_opacity_radius_highlight_and_edit_overlay(smpl: bh.HumanModel) -> None:
    session, layers, server = _setup(smpl)
    layers.set_opacity(0.4)
    layers.set_radius(0.03)
    assert layers.mesh.opacity == 0.4 and layers.radius == 0.03 and np.all(layers.joints.radii == np.float32(0.03))
    layers.set_highlight(7)
    selected = server.scene.nodes["/selection/selected/spheres"]
    np.testing.assert_allclose(selected.batched_positions[0], session.state.joint_translations[7].numpy(), atol=1e-6)
    assert selected.batched_scales[0] > 0.03
    layers.set_highlight(None)
    assert selected.batched_scales[0] == 0.0
    layers.set_edit_vertices(np.array([0, 1, 2]))
    np.testing.assert_allclose(layers.edit.points[0], session.vertices[[0, 1, 2]].numpy(), atol=1e-6)


def test_pose_change_updates_arrays_in_place(smpl: bh.HumanModel) -> None:
    session, layers, _ = _setup(smpl)
    mesh, vertices = layers.mesh, layers.mesh.vertices
    elbow = next(h for h in session.handles if h.name == "left_elbow")
    old = session.joint_poses[elbow.native_index].clone()
    new = old.clone()
    new[3:] = torch.tensor([0.0, 0.0, 0.3827, 0.9239], dtype=old.dtype)
    session.rotate_joint(elbow, old, new)
    assert layers.mesh is mesh and layers.mesh.vertices is vertices
    np.testing.assert_allclose(vertices[0], session.vertices.numpy(), atol=1e-5)
    np.testing.assert_allclose(layers.joints.centers[0], session.state.joint_translations.numpy(), atol=1e-5)


def test_visibility_defaults_and_body_reload(smpl: bh.HumanModel) -> None:
    session, layers, _ = _setup(smpl)
    assert {name: layer.visible for name, layer in layers.layers.items()} == dict.fromkeys(layers.layers, True)
    assert layers.viewer.tags == ["body", "skeleton", "selection", "frames", "ground"]
    layers.viewer.set_visible("body", False)
    layers.set_mode("parts")
    layers.set_highlight(3)
    old = dict(layers.layers)
    session.load_body(smpl, "smpl")
    # Each layer is rebuilt from the new body and takes its predecessor's place in the viewer.
    assert list(layers.layers) == list(old) and layers.viewer.layers == list(layers.layers.values())
    assert not any(layers.layers[name] is old[name] for name in old)
    assert layers.highlight is None and layers.mesh.visible is False and layers.mode == "parts"


def test_mode_switch_keeps_one_mesh_layer_and_visibility(smpl: bh.HumanModel) -> None:
    _, layers, server = _setup(smpl)
    for mode in ("segmentation", "parts", "skin"):
        layers.set_mode(mode)
        assert [layer.name for layer in layers.viewer.layers].count("mesh") == 1
        assert layers.mesh.visible and all(node.visible for node in _mesh_nodes(server))
    assert [node.kind for node in _mesh_nodes(server)] == ["MeshHandle"]
    layers.viewer.set_visible("body", False)
    layers.set_mode("parts")
    assert not layers.mesh.visible and _mesh_nodes(server) and not any(node.visible for node in _mesh_nodes(server))


def test_hidden_parts_follow_shape_and_segmentation_and_show_again(smpl: bh.HumanModel) -> None:
    session, layers, server = _setup(smpl)
    layers.set_mode("parts")
    layers.viewer.set_visible("body", False)
    betas = session.shape_tensor("betas", [2.0] + [0.0] * (smpl.spec.identity_groups["betas"] - 1))
    session.set_shape(betas=betas)
    session.set_segmentation(("right", "left"), (session.vertices[:, 0] > 0).to(torch.long))
    assert not any(node.visible for node in _mesh_nodes(server)) and len(_mesh_nodes(server)) == 2
    layers.viewer.set_visible("body", True)
    assert all(node.visible for node in _mesh_nodes(server)) and len(_mesh_nodes(server)) == 2


def test_visible_parts_keep_visibility_through_a_shape_change(smpl: bh.HumanModel) -> None:
    session, layers, server = _setup(smpl)
    layers.set_mode("parts")
    count = len(_mesh_nodes(server))
    session.set_shape(betas=session.shape_tensor("betas", [1.0] + [0.0] * (smpl.spec.identity_groups["betas"] - 1)))
    assert all(node.visible for node in _mesh_nodes(server)) and len(_mesh_nodes(server)) == count
    assert layers.mesh.visible is True


def test_opacity_and_radius_keep_layer_objects_and_edit_replaces_its_layer(smpl: bh.HumanModel) -> None:
    _, layers, server = _setup(smpl)
    before = dict(layers.layers)
    layers.set_opacity(0.4)
    layers.set_radius(0.03)
    assert all(layers.layers[name] is before[name] for name in before)
    layers.set_edit_vertices(np.arange(5))
    assert layers.edit is not before["edit"] and layers.viewer.layers == list(layers.layers.values())
    assert all(layers.layers[name] is before[name] for name in before if name != "edit")
    assert _mesh_nodes(server)[0].opacity == 0.4
    assert layers.edit.points.shape[1] == 5 and layers.edit.visible


def test_the_ground_stands_under_the_feet_and_follows_the_shape(smpl: bh.HumanModel) -> None:
    session, layers, _ = _setup(smpl)
    rest = session.identity.forward(session.identity.model.q_neutral).vertices
    assert layers.layers["ground"].height == pytest.approx(float(rest[:, 1].min()))
    session.set_shape(betas=session.shape_tensor("betas", [3.0] + [0.0] * (smpl.spec.identity_groups["betas"] - 1)))
    rest = session.identity.forward(session.identity.model.q_neutral).vertices
    assert layers.layers["ground"].height == pytest.approx(float(rest[:, 1].min()))
