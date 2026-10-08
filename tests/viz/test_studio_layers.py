"""Studio layers on the fake viewer server: segmentation, closed parts, overlays, visibility and reloads."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch
from better_robot.viewer import Viewer

import better_human as bh
from better_human.viz.studio import layers as studio_layers
from better_human.viz.studio.layers import StudioLayers
from better_human.viz.studio.state import StudioSession

from .fake_server import FakeServer

SMPL_PATH = Path(__file__).parents[2] / "models" / "smpl" / "SMPL_NEUTRAL.npz"


@pytest.fixture(scope="module")
def smpl() -> bh.Body:
    if not SMPL_PATH.is_file():
        pytest.skip(f"licensed SMPL fixture is absent: {SMPL_PATH}")
    return bh.load("smpl", SMPL_PATH)


def _setup(body: bh.Body) -> tuple[StudioSession, StudioLayers, FakeServer]:
    server = FakeServer()
    session = StudioSession(body, "smpl")
    return session, StudioLayers(Viewer(server=server), session), server


def _part_nodes(server: FakeServer, layer: str) -> list[str]:
    return [name for name, node in server.scene.nodes.items() if f"/{layer}/part_" in name and not node.removed]


def test_segmentation_face_groups_cover_every_face(smpl: bh.Body) -> None:
    _, layers, server = _setup(smpl)
    layers.set_mode("segmentation")
    groups = layers.mesh.segmentation.face_groups()
    assert sum(len(faces) for faces, _ in groups) == smpl.faces.shape[0]
    assert len(_part_nodes(server, "mesh")) == len(groups)


def test_segmentation_rebuild_on_edit(smpl: bh.Body) -> None:
    session, layers, server = _setup(smpl)
    layers.set_mode("segmentation")
    ids = (session.vertices[:, 0] > 0).to(torch.long)
    session.set_segmentation(("right", "left"), ids)
    groups = layers.mesh.segmentation.face_groups()
    assert len(groups) == 2 and len(_part_nodes(server, "mesh")) == 2
    session.reset_segmentation()
    assert len(layers.mesh.segmentation.face_groups()) >= len(session.part_names)


def test_parts_are_closed_meshes(smpl: bh.Body) -> None:
    session, layers, server = _setup(smpl)
    layers.set_mode("parts")
    assert layers.mesh.parts.closed
    rest = session.identity.forward(session.identity.q_neutral).vertices
    meshes = layers.mesh.parts.part_meshes(rest)
    assert len(meshes) == len(_part_nodes(server, "mesh")) == len(session.part_names)
    for _, _, vertices, faces in meshes:
        edges = Counter(tuple(sorted(edge)) for face in faces.tolist() for edge in zip(face, face[1:] + face[:1]))
        assert set(edges.values()) == {2}, "every edge of a closed part is shared by exactly two faces"
        assert faces.max() < len(vertices)


def test_parts_fall_back_to_open_submeshes(smpl: bh.Body, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*_args: object) -> None:
        raise ValueError("no closed topology")

    monkeypatch.setattr(studio_layers, "build_part_topology", fail)
    session, layers, server = _setup(smpl)
    layers.set_mode("parts")
    assert not layers.mesh.parts.closed
    assert len(_part_nodes(server, "mesh")) == len(session.part_names)


def test_parts_explode_moves_parts_apart(smpl: bh.Body) -> None:
    _, layers, _ = _setup(smpl)
    layers.set_mode("parts")
    layers.mesh.parts.set_step(0)
    assembled = np.stack([np.asarray(handle.position) for _, handle, _ in layers.mesh.parts._handles_by_part])
    layers.set_explode(0.5)
    layers.mesh.parts.set_step(0)
    exploded = np.stack([np.asarray(handle.position) for _, handle, _ in layers.mesh.parts._handles_by_part])
    assert layers.mesh.parts.explode == 0.5
    assert np.linalg.norm(exploded - assembled, axis=1).max() > 0.05


def test_opacity_radius_highlight_and_edit_overlay(smpl: bh.Body) -> None:
    session, layers, _ = _setup(smpl)
    layers.set_opacity(0.4)
    layers.set_radius(0.03)
    assert layers.mesh.segmentation.opacity == 0.4 and layers.radius == 0.03
    layers.set_highlight(7)
    assert layers.skeleton.colors[0, 7].tolist() == [255, 200, 40]
    layers.set_highlight(None)
    assert layers.skeleton.colors[0, 7].tolist() != [255, 200, 40]
    layers.set_edit_vertices(np.array([0, 1, 2]))
    np.testing.assert_allclose(layers.edit.points[0], session.vertices[[0, 1, 2]].numpy(), atol=1e-6)


def test_pose_change_updates_arrays_in_place(smpl: bh.Body) -> None:
    session, layers, _ = _setup(smpl)
    vertices = layers.mesh.segmentation.vertices
    elbow = next(h for h in session.handles if h.name == "left_elbow")
    old = session.joint_poses[elbow.native_index].clone()
    new = old.clone()
    new[3:] = torch.tensor([0.0, 0.0, 0.3827, 0.9239], dtype=old.dtype)
    session.rotate_joint(elbow, old, new)
    assert layers.mesh.segmentation.vertices is vertices
    np.testing.assert_allclose(vertices[0], session.vertices.numpy(), atol=1e-5)
    np.testing.assert_allclose(layers.skeleton.joints[0], session.state.joints.numpy(), atol=1e-5)


def test_visibility_defaults_and_body_reload(smpl: bh.Body) -> None:
    session, layers, _ = _setup(smpl)
    assert layers.visible == {"mesh": True, "frames": True, "skeleton": True, "edit": False}
    layers.set_visible("mesh", False)
    layers.set_mode("parts")
    layers.set_highlight(3)
    old = dict(layers.layers)
    session.load_body(smpl, "smpl")
    assert layers.layers == old and all(layers.layers[name] is old[name] for name in old)
    assert layers.highlight is None and layers.visible["mesh"] is False and layers.mesh.mode == "parts"


def _mesh_nodes(server: FakeServer) -> list[str]:
    return [name for name, node in server.scene.nodes.items() if "/mesh/" in name and not node.removed]


def _visible_parts(server: FakeServer) -> int:
    return sum(bool(server.scene.nodes[name].visible) for name in _part_nodes(server, "mesh"))


def test_mode_switch_keeps_one_mesh_layer_and_visibility(smpl: bh.Body) -> None:
    session, layers, server = _setup(smpl)
    mesh = layers.mesh
    for mode in ("segmentation", "parts", "skin"):
        layers.set_mode(mode)
        assert layers.mesh is mesh and [layer.name for layer in layers.viewer.layers].count("mesh") == 1
        assert mesh.visible and all(server.scene.nodes[name].visible for name in _mesh_nodes(server))
    assert _mesh_nodes(server) == [name for name in _mesh_nodes(server) if name.endswith("/mesh/mesh")]
    layers.set_visible("mesh", False)
    layers.set_mode("segmentation")
    assert not mesh.visible and _visible_parts(server) == 0 and len(_part_nodes(server, "mesh")) > 0


def test_hidden_parts_stay_hidden_after_shape_and_segmentation_change(smpl: bh.Body) -> None:
    session, layers, server = _setup(smpl)
    layers.set_mode("parts")
    layers.set_visible("mesh", False)
    betas = session.shape_tensor("betas", [2.0] + [0.0] * (smpl.identity_groups["betas"] - 1))
    session.set_shape(betas=betas)
    session.set_segmentation(("right", "left"), (session.vertices[:, 0] > 0).to(torch.long))
    assert _visible_parts(server) == 0
    assert layers.mesh.parts.stale and layers.mesh.segmentation.stale
    layers.set_visible("mesh", True)
    assert not layers.mesh.parts.stale
    assert _visible_parts(server) == len(_part_nodes(server, "mesh")) == 2


def test_visible_parts_keep_visibility_through_viewer_rebuild(smpl: bh.Body) -> None:
    session, layers, server = _setup(smpl)
    layers.set_mode("parts")
    count = len(_part_nodes(server, "mesh"))
    session.set_shape(betas=session.shape_tensor("betas", [1.0] + [0.0] * (smpl.identity_groups["betas"] - 1)))
    assert _visible_parts(server) == len(_part_nodes(server, "mesh")) == count
    assert layers.visible["mesh"] is True


def test_opacity_radius_and_edit_keep_layer_objects(smpl: bh.Body) -> None:
    _, layers, server = _setup(smpl)
    before = {name: layer for name, layer in layers.layers.items()}
    layers.set_opacity(0.4)
    layers.set_radius(0.03)
    layers.set_edit_vertices(np.arange(5))
    assert all(layers.layers[name] is before[name] for name in before)
    assert server.scene.nodes[_mesh_nodes(server)[0]].opacity == 0.4
    assert layers.edit.points.shape[1] == 5 and layers.visible["edit"]
