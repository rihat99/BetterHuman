"""Headless HumanStudio tests on the fake viser server."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from better_human.viz.studio import persistence
from better_human.viz.studio import state as studio_state
from better_human.viz.studio.app import HumanStudio
from tests.viz.fake_server import FakeHandle, FakeServer

pytestmark = pytest.mark.skipif(
    not Path(os.environ.get("BETTERHUMAN_MODELS_DIR", "models"), "smpl", "SMPL_NEUTRAL.npz").exists(),
    reason="licensed SMPL weights absent",
)


@pytest.fixture
def studio(tmp_path: Path):
    return HumanStudio(server=FakeServer(), data_dir=tmp_path)


def _click(studio, origin, target) -> None:
    origin = np.asarray(origin, float)
    event = SimpleNamespace(ray_origin=origin, ray_direction=np.asarray(target, float) - origin)
    studio.viewer.server.scene.pointer_callbacks["click"](event)


def test_imports_without_running_server() -> None:
    assert callable(HumanStudio) and hasattr(studio_state, "StudioSession")


def test_layers_session_and_tabs_built(studio) -> None:
    assert studio.session.model_key == "smpl_neutral"
    assert "click" in studio.viewer.server.scene.pointer_callbacks
    assert isinstance(studio.viewer.server.gui.control("Build bodies"), FakeHandle)
    assert studio.status.startswith("smpl_neutral loaded")


def test_shape_slider_changes_vertices(studio) -> None:
    before = studio.session.vertices.clone()
    slider = (
        studio.shape_sliders["betas"][0]
        if "betas" in studio.shape_sliders
        else next(iter(studio.shape_sliders.values()))[0]
    )
    slider.value = 2.0
    slider.fire()
    assert not torch.allclose(before, studio.session.vertices)


def test_click_on_joint_selects_and_highlights(studio) -> None:
    joint = studio.session.state.joints.detach().reshape(-1, 3)[5].numpy()
    _click(studio, joint + np.array([0.0, 0.0, 2.0]), joint)
    assert studio.gizmo.joint == 5


def test_pose_save_and_load_roundtrip(studio) -> None:
    q = studio.session.q.clone()
    q[..., -1] += 0.3
    studio.session.set_q(q)
    studio.save_pose("p")
    studio.session.reset_pose()
    studio.load_pose("p")
    assert torch.allclose(studio.session.q, q)


def test_segment_paint_and_build_bodies(studio) -> None:
    studio.segment.load_builtin()
    studio.segment.add_part("blob")
    studio.segment.set_painting(True)
    assert studio.mode == "segment"
    center = studio.session.vertices[0].detach().numpy()
    _click(studio, center + np.array([0.0, 0.0, 2.0]), center)
    rows = studio.segment.build_bodies()
    assert rows[-1][0] == "total" and rows[-1][1] > 10.0
    assert "blob" in dict(rows)


def test_frame_add_rename_delete_export(studio) -> None:
    studio.frames.set_adding(True)
    point = studio.session.vertices[100].detach().numpy()
    _click(studio, point + np.array([0.0, 0.0, 2.0]), point)
    name = studio.frames.active
    assert name in studio.session.frame_set
    studio.frames.rename(name, "toe")
    studio.frames.set_joint("toe", studio.session.body.joint_names[1])
    snippet = studio.frames.export()
    namespace: dict[str, object] = {}
    exec(snippet, {}, namespace)  # noqa: S102
    assert namespace["contact_frames"]["toe"][1] == studio.session.body.joint_names[1]
    studio.frames.save("set")
    studio.frames.delete("toe")
    assert "toe" not in studio.session.frame_set
    studio.frames.load("set")
    assert "toe" in studio.session.frame_set


def test_joint_selection_highlights_dims_and_reverts(studio) -> None:
    opacities: list[float] = []
    studio.layers.set_opacity = opacities.append
    before = np.array(studio.layers.skeleton.colors, copy=True)
    studio.select_joint(5)
    during = np.array(studio.layers.skeleton.colors, copy=True)
    changed = np.flatnonzero(~np.isclose(during, before).all(-1).reshape(-1))
    assert changed.tolist() == [5]
    studio.set_opacity(0.8)
    assert opacities[-1] == pytest.approx(0.8 * 0.35)
    studio.select_joint(None)
    assert opacities[-1] == pytest.approx(0.8)
    np.testing.assert_allclose(studio.layers.skeleton.colors, before)


def test_gizmo_drag_rotates_session_vertices(studio) -> None:
    studio.select_joint(5)
    before = studio.session.vertices.clone()
    pose = studio.session.joint_poses[5].detach().numpy().copy()
    pose[3:] = [np.sin(0.4), 0.0, 0.0, np.cos(0.4)]
    studio.gizmo._on_drag(studio.gizmo.handle_for(5), pose)
    assert (studio.session.vertices - before).abs().max() > 1e-3


def test_frame_click_selects_and_syncs_editor(studio) -> None:
    studio.frames.set_adding(True)
    point = studio.session.vertices[100].detach().numpy()
    _click(studio, point + np.array([0.0, 0.0, 2.0]), point)
    name = studio.frames.active
    studio.frames.set_adding(False)
    studio.frames.active = None
    studio.frames.frame_dropdown.value = None
    _click(studio, point + np.array([0.0, 0.0, 2.0]), point)
    assert studio.frames.active == name
    assert studio.frames.frame_dropdown.value == name


def test_marquee_assigns_vertices_from_screen_rect(studio) -> None:
    studio.segment.add_part("blob")
    blob = studio.session.part_names.index("blob")
    studio.start_marquee()
    camera = SimpleNamespace(position=(0.0, -0.3, 4.0), wxyz=(0.0, 0.0, 1.0, 0.0), fov=1.0, aspect=1.0)
    event = SimpleNamespace(client=SimpleNamespace(camera=camera), screen_pos=((0.0, 0.0), (1.0, 1.0)))
    studio.viewer.server.scene.pointer_callbacks["rect-select"](event)
    assert int((studio.session.part_ids == blob).sum()) == studio.session.vertices.shape[-2]
    assert set(studio.viewer.server.scene.pointer_callbacks) == {"click"}


def test_build_bodies_total_matches_session_mass(studio) -> None:
    rows = studio.segment.build_bodies()
    assert rows[-1][1] == pytest.approx(float(studio.session.total_mass), rel=0.01)


def test_density_table_and_saved_pose_dropdown(studio) -> None:
    studio.density_input.value = "Dempster"
    assert studio.chosen_density() == "Dempster"
    studio.reload(studio.chosen_density())
    assert studio.session.body.density.numel() > 1
    studio.save_pose("first")
    assert studio.saved_poses.options == ("first",) and studio.saved_poses.value == "first"


@pytest.mark.skipif(
    not Path(os.environ.get("BETTERHUMAN_MODELS_DIR", "models"), "smplx", "SMPLX_NEUTRAL.npz").exists(),
    reason="licensed SMPL-X weights absent",
)
def test_frames_layer_follows_frame_set() -> None:
    studio = HumanStudio(family="smplx", server=FakeServer(), data_dir=persistence.DEFAULT_ROOT)
    studio.frames.load("contacts")
    count = len(studio.session.frame_set)
    assert count > 0 and studio.layers.frames.poses.shape == (1, count, 7)
    studio.frames.add_at(studio.session.vertices[0].detach().numpy())
    assert studio.layers.frames.poses.shape == (1, count + 1, 7)


def test_model_switch_rebuilds_body_widgets(studio) -> None:
    for family, key in (("smplx", "smplx_neutral"), ("mhr", "mhr_lod1")):
        if family == "mhr" and not Path(os.environ.get("BETTERHUMAN_MODELS_DIR", "models"), "MHR").exists():
            continue
        studio.load_model(family, gender="neutral", lod=1)
        body = studio.session.body
        assert studio.session.model_key == key and body.spec.name == family
        groups = body.identity_groups
        assert set(studio.shape_sliders) == set(groups)
        assert all(len(studio.shape_sliders[g]) == min(w, 10) for g, w in groups.items())
        names = [layer.name for layer in studio.layers.layers.values()]
        assert len(names) == len(set(names)) and "frames" in names
        assert key in studio.status


def test_model_switch_on_real_viser_server(tmp_path: Path) -> None:
    """Regression: viser's tab-group remove() raised when the Model tab rebuilt the tabs."""
    viser = pytest.importorskip("viser")
    server = viser.ViserServer(host="127.0.0.1", port=8097, verbose=False)
    try:
        studio = HumanStudio(server=server, data_dir=tmp_path)
        studio.load_model("smpl", gender="neutral")
        assert studio.status.startswith("smpl_neutral loaded")
    finally:
        server.stop()
