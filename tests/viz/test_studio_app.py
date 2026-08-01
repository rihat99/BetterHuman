"""Headless construction and session-driven behavior of the HumanStudio app."""

from __future__ import annotations

import socket
from pathlib import Path

import pytest
import torch

pytest.importorskip("viser")

from better_robot.lie import so3  # noqa: E402 - after importorskip guard

from better_human.viz.studio import app, frames_editor, segment_editor, state  # noqa: E402
from better_human.viz.studio.app import HumanStudio  # noqa: E402
from better_human.viz.studio.layers import FramesLayer, MeshLayer, PartsLayer, SkeletonLayer  # noqa: E402

MODELS_DIR = Path(__file__).parents[2] / "models"
SMPL_PATH = MODELS_DIR / "smpl" / "SMPL_NEUTRAL.npz"


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("", 0))
        return probe.getsockname()[1]


@pytest.fixture
def studio(tmp_path: Path):
    if not SMPL_PATH.is_file():
        pytest.skip(f"licensed SMPL fixture is absent: {SMPL_PATH}")
    instance = HumanStudio(model="smpl", port=_free_port(), data_dir=tmp_path, models_dir=MODELS_DIR)
    yield instance
    instance.stop()


def test_imports_without_running_server() -> None:
    # Module import (above, after importorskip) already proves the studio
    # surface imports without weights or a live server; confirm the symbols.
    assert hasattr(app, "HumanStudio")
    assert hasattr(state, "StudioSession")
    assert hasattr(segment_editor, "SegmentEditor")
    assert hasattr(frames_editor, "FramesEditor")
    assert all(cls is not None for cls in (MeshLayer, SkeletonLayer, FramesLayer, PartsLayer))


def test_layers_and_session_built(studio: HumanStudio) -> None:
    assert set(studio._layers) == {"mesh", "skeleton", "frames", "parts"}
    session = studio._session
    assert session.body.robot.nq == 99
    assert len(session.handles) == 24
    assert len(session.part_names) == 24
    assert session.total_mass is not None and session.total_mass > 0.0
    # The mesh and skeleton layers created scene nodes.
    assert studio._layers["mesh"]._nodes
    assert studio._layers["skeleton"]._spheres


def test_select_joint_attaches_and_detaches_gizmo(studio: HumanStudio) -> None:
    studio.select_joint(18)
    assert studio._gizmo is not None and studio._selected == 18
    studio.select_joint(None)
    assert studio._gizmo is None and studio._selected is None


def test_select_joint_highlights_sphere(studio: HumanStudio) -> None:
    from better_human.viz.studio.layers.base import JOINT_COLOR, SELECTED_JOINT_COLOR  # noqa: PLC0415

    skeleton = studio._layers["skeleton"]
    color_of = lambda idx: next(tuple(h.color) for h, i in skeleton._spheres if i == idx)  # noqa: E731
    studio.select_joint(18)
    assert color_of(18) == SELECTED_JOINT_COLOR
    studio.select_joint(None)
    assert color_of(18) == JOINT_COLOR


def test_frame_click_selects_and_syncs(studio: HumanStudio) -> None:
    frames = studio._layers["frames"]
    editor = studio._frames_editor
    name = next(iter(studio._session.frame_set))
    # A viewport frame-sphere click activates it and syncs the dropdown.
    studio._select_frame_from_click(name)
    assert frames._active == name
    assert editor._selected.value == name
    # Clicking the same frame again clears the highlight; the dropdown value stays.
    studio._select_frame_from_click(name)
    assert frames._active is None
    assert editor._selected.value == name


def test_frame_parent_joint_override_and_roundtrip(studio: HumanStudio) -> None:
    session = studio._session
    editor = studio._frames_editor
    frames = studio._layers["frames"]
    studio._set_layer_visible("frames", True)
    studio._set_layer_visible("skeleton", True)

    name = next(iter(session.frame_set))
    entry = session.frame_set[name]
    studio._select_frame_from_click(name)
    # Activating a frame shows the argmax label in the Parent-joint dropdown.
    assert editor._parent_joint.value.startswith("argmax (")

    joint_names = list(session.body.structure.joint_names)
    argmax_joint = session.resolved_joint(entry)
    other = (argmax_joint + 1) % len(joint_names)
    editor._parent_joint.value = joint_names[other]  # fires on_update -> set_frame_joint
    assert session.frame_set[name].joint == other
    # The connector for this frame now targets the overridden joint.
    targets = [nj for _, vid, nj in frames._connectors if vid == entry.vertex]
    assert targets == [other]

    # Save, then reload the saved set: the override survives the round-trip.
    editor._save_name.value = "override_set"
    editor._save()
    editor._source.options = editor._source_options()
    editor._source.value = "override_set"
    editor._load_source()
    assert session.frame_set[name].joint == other


def test_session_rotation_changes_vertices(studio: HumanStudio) -> None:
    session = studio._session
    before = session.vertices.clone()
    studio.select_joint(18)
    handle = next(item for item in session.handles if item.native_index == 18)
    world_old = session.native_poses[18].clone()
    world_new = world_old.clone()
    world_new[3:7] = so3.normalize(so3.compose(world_old[3:7], so3.exp(torch.tensor([0.3, -0.2, 0.1]))))
    session.rotate_joint(handle, world_old, world_new)
    assert (session.vertices - before).abs().max().item() > 1e-3
    # The gizmo is re-seated after the recompute and the server is still usable.
    assert studio._gizmo is not None
    assert studio._server.get_clients() == {}


def test_session_shape_change_changes_vertices(studio: HumanStudio) -> None:
    session = studio._session
    before = session.vertices.clone()
    session.set_shape(betas=session.shape_tensor("betas", [1.5] + [0.0] * 9))
    assert (session.vertices - before).abs().max().item() > 1e-3


def test_pose_save_and_load_roundtrip(studio: HumanStudio, tmp_path: Path) -> None:
    session = studio._session
    posed = session.body.robot.integrate(session.q, torch.full((session.body.robot.nv,), 0.1))
    session.set_q(posed)
    studio._pose_name.value = "roundtrip"
    studio._save_pose()
    session.reset_pose()
    assert not torch.allclose(session.q, posed)
    studio._pose_dropdown.value = "roundtrip"
    studio._load_pose()
    torch.testing.assert_close(session.q, posed)


def test_build_default_parts_topology(studio: HumanStudio) -> None:
    studio._ensure_parts_topology()
    assert len(studio._layers["parts"]._nodes) == len(studio._session.part_names)
