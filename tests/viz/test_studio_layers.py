"""Per-joint skeleton radii, hidden weightless joints, and fast GPU skinning."""

from __future__ import annotations

import socket
from pathlib import Path

import pytest
import torch

pytest.importorskip("viser")

import viser  # noqa: E402

import better_human as bh  # noqa: E402
from better_human.core.mass import segment_vertices  # noqa: E402
from better_human.viz.studio.layers.base import JOINT_COLOR, SELECTED_JOINT_COLOR  # noqa: E402
from better_human.viz.studio.layers.mesh import MeshLayer, fast_skinning_available  # noqa: E402
from better_human.viz.studio.layers.skeleton import SkeletonLayer, _hidden_joints, _joint_radii  # noqa: E402
from better_human.viz.studio.state import StudioSession  # noqa: E402

MODELS_DIR = Path(__file__).parents[2] / "models"
SMPL_PATH = MODELS_DIR / "smpl" / "SMPL_NEUTRAL.npz"
SMPLX_PATH = MODELS_DIR / "smplx" / "SMPLX_NEUTRAL.npz"
MHR_PATH = MODELS_DIR / "MHR" / "converted" / "mhr_lod1.npz"


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("", 0))
        return probe.getsockname()[1]


def test_joint_radii_finger_smaller_than_hip() -> None:
    if not SMPLX_PATH.is_file():
        pytest.skip(f"licensed SMPL-X fixture is absent: {SMPLX_PATH}")
    body = bh.SMPLX(str(SMPLX_PATH), compute_mass=False)
    poses = body.native_joint_poses(body.fk(body.robot.q_neutral))
    radii = _joint_radii(poses[:, :3], body.structure.parents, base=0.02)
    names = list(body.structure.joint_names)
    finger = radii[names.index("left_index2")]
    hip = radii[names.index("left_hip")]
    assert finger < hip
    # Every radius stays inside the [0.25 * base, base] clamp.
    assert all(0.25 * 0.02 - 1e-9 <= radius <= 0.02 + 1e-9 for radius in radii)


def test_hidden_joints_mhr_includes_world_root_only_when_unposeable() -> None:
    if not MHR_PATH.is_file():
        pytest.skip(f"licensed MHR fixture is absent: {MHR_PATH}")
    body = bh.MHR(model_path=str(MHR_PATH), lod=1, compute_mass=False)
    session = StudioSession(body, "mhr_lod1")
    hidden = _hidden_joints(session)
    names = body.structure.joint_names
    # The disconnected world root is hidden; the pelvis-level free-flyer root is not.
    assert names.index("body_world") in hidden
    assert names.index("root") not in hidden
    # No poseable joint is ever hidden (hips ride on twist procs but stay clickable).
    poseable = {handle.native_index for handle in session.handles}
    assert not (hidden & poseable)
    assert names.index("l_upleg") not in hidden


def test_argmax_all_native_joint_part_count() -> None:
    if not MHR_PATH.is_file():
        pytest.skip(f"licensed MHR fixture is absent: {MHR_PATH}")
    body = bh.MHR(model_path=str(MHR_PATH), lod=1, compute_mass=False)
    winners = segment_vertices(body.values.skinning_weights, body.structure.skinning_indices)
    part_count = len(set(winners.tolist()))
    # Far more than the ~19 authored MHR groups (exposes all weighted joints).
    assert part_count > 19
    assert part_count == len({int(joint) for joint in winners.unique().tolist()})


@pytest.mark.parametrize("path,ctor", [
    (SMPL_PATH, lambda p: bh.SMPL(str(p), compute_mass=False)),
    (SMPLX_PATH, lambda p: bh.SMPLX(str(p), compute_mass=False)),
    (MHR_PATH, lambda p: bh.MHR(model_path=str(p), lod=1, compute_mass=False)),
], ids=["smpl", "smplx", "mhr"])
def test_fast_skinning_available(path: Path, ctor) -> None:
    if not path.is_file():
        pytest.skip(f"licensed fixture is absent: {path}")
    # SMPL/SMPL-X have no joint scales; MHR lod1's scales are the identity.
    assert fast_skinning_available(ctor(path))


def _color_of(layer: SkeletonLayer, native_index: int):
    return next(handle.color for handle, index in layer._spheres if index == native_index)


def test_finger_joints_uniform_small_radius() -> None:
    if not MHR_PATH.is_file():
        pytest.skip(f"licensed MHR fixture is absent: {MHR_PATH}")
    body = bh.MHR(model_path=str(MHR_PATH), lod=1, compute_mass=False)
    session = StudioSession(body, "mhr_lod1")
    server = viser.ViserServer(port=_free_port())
    try:
        layer = SkeletonLayer(lambda _i: None, base_radius=0.015)
        layer.build(server, session)
        names = list(body.structure.joint_names)
        knuckle = layer._radii[names.index("l_index1")]
        tip = layer._radii[names.index("l_index3")]
        thumb0 = layer._radii[names.index("l_thumb0")]
        expected = 0.30 * 0.015
        # All finger joints share one small radius regardless of bone length.
        assert knuckle == pytest.approx(expected)
        assert tip == pytest.approx(expected)
        assert thumb0 == pytest.approx(expected)
        # A non-finger limb joint keeps the (larger) bone-length rule.
        assert layer._radii[names.index("l_lowleg")] > expected
    finally:
        server.stop()


def test_selected_joint_highlight_transitions() -> None:
    if not SMPL_PATH.is_file():
        pytest.skip(f"licensed SMPL fixture is absent: {SMPL_PATH}")
    body = bh.SMPL(str(SMPL_PATH), compute_mass=False)
    session = StudioSession(body, "smpl_neutral")
    server = viser.ViserServer(port=_free_port())
    try:
        layer = SkeletonLayer(lambda _i: None, base_radius=0.02)
        layer.build(server, session)
        layer.set_selected(18)
        assert tuple(_color_of(layer, 18)) == SELECTED_JOINT_COLOR
        assert tuple(_color_of(layer, 5)) == JOINT_COLOR
        layer.set_selected(5)  # A -> B
        assert tuple(_color_of(layer, 5)) == SELECTED_JOINT_COLOR
        assert tuple(_color_of(layer, 18)) == JOINT_COLOR
        layer.set_selected(None)  # B -> None
        assert tuple(_color_of(layer, 5)) == JOINT_COLOR
        assert tuple(_color_of(layer, 18)) == JOINT_COLOR
    finally:
        server.stop()


def test_wireframe_overlay_lifecycle() -> None:
    if not SMPL_PATH.is_file():
        pytest.skip(f"licensed SMPL fixture is absent: {SMPL_PATH}")
    body = bh.SMPL(str(SMPL_PATH), compute_mass=False)
    session = StudioSession(body, "smpl_neutral")
    server = viser.ViserServer(port=_free_port())
    try:
        layer = MeshLayer()
        layer.build(server, session)
        # ON: a separate overlay appears; the solid base survives.
        layer.set_wireframe(True)
        assert layer._wire_overlay is not None
        assert layer._nodes  # solid base still present (not switched to wireframe)
        before = layer._wire_overlay.vertices.copy()
        session.set_q(body.robot.integrate(session.q, torch.full((body.robot.nv,), 0.2)))
        layer.update(session, "pose")
        assert not (layer._wire_overlay.vertices == before).all()  # tracks pose
        # Hidden with the mesh layer.
        layer.set_visible(False)
        assert layer._wire_overlay.visible is False
        layer.set_visible(True)
        # Fast mode greys it out: overlay removed while skinned mesh is active.
        layer.set_fast(True)
        assert layer._wire_overlay is None and layer._skinned is not None
        layer.set_fast(False)
        assert layer._wire_overlay is not None  # restored (checkbox stayed on)
        # OFF: overlay removed.
        layer.set_wireframe(False)
        assert layer._wire_overlay is None
    finally:
        server.stop()


def test_fast_skinned_mesh_uploads_and_deforms() -> None:
    if not SMPL_PATH.is_file():
        pytest.skip(f"licensed SMPL fixture is absent: {SMPL_PATH}")
    body = bh.SMPL(str(SMPL_PATH), compute_mass=False)
    session = StudioSession(body, "smpl_neutral")
    server = viser.ViserServer(port=_free_port())
    try:
        layer = MeshLayer()
        layer.build(server, session)
        layer.set_fast(True)
        assert layer.fast and layer._skinned is not None
        assert len(layer._skinned.bones) == len(body.structure.joint_names)
        before = layer._skinned.bones[1].position.copy()
        posed = body.robot.integrate(session.q, torch.full((body.robot.nv,), 0.2))
        session.set_q(posed)
        layer.update(session, "pose")
        after = layer._skinned.bones[1].position
        # A pose change only rewrites bone transforms; the bone moved.
        assert float(((torch.tensor(before) - torch.tensor(after)) ** 2).sum()) > 1e-8
    finally:
        server.stop()
