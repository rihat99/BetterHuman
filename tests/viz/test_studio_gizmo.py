"""Coalesced gizmo drive: streaming many events must not drift the pose."""

from __future__ import annotations

import socket
from pathlib import Path

import pytest
import torch

pytest.importorskip("viser")

import viser  # noqa: E402
from better_robot.lie import so3  # noqa: E402

import better_human as bh  # noqa: E402
from better_human.viz.studio.gizmo import GizmoController  # noqa: E402
from better_human.viz.studio.state import StudioSession  # noqa: E402

MODELS_DIR = Path(__file__).parents[2] / "models"
SMPL_PATH = MODELS_DIR / "smpl" / "SMPL_NEUTRAL.npz"
MHR_PATH = MODELS_DIR / "MHR" / "converted" / "mhr_lod1.npz"


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("", 0))
        return probe.getsockname()[1]


def _rotation_error(quat_a: torch.Tensor, quat_b: torch.Tensor) -> float:
    return so3.log(so3.compose(so3.inverse(so3.normalize(quat_a)), so3.normalize(quat_b))).norm().item()


def _target_pose(world_old: torch.Tensor, tangent: torch.Tensor, translate: torch.Tensor) -> torch.Tensor:
    world_new = world_old.clone()
    world_new[3:7] = so3.normalize(so3.compose(world_old[3:7], so3.exp(tangent)))
    world_new[:3] = world_old[:3] + translate
    return world_new


def _interp(world_old: torch.Tensor, world_new: torch.Tensor, fraction: float) -> torch.Tensor:
    delta = so3.log(so3.compose(so3.inverse(world_old[3:7]), world_new[3:7]))
    pose = world_old.clone()
    pose[3:7] = so3.normalize(so3.compose(world_old[3:7], so3.exp(fraction * delta)))
    pose[:3] = world_old[:3] + fraction * (world_new[:3] - world_old[:3])
    return pose


def _single_jump_q(server, session, handle, native_pose, target) -> torch.Tensor:
    controller = GizmoController(server, session)
    try:
        controller.attach(handle, native_pose)
        controller.submit_pose(target, "end")
        assert controller.wait_idle()
        return session.q.clone()
    finally:
        controller.detach()
        controller.shutdown()


def _run_case(server, body, native_index: int, tangent, translate):
    session = StudioSession(body, "test")
    base_q = body.robot.q_neutral.clone()
    session.set_q(base_q.clone())
    handle = next(item for item in session.handles if item.native_index == native_index)
    native_pose = session.native_poses[native_index].clone()
    target = _target_pose(native_pose, tangent, translate)

    q_jump = _single_jump_q(server, session, handle, native_pose, target)

    # Stream 50 interpolated events, processing every one (no drift over a chain
    # of small deltas), then a rapid burst that the worker coalesces.
    session.set_q(base_q.clone())
    controller = GizmoController(server, session)
    try:
        controller.attach(handle, session.native_poses[native_index].clone())
        for step in range(1, 51):
            pose = _interp(native_pose, target, step / 50.0)
            controller.submit_pose(pose, "end" if step == 50 else "update")
            assert controller.wait_idle()
        q_sequential = session.q.clone()

        session.set_q(base_q.clone())
        controller.detach()
        controller.attach(handle, session.native_poses[native_index].clone())
        for step in range(1, 51):
            pose = _interp(native_pose, target, step / 50.0)
            controller.submit_pose(pose, "end" if step == 50 else "update")
        assert controller.wait_idle()
        q_burst = session.q.clone()
        achieved = session.native_poses[native_index].clone()
    finally:
        controller.detach()
        controller.shutdown()

    # No drift: a 50-step chain and a coalesced burst both land on the single
    # jump exactly (up to float32 SO(3) round-off).
    torch.testing.assert_close(q_sequential, q_jump, atol=1e-4, rtol=0)
    torch.testing.assert_close(q_burst, q_jump, atol=1e-4, rtol=0)
    return achieved, target, handle


@pytest.fixture(scope="module")
def server():
    instance = viser.ViserServer(port=_free_port())
    yield instance
    instance.stop()


@pytest.mark.parametrize("native_index,translate", [(0, [0.0, 0.0, 0.0]), (22, [0.0, 0.0, 0.0])],
                         ids=["smpl-root", "smpl-hand"])
def test_smpl_gizmo_stream_no_drift(server, native_index, translate) -> None:
    if not SMPL_PATH.is_file():
        pytest.skip(f"licensed SMPL fixture is absent: {SMPL_PATH}")
    body = bh.SMPL(str(SMPL_PATH), compute_mass=False)
    tangent = torch.tensor([0.25, -0.15, 0.1])
    translate_t = torch.tensor(translate) if native_index != 0 else torch.tensor([0.1, -0.05, 0.08])
    achieved, target, handle = _run_case(server, body, native_index, tangent, translate_t)
    # Root and spherical joints reach the requested world rotation exactly.
    assert _rotation_error(achieved[3:7], target[3:7]) < 1e-3
    if handle.kind == "free_flyer":
        torch.testing.assert_close(achieved[:3], target[:3], atol=1e-4, rtol=0)


@pytest.mark.parametrize("joint_name,translate", [("root", [0.12, 0.0, -0.06]), ("l_index1", [0.0, 0.0, 0.0])],
                         ids=["mhr-root", "mhr-finger"])
def test_mhr_gizmo_stream_no_drift(server, joint_name, translate) -> None:
    if not MHR_PATH.is_file():
        pytest.skip(f"licensed MHR fixture is absent: {MHR_PATH}")
    body = bh.MHR(model_path=str(MHR_PATH), lod=1, compute_mass=False)
    native_index = body.structure.joint_names.index(joint_name)
    tangent = torch.tensor([0.2, -0.1, 0.05])
    achieved, target, handle = _run_case(server, body, native_index, tangent, torch.tensor(translate))
    if handle.kind == "free_flyer":
        assert _rotation_error(achieved[3:7], target[3:7]) < 1e-3
        torch.testing.assert_close(achieved[:3], target[:3], atol=1e-3, rtol=0)


def _stream_root_rotation(server, body, degrees: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    session = StudioSession(body, "root-pivot")
    root = next(item for item in session.handles if item.kind == "free_flyer")
    pelvis_index = root.native_index
    anchor = session.native_poses[pelvis_index].clone()
    pelvis_before = anchor[:3].clone()
    hand_index = _hand_index(body)
    hand_before = session.native_poses[hand_index, :3].clone()
    controller = GizmoController(server, session)
    try:
        controller.attach(root, anchor.clone())
        radians = degrees * torch.pi / 180.0
        for step in range(1, 21):
            angle = (step / 20.0) * radians
            pose = anchor.clone()
            pose[3:7] = so3.normalize(so3.compose(so3.exp(torch.tensor([0.0, 0.0, angle])), anchor[3:7]))
            controller.submit_pose(pose, "end" if step == 20 else "update")
            assert controller.wait_idle()
        pelvis_after = session.native_poses[pelvis_index, :3].clone()
        hand_after = session.native_poses[hand_index, :3].clone()
    finally:
        controller.detach()
        controller.shutdown()
    return pelvis_before, pelvis_after, hand_after - hand_before


def _hand_index(body) -> int:
    names = body.structure.joint_names
    for candidate in ("l_wrist", "left_wrist", "left_hand"):
        if candidate in names:
            return names.index(candidate)
    raise AssertionError("no hand joint found")


def test_mhr_root_rotation_pivots_at_pelvis(server) -> None:
    if not MHR_PATH.is_file():
        pytest.skip(f"licensed MHR fixture is absent: {MHR_PATH}")
    body = bh.MHR(model_path=str(MHR_PATH), lod=1, compute_mass=False)
    pelvis_before, pelvis_after, hand_delta = _stream_root_rotation(server, body, 30.0)
    # Pelvis at ~0.92 m stays put; the hand swings about it.
    assert float(pelvis_before[1]) > 0.5
    torch.testing.assert_close(pelvis_after, pelvis_before, atol=1e-5, rtol=0)
    assert float(hand_delta.norm()) > 0.05


def test_smpl_root_rotation_pivots_at_pelvis(server) -> None:
    if not SMPL_PATH.is_file():
        pytest.skip(f"licensed SMPL fixture is absent: {SMPL_PATH}")
    body = bh.SMPL(str(SMPL_PATH), compute_mass=False)
    pelvis_before, pelvis_after, hand_delta = _stream_root_rotation(server, body, 30.0)
    torch.testing.assert_close(pelvis_after, pelvis_before, atol=1e-5, rtol=0)
    assert float(hand_delta.norm()) > 0.05


def test_worker_survives_apply_exception(server) -> None:
    if not SMPL_PATH.is_file():
        pytest.skip(f"licensed SMPL fixture is absent: {SMPL_PATH}")
    body = bh.SMPL(str(SMPL_PATH), compute_mass=False)
    session = StudioSession(body, "smpl")
    messages: list[str] = []
    controller = GizmoController(server, session, messages.append)
    try:
        handle = next(item for item in session.handles if item.native_index == 18)
        native_pose = session.native_poses[18].clone()
        controller.attach(handle, native_pose.clone())

        # Inject a fault into the first apply only.
        original = session.rotate_joint
        calls = {"count": 0}

        def flaky(*args, **kwargs):
            calls["count"] += 1
            if calls["count"] == 1:
                raise RuntimeError("injected apply fault")
            return original(*args, **kwargs)

        session.rotate_joint = flaky
        before = session.q.clone()

        bad = native_pose.clone()
        bad[3:7] = so3.normalize(so3.compose(native_pose[3:7], so3.exp(torch.tensor([0.2, 0.0, 0.0]))))
        controller.submit_pose(bad, "update")
        assert controller.wait_idle()
        # The faulted edit did not apply, the worker is still alive, and it
        # surfaced a one-line status message.
        assert controller._worker.is_alive()
        torch.testing.assert_close(session.q, before)
        assert any("gizmo edit failed" in message for message in messages)

        # A subsequent valid edit still lands: the worker recovered.
        good = native_pose.clone()
        good[3:7] = so3.normalize(so3.compose(native_pose[3:7], so3.exp(torch.tensor([0.0, 0.3, 0.0]))))
        controller.submit_pose(good, "update")
        assert controller.wait_idle()
        assert calls["count"] >= 2
        assert not torch.allclose(session.q, before)
    finally:
        controller.detach()
        controller.shutdown()
