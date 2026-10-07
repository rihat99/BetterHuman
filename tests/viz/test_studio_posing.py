"""Joint handles and local-rotation posing for SMPL (free-flyer + spherical) and MHR (revolute channels)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import better_human as bh
from better_human.viz.studio.posing import apply_local_rotation, joint_handles
from better_human.viz.studio.state import StudioSession

MODELS = Path(__file__).parents[2] / "models"
SMPL_PATH = MODELS / "smpl" / "SMPL_NEUTRAL.npz"
MHR_PATH = MODELS / "MHR" / "converted" / "mhr_lod1.npz"


@pytest.fixture(scope="module")
def smpl() -> bh.Body:
    if not SMPL_PATH.is_file():
        pytest.skip(f"licensed SMPL fixture is absent: {SMPL_PATH}")
    return bh.load("smpl", SMPL_PATH)


@pytest.fixture(scope="module")
def mhr() -> bh.Body:
    if not MHR_PATH.is_file():
        pytest.skip(f"licensed MHR fixture is absent: {MHR_PATH}")
    return bh.load("mhr", MHR_PATH, lod=1)


def _rotated(world: torch.Tensor, quat: list[float]) -> torch.Tensor:
    """Return ``world (7,)`` with its rotation left-multiplied by scalar-last ``quat``."""
    unit = torch.tensor(quat, dtype=world.dtype)
    x1, y1, z1, w1 = unit / unit.norm()
    x2, y2, z2, w2 = world[3:]
    new = world.clone()
    new[3:] = torch.stack(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ]
    )
    return new


def _angle(quat_a: torch.Tensor, quat_b: torch.Tensor) -> float:
    return float(2 * torch.acos(torch.clamp((quat_a * quat_b).sum().abs(), max=1.0)))


def _pose_with(body: bh.Body, handle_name: str, quat: list[float]) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Rotate one joint by ``quat`` in the world; return ``(target (7,), posed (7,), native index)``."""
    identity = body.bake()
    handle = next(h for h in joint_handles(identity) if h.name == handle_name)
    q = identity.q_neutral.clone()
    world_old = identity.forward(q).joint_poses[handle.native_index]
    world_new = _rotated(world_old, quat)
    q_new = apply_local_rotation(q, handle, world_old, world_new)
    return world_new, identity.forward(q_new).joint_poses[handle.native_index], handle.native_index


def test_smpl_handle_table(smpl: bh.Body) -> None:
    identity = smpl.bake()
    handles = joint_handles(identity)
    assert len(handles) == len(smpl.joint_names)
    root, *rest = handles
    assert (root.kind, root.idx_q, root.native_index) == ("free_flyer", 0, 0)
    assert all(h.kind == "spherical" and not h.channels for h in rest)
    starts = [h.idx_q for h in rest]
    assert starts == list(range(7, 7 + 4 * len(rest), 4))
    assert 7 + 4 * len(rest) == identity.q_neutral.shape[-1]


def test_mhr_handle_table(mhr: bh.Body) -> None:
    handles = joint_handles(mhr.bake())
    assert 0 < len(handles) < len(mhr.joint_names), "fixed MHR joints must have no handle"
    assert [h.kind for h in handles].count("free_flyer") == 1
    channels = [h for h in handles if h.kind == "channels"]
    assert channels and all(h.idx_q == -1 and h.channels for h in channels)
    slots = [idx for h in channels for _, idx in h.channels]
    assert len(slots) == len(set(slots))
    assert all(axis in ("x", "y", "z") for h in channels for axis, _ in h.channels)
    assert len({h.native_index for h in handles}) == len(handles)


def test_smpl_spherical_round_trip(smpl: bh.Body) -> None:
    target, posed, _ = _pose_with(smpl, "left_elbow", [0.0, 0.0, 0.3827, 0.9239])
    assert _angle(target[3:], posed[3:]) < 1e-4
    torch.testing.assert_close(posed[:3], target[:3], atol=1e-5, rtol=0)


def test_smpl_free_flyer_rotation_only(smpl: bh.Body) -> None:
    identity = smpl.bake()
    root = joint_handles(identity)[0]
    q = identity.q_neutral.clone()
    world_old = identity.forward(q).joint_poses[0]
    world_new = _rotated(world_old, [0.0, 0.3827, 0.0, 0.9239])
    world_new[:3] += 0.5
    q_new = apply_local_rotation(q, root, world_old, world_new)
    posed = identity.forward(q_new).joint_poses[0]
    assert _angle(world_new[3:], posed[3:]) < 1e-4
    torch.testing.assert_close(posed[:3], world_old[:3], atol=1e-5, rtol=0)


def test_mhr_three_channel_round_trip(mhr: bh.Body) -> None:
    handle = next(h for h in joint_handles(mhr.bake()) if h.kind == "channels" and len(h.channels) == 3)
    target, posed, _ = _pose_with(mhr, handle.name, [0.1, -0.2, 0.15, 0.9618])
    assert _angle(target[3:], posed[3:]) < 1e-3


def test_mhr_missing_channels_touch_only_their_slots(mhr: bh.Body) -> None:
    identity = mhr.bake()
    handle = next(h for h in joint_handles(identity) if h.kind == "channels" and len(h.channels) < 3)
    q = identity.q_neutral.clone()
    world_old = identity.forward(q).joint_poses[handle.native_index]
    q_new = apply_local_rotation(q, handle, world_old, _rotated(world_old, [0.2, 0.2, 0.2, 0.9381]))
    changed = set(torch.nonzero(q_new != q).flatten().tolist())
    assert changed and changed <= {idx for _, idx in handle.channels}


def test_session_rotate_joint_and_move_root(smpl: bh.Body) -> None:
    session = StudioSession(smpl, "smpl")
    kinds: list[str] = []
    session.subscribe(kinds.append)
    elbow = next(h for h in session.handles if h.name == "left_elbow")
    old = session.joint_poses[elbow.native_index].clone()
    session.rotate_joint(elbow, old, _rotated(old, [0.0, 0.0, 0.3827, 0.9239]))
    assert _angle(_rotated(old, [0.0, 0.0, 0.3827, 0.9239])[3:], session.joint_poses[elbow.native_index, 3:]) < 1e-4
    root = session.handles[0]
    old_root = session.joint_poses[0].clone()
    moved = old_root.clone()
    moved[:3] += torch.tensor([0.1, 0.2, -0.3], dtype=moved.dtype)
    session.move_root(root, old_root, moved)
    torch.testing.assert_close(session.joint_poses[0, :3], moved[:3], atol=1e-5, rtol=0)
    assert kinds == ["pose", "pose"]
