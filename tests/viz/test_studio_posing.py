"""Handle tables and FK gizmo round-trips for the HumanStudio joint editor."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from better_robot.data_model.model_structure import JOINT_KIND_CODES
from better_robot.lie import so3

import better_human as bh
from better_human.viz.studio import posing

MODELS_DIR = Path(__file__).parents[2] / "models"
SMPL_PATH = MODELS_DIR / "smpl" / "SMPL_NEUTRAL.npz"
MHR_PATH = MODELS_DIR / "MHR" / "converted" / "mhr_lod1.npz"
_REVOLUTE_CODES = {JOINT_KIND_CODES[name] for name in ("revolute_rx", "revolute_ry", "revolute_rz")}


@pytest.fixture(scope="module")
def smpl_body() -> bh.SMPL:
    if not SMPL_PATH.is_file():
        pytest.skip(f"licensed SMPL fixture is absent: {SMPL_PATH}")
    return bh.SMPL(SMPL_PATH, compute_mass=False)


@pytest.fixture(scope="module")
def mhr_body() -> bh.MHR:
    if not MHR_PATH.is_file():
        pytest.skip(f"licensed MHR fixture is absent: {MHR_PATH}")
    return bh.MHR(model_path=MHR_PATH, lod=1, compute_mass=False)


def _seeded_pose(body: bh.ParametricBody, scale: float) -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    tangent = torch.randn(body.robot.nv, generator=generator) * scale
    return body.robot.integrate(body.robot.q_neutral, tangent)


def _perturbed_world(world: torch.Tensor, tangent: torch.Tensor) -> torch.Tensor:
    world_new = world.clone()
    world_new[3:7] = so3.normalize(so3.compose(world[3:7], so3.exp(tangent)))
    return world_new


def _rotation_error(quat_a: torch.Tensor, quat_b: torch.Tensor) -> float:
    return so3.log(so3.compose(so3.inverse(so3.normalize(quat_a)), so3.normalize(quat_b))).norm().item()


def _host_idx_qs(body: bh.MHR) -> set[int]:
    robot = body.robot
    codes = robot.structure.joint_kind_codes
    return {robot.idx_qs[jid] for jid in range(robot.njoints) if robot.nqs[jid] == 1 and codes[jid] in _REVOLUTE_CODES}


def test_smpl_handle_table(smpl_body: bh.SMPL) -> None:
    handles = posing.joint_handles(smpl_body)
    robot = smpl_body.robot
    assert len(handles) == len(smpl_body.structure.joint_names) == 24
    assert [handle.native_index for handle in handles] == list(range(24))
    root = handles[0]
    assert root.kind == "free_flyer" and root.idx_q == 0 and root.channels == ()
    for handle in handles[1:]:
        assert handle.kind == "spherical"
        assert handle.channels == ()
        branch = handle.native_index + 1
        assert handle.idx_q == robot.idx_qs[branch]
        assert robot.nqs[branch] == 4


def test_mhr_handle_table(mhr_body: bh.MHR) -> None:
    handles = posing.joint_handles(mhr_body)
    kinds = [handle.kind for handle in handles]
    assert kinds.count("free_flyer") == 1
    assert handles[0].kind == "free_flyer" and handles[0].native_index == 1
    assert len(handles) == 63
    assert kinds[1:] == ["channels"] * 62

    host_slots = _host_idx_qs(mhr_body)
    by_name = {handle.name: handle for handle in handles}
    for handle in handles[1:]:
        axes = [axis for axis, _ in handle.channels]
        assert axes and all(axis in {"x", "y", "z"} for axis in axes)
        for _, idx_q in handle.channels:
            assert idx_q in host_slots

    # Clean three-channel joint keeps the z, y, x application order.
    assert [axis for axis, _ in by_name["r_upleg"].channels] == ["z", "y", "x"]
    # A single-channel joint exposes exactly one host.
    assert [axis for axis, _ in by_name["l_transversetarsal"].channels] == ["x"]
    # A spine host stacks two hosts per axis (independent spine parameters).
    assert [axis for axis, _ in by_name["c_spine0"].channels] == ["z", "z", "y", "y", "x", "x"]


def test_smpl_spherical_round_trip(smpl_body: bh.SMPL) -> None:
    handles = {handle.name: handle for handle in posing.joint_handles(smpl_body)}
    handle = handles["left_elbow"]
    base = _seeded_pose(smpl_body, 0.1)
    world_old = smpl_body.native_joint_poses(smpl_body.fk(base))[handle.native_index]
    world_new = _perturbed_world(world_old, torch.tensor([0.3, -0.2, 0.15]))

    updated = posing.apply_local_rotation(base, handle, world_old, world_new)
    world_actual = smpl_body.native_joint_poses(smpl_body.fk(updated))[handle.native_index]
    assert _rotation_error(world_actual[3:7], world_new[3:7]) < 1e-5


def test_smpl_free_flyer_rotation_only(smpl_body: bh.SMPL) -> None:
    handles = posing.joint_handles(smpl_body)
    base = _seeded_pose(smpl_body, 0.1)
    world_old = smpl_body.native_joint_poses(smpl_body.fk(base))[0]
    world_new = _perturbed_world(world_old, torch.tensor([0.2, 0.1, -0.3]))

    updated = posing.apply_local_rotation(base, handles[0], world_old, world_new)
    world_actual = smpl_body.native_joint_poses(smpl_body.fk(updated))[0]
    assert _rotation_error(world_actual[3:7], world_new[3:7]) < 1e-5
    # A rotation-only free-flyer edit leaves the root position untouched.
    torch.testing.assert_close(world_actual[:3], world_old[:3], atol=1e-6, rtol=0)


def test_mhr_three_channel_round_trip(mhr_body: bh.MHR) -> None:
    handle = {handle.name: handle for handle in posing.joint_handles(mhr_body)}["r_upleg"]
    base = _seeded_pose(mhr_body, 0.05)
    world_old = mhr_body.native_joint_poses(mhr_body.fk(base))[handle.native_index]
    world_new = _perturbed_world(world_old, torch.tensor([0.2, -0.15, 0.1]))

    updated = posing.apply_local_rotation(base, handle, world_old, world_new)
    world_actual = mhr_body.native_joint_poses(mhr_body.fk(updated))[handle.native_index]
    assert _rotation_error(world_actual[3:7], world_new[3:7]) < 1e-5


def test_mhr_missing_channels_projection(mhr_body: bh.MHR) -> None:
    handle = {handle.name: handle for handle in posing.joint_handles(mhr_body)}["l_transversetarsal"]
    assert len(handle.channels) == 1 and handle.channels[0][0] == "x"
    base = _seeded_pose(mhr_body, 0.05)
    world_old = mhr_body.native_joint_poses(mhr_body.fk(base))[handle.native_index]
    # Perturb with all three components; only the x channel is reachable.
    world_new = _perturbed_world(world_old, torch.tensor([0.3, 0.2, -0.15]))

    updated = posing.apply_local_rotation(base, handle, world_old, world_new)
    world_actual = mhr_body.native_joint_poses(mhr_body.fk(updated))[handle.native_index]

    achieved = so3.log(so3.compose(so3.inverse(world_old[3:7]), world_actual[3:7]))
    # The achieved local rotation is pure-x: the unreachable y/z parts stay zero.
    assert abs(achieved[1].item()) < 1e-5
    assert abs(achieved[2].item()) < 1e-5

    # The reachable x-euler component matches the requested target.
    x_old = base[handle.channels[0][1]]
    rotation_target = so3.compose(
        so3.from_euler(torch.stack((x_old, x_old.new_zeros(()), x_old.new_zeros(())))),
        so3.exp(torch.tensor([0.3, 0.2, -0.15])),
    )
    x_target = so3.to_euler(rotation_target)[0]
    assert abs(achieved[0].item() - (x_target - x_old).item()) < 1e-5


def test_mhr_spine_host_mimics_follow(mhr_body: bh.MHR) -> None:
    handles = {handle.name: handle for handle in posing.joint_handles(mhr_body)}
    spine = handles["c_spine0"]
    child = handles["c_spine1"]
    base = _seeded_pose(mhr_body, 0.03)

    data = mhr_body.fk(base)
    vertices_before = mhr_body.vertices_from_data(data)
    spine_old = mhr_body.native_joint_poses(data)[spine.native_index]
    child_old = mhr_body.native_joint_poses(data)[child.native_index][3:7]
    spine_new = _perturbed_world(spine_old, torch.tensor([0.15, 0.1, -0.1]))

    updated = posing.apply_local_rotation(base, spine, spine_old, spine_new)
    data_new = mhr_body.fk(updated)
    vertices_after = mhr_body.vertices_from_data(data_new)
    spine_actual = mhr_body.native_joint_poses(data_new)[spine.native_index][3:7]
    child_new = mhr_body.native_joint_poses(data_new)[child.native_index][3:7]

    # The edit reshapes the mesh without error.
    assert (vertices_after - vertices_before).abs().max().item() > 1e-3
    # c_spine0 carries no own-chain mimic, so the equal split is exact.
    assert _rotation_error(spine_actual, spine_new[3:7]) < 1e-5
    # The spine host drives mimic joints, so c_spine1's world rotation follows.
    assert _rotation_error(child_old, child_new) > 1e-2
