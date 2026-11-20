"""The coupling standard (principle 1.3): BetterHuman's coupling test against BetterRobot's page, and
every line of ``docs/contracts/coupling-standard.md`` checked against the library.

The first tests check the BetterRobot lines BetterHuman relies on, against the BetterRobot it runs with, on small
models built with the builder calls BetterHuman's rigs make: joint order, batched model values, the ``q`` layout of
the joints the rigs add (mimic joints included), the state's transforms and poses, the Euler order, the packed
inertia of a mesh, the builder's names, ids and gravity, a revolute axis as data, a frame's parent joint and external
forces in joint frames. The BetterRobot functions BetterHuman calls are names, not lines of the standard:
``tests/contract/test_br_imports.py`` keeps every import off BetterRobot's private modules,
``tests/contract/test_studio_boundary.py`` keeps the studio on the viewer's public layer surface, and the tests under
``tests/terms/`` evaluate BetterHuman's ``optim`` terms.

The next tests check this repository's page on the random-weight SMPL model (full tier: its archive read stalls for
seconds on hosts with transparent huge pages) and on MHR and SOMA (release tier: licensed weights), each against
formulas written here (the scalar-last quaternion matrix) or BetterRobot's own state; all in float64. The last two read
the page: no line of its ``text`` blocks leaves it, and every line it adds is frozen in
``tests/contract/coupling_standard_lines.txt`` in the same commit (a run of whitespace counts as one space).
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Sequence
from functools import cache
from pathlib import Path

import pytest
import torch
from better_robot import Model, ModelBuilder, forward_kinematics, frame_jacobian, inverse_dynamics
from better_robot.exceptions import BetterRobotError, ShapeError
from better_robot.io import build_model
from better_robot.kinematics import difference, full_configuration, integrate
from better_robot.lie import so3
from better_robot.model import JointFixed, JointFreeFlyer, JointRevolute, JointSpherical
from better_robot.spatial import inertia_of_mesh

import better_human as bh
from better_human.io import weight_path
from better_human.io.formats import LATEST
from tests.support.random_archives import random_weight_spec

PAGE = Path(__file__).resolve().parents[2] / "docs" / "contracts" / "coupling-standard.md"
FROZEN = Path(__file__).parent / "coupling_standard_lines.txt"
F64 = torch.float64
UP = (0.0, -9.81, 0.0)
X, Y, Z = (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)
QUARTER_TURN_Z = (0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5))  # [qx, qy, qz, qw]
#: Native joints of a small branching body: ``(name, parent)``, parents first, as a family's tables list them.
NATIVE = (("pelvis", -1), ("left_hip", 0), ("right_hip", 0), ("spine", 0), ("left_knee", 1), ("neck", 3))
#: The same with a world anchor as native joint 0, as SOMA's ``Root``.
ANCHORED = (("Root", -1), ("Hips", 0), ("LeftLeg", 1), ("Spine", 1), ("LeftFoot", 2), ("Neck", 3))


def _tensor(*values: float) -> torch.Tensor:
    return torch.tensor(values, dtype=F64)


def _quaternion_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    """The rotations ``(..., 3, 3)`` of unit quaternions ``(..., 4)`` written ``[x, y, z, w]``."""
    x, y, z, w = quaternion.unbind(-1)
    rows = (
        (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
        (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
        (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
    )
    return torch.stack([torch.stack(row, dim=-1) for row in rows], dim=-2)


def _rotation(axis: Sequence[float], angle: float) -> torch.Tensor:
    """Rodrigues: the rotation ``(3, 3)`` by ``angle`` about the unit ``axis``."""
    x, y, z = axis
    skew = torch.tensor([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=F64)
    return torch.eye(3, dtype=F64) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * skew @ skew


def _native_body(native: tuple[tuple[str, int], ...] = NATIVE) -> Model:
    """A body as the SMPL rig builds it: a free-flyer root and a spherical joint per native joint, in native order,
    every body with mass, and a frame ``heel`` on the fifth native joint. A world anchor (``ANCHORED``) gets no body
    or joint, as in the SOMA rig."""
    builder = ModelBuilder("native", gravity=UP)
    names = [name for name, _ in native]
    root = 1 if native is ANCHORED else 0
    for name in names[root:]:
        builder.add_body(name, mass=1.0, com=_tensor(0.0, -0.1, 0.0), inertia=0.01 * torch.eye(3, dtype=F64))
    builder.add_free_flyer_root(names[root], child=names[root])
    for name, parent in native[root + 1 :]:
        builder.add_spherical(name, parent=names[parent], child=name, origin=_tensor(0.0, -0.2, 0.0, 0, 0, 0, 1))
    builder.add_frame("heel", parent=names[4], placement=_tensor(0.0, -0.4, 0.05, 0.0, 0.0, 0.0, 1.0))
    return build_model(builder.finalize(), dtype=F64)


def _channel_body() -> Model:
    """A root and one native joint as the MHR rig builds it: revolute channels z, y, x in a chain, the last a mimic of
    the first, then a fixed joint to a native joint without channels."""
    builder = ModelBuilder("channels", gravity=UP)
    for body in ("root", "node_z", "node_y", "thigh", "toe"):
        builder.add_body(body)
    builder.add_free_flyer_root("root", child="root", origin=_tensor(0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 1.0))
    builder.add_revolute("thigh_rz", parent="root", child="node_z", axis=Z, lower=-1.0, upper=1.0)
    builder.add_revolute("thigh_ry", parent="node_z", child="node_y", axis=Y)
    builder.add_revolute(
        "thigh_rx", parent="node_y", child="thigh", axis=X, mimic_source="thigh_rz", mimic_multiplier=0.5
    )
    builder.add_fixed("toe_fixed", parent="thigh", child="toe", origin=_tensor(0.0, -0.4, 0.0, 0.0, 0.0, 0.0, 1.0))
    return build_model(builder.finalize(), dtype=F64)


def _random_q(model: Model, *batch: int, seed: int) -> torch.Tensor:
    """``q (*batch, nq)``: ``q_neutral`` moved by a seeded random tangent."""
    generator = torch.Generator().manual_seed(seed)
    v = 0.4 * torch.randn(*batch, model.nv, generator=generator, dtype=F64)
    return integrate(model, model.q_neutral.expand(*batch, model.nq), v)


@pytest.mark.parametrize(("native", "offset"), [(NATIVE, 1), (ANCHORED, 0)], ids=["root first", "world anchor first"])
def test_joints_keep_the_order_the_rig_adds_them(native: tuple[tuple[str, int], ...], offset: int) -> None:
    """Joints come in the order the rig adds them, parents first, after the universe, each moving the body of its
    name: the SMPL rig's native joint ``j`` is joint ``j + 1``; with native joint 0 a world anchor (SOMA), native joint
    ``j`` is joint ``j`` and the anchor is the universe."""
    model = _native_body(native)
    names, parents = zip(*native, strict=True)
    added = names[1 - offset :]
    assert model.joint_names == ("universe", *added)
    assert model.body_names[1:] == added
    assert model.parent_joint_ids == (-1, *(parent + offset for parent in parents[1 - offset :]))
    assert (type(model.joints[1]), model.idx_qs[1], model.idx_vs[1]) == (JointFreeFlyer, 0, 0)


def test_identity_batch_broadcasts_against_input_batch() -> None:
    """Model values with ``I = (2,)`` against ``q`` with ``B = (3, 1)`` give ``(3, 2, ...)``, each slice one identity
    at one input; a ``B`` that does not broadcast with ``I`` raises ``ShapeError``."""
    model = _native_body()
    longer = model.joint_placements.clone()
    longer[model.joint_id("left_knee"), 1] -= 0.25
    heavier = model.body_inertias.clone()
    heavier[1:, 0] *= 2.0
    lower = model.frame_placements.clone()
    lower[model.frame_id("heel"), 1] -= 0.1
    batched = model.with_values(
        joint_placements=torch.stack((model.joint_placements, longer)),
        body_inertias=torch.stack((model.body_inertias, heavier)),
        frame_placements=torch.stack((model.frame_placements, lower)),
    )
    q = _random_q(model, 3, 1, seed=0)
    v, a = (torch.randn(3, 1, model.nv, generator=torch.Generator().manual_seed(seed), dtype=F64) for seed in (1, 2))
    state = forward_kinematics(batched, q)
    jacobian = frame_jacobian(batched, state, frame="heel")
    torque = inverse_dynamics(batched, state, v=v, a=a)
    assert state.joint_transforms.shape == (3, 2, model.njoints, 3, 4)
    assert state.frame_poses.shape == (3, 2, model.nframes, 7)
    assert (jacobian.shape, torque.shape) == ((3, 2, 6, model.nv), (3, 2, model.nv))
    for identity, values in enumerate(
        ({}, {"joint_placements": longer, "body_inertias": heavier, "frame_placements": lower})
    ):
        single = model.with_values(**values)
        for sample in range(3):
            one = forward_kinematics(single, q[sample, 0])
            torch.testing.assert_close(state.frame_poses[sample, identity], one.frame_poses)
            torch.testing.assert_close(jacobian[sample, identity], frame_jacobian(single, one, frame="heel"))
            expected = inverse_dynamics(single, one, v=v[sample, 0], a=a[sample, 0])
            torch.testing.assert_close(torque[sample, identity], expected)
    with pytest.raises(ShapeError):
        forward_kinematics(batched, q.expand(3, 5, model.nq))


def test_q_layout_of_the_joints_the_rigs_add() -> None:
    """Free flyer ``[t, qx, qy, qz, qw]`` 7/6, spherical ``[qx, qy, qz, qw]`` 4/3, revolute 1/1 about its axis, fixed
    0/0; a mimic joint has no public slice and takes ``multiplier * q_source`` in the full layout."""
    spherical = _native_body()
    knee = spherical.joint_id("left_knee")
    assert (spherical.nqs[1], spherical.nvs[1], spherical.nqs[knee], spherical.nvs[knee]) == (7, 6, 4, 3)
    q = spherical.q_neutral.clone()
    q[:7] = _tensor(0.3, -0.2, 0.5, *QUARTER_TURN_Z)
    q[spherical.idx_qs[knee] : spherical.idx_qs[knee] + 4] = _tensor(*QUARTER_TURN_Z)
    state = forward_kinematics(spherical, q)
    turn = _quaternion_matrix(_tensor(*QUARTER_TURN_Z))
    torch.testing.assert_close(state.joint_transforms[1], torch.cat((turn, _tensor(0.3, -0.2, 0.5)[:, None]), dim=-1))
    hip = spherical.parent_joint_ids[knee]
    torch.testing.assert_close(state.joint_rotations[knee], state.joint_rotations[hip] @ turn)

    channels = _channel_body()
    rz, ry, rx, toe = (channels.joint_id(name) for name in ("thigh_rz", "thigh_ry", "thigh_rx", "toe_fixed"))
    assert (channels.nq, channels.nv, channels.nq_full, channels.nv_full) == (9, 8, 10, 9)
    assert [channels.nqs[joint] for joint in (rz, ry, rx, toe)] == [1, 1, 0, 0]
    assert [channels.nqs_full[joint] for joint in (rz, ry, rx, toe)] == [1, 1, 1, 0]
    assert (channels.idx_qs[rz], channels.idx_vs[rz], channels.idx_qs[ry]) == (7, 6, 8)
    q = channels.q_neutral.clone()
    q[7], q[8] = 0.6, -0.4
    full = full_configuration(channels, q)
    torch.testing.assert_close(full[channels.idx_qs_full[rx]], _tensor(0.3)[0])
    state = forward_kinematics(channels, q)
    expected = _rotation(Z, 0.6) @ _rotation(Y, -0.4) @ _rotation(X, 0.3)
    torch.testing.assert_close(state.joint_rotations[rx], expected)
    step = torch.linspace(0.1, 0.8, channels.nv, dtype=F64)
    torch.testing.assert_close(difference(channels, q, integrate(channels, q, step)), step)


def test_the_state_holds_transforms_poses_and_translations() -> None:
    """``joint_transforms (..., nj, 3, 4)`` are ``[R | p]``; ``joint_poses (..., nj, 7)`` the same with a scalar-last
    quaternion; ``joint_translations`` their ``p``; the frame fields alike."""
    model = _native_body()
    state = forward_kinematics(model, _random_q(model, 2, seed=3))
    for kind, count in (("joint", model.njoints), ("frame", model.nframes)):
        transforms, poses = getattr(state, f"{kind}_transforms"), getattr(state, f"{kind}_poses")
        assert (transforms.shape, poses.shape) == ((2, count, 3, 4), (2, count, 7))
        torch.testing.assert_close(getattr(state, f"{kind}_translations"), transforms[..., 3])
        torch.testing.assert_close(getattr(state, f"{kind}_rotations"), transforms[..., :3])
        torch.testing.assert_close(poses[..., :3], transforms[..., 3])
        torch.testing.assert_close(_quaternion_matrix(poses[..., 3:]), transforms[..., :3])


def test_euler_angles_compose_rz_ry_rx() -> None:
    """``[roll, pitch, yaw]`` is ``Rz(yaw) Ry(pitch) Rx(roll)``, both ways: the MHR rig's channels and the studio's
    posing compose this order."""
    roll, pitch, yaw = 0.3, -0.5, 1.1
    rotation = _rotation(Z, yaw) @ _rotation(Y, pitch) @ _rotation(X, roll)
    torch.testing.assert_close(so3.to_matrix(so3.from_euler(_tensor(roll, pitch, yaw))), rotation)
    torch.testing.assert_close(so3.to_euler(so3.from_matrix(rotation)), _tensor(roll, pitch, yaw))


def test_the_inertia_of_a_mesh_is_packed_and_differentiable() -> None:
    """A unit cube of density 2 is ``[m, cx, cy, cz, Ixx, Iyy, Izz, Ixy, Ixz, Iyz] = [2, 0.5 (3), 1/3 (3), 0 (3)]``
    about its centre, and its mass has a gradient in the vertices (the mass tools integrate posed parts)."""
    corners = torch.tensor([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], dtype=F64, requires_grad=True)
    faces = torch.tensor(
        [[0, 2, 3], [0, 3, 1], [4, 5, 7], [4, 7, 6], [0, 1, 5], [0, 5, 4]]
        + [[2, 6, 7], [2, 7, 3], [0, 4, 6], [0, 6, 2], [1, 3, 7], [1, 7, 5]]
    )
    inertia = inertia_of_mesh(corners, faces, density=2.0)
    torch.testing.assert_close(inertia.detach(), _tensor(2.0, 0.5, 0.5, 0.5, 1 / 3, 1 / 3, 1 / 3, 0.0, 0.0, 0.0))
    inertia[0].backward()
    assert corners.grad is not None and bool(corners.grad.abs().sum() > 0)


def test_builder_keeps_names_resolves_ids_and_sets_gravity() -> None:
    """Names stay as given; joint, frame and body ids take an int or a name and check both; the builder's gravity is
    the linear part of ``gravity (6,)``."""
    model = _native_body()
    assert model.joint_names[1:] == tuple(name for name, _ in NATIVE)
    assert model.frame_names[-1] == "heel"
    torch.testing.assert_close(model.gravity, _tensor(*UP, 0.0, 0.0, 0.0))
    for kind in ("joint", "frame", "body"):
        resolve, names = getattr(model, f"{kind}_id"), getattr(model, f"{kind}_names")
        assert [resolve(name) for name in names] == [resolve(index) for index in range(len(names))]
        for bad in (len(names), "missing"):
            with pytest.raises(BetterRobotError):
                resolve(bad)


def test_a_revolute_axis_is_data() -> None:
    """A revolute joint carries its unit axis as a tuple of floats, which the studio reads to name a channel."""
    model = _channel_body()
    axes = [model.joints[model.joint_id(name)] for name in ("thigh_rz", "thigh_ry", "thigh_rx")]
    assert all(type(joint) is JointRevolute for joint in axes)
    assert [joint.axis for joint in axes] == [Z, Y, X]
    assert type(model.joints[model.joint_id("toe_fixed")]) is JointFixed
    assert type(_native_body().joints[2]) is JointSpherical


def test_frames_know_their_joint_and_external_forces_act_in_joint_frames() -> None:
    """``frame_parent_joint_ids`` names the joint whose body carries each frame; an external wrench ``[force, torque]``
    acts in its joint's frame: holding a turned floating body takes ``[R^T f, c x R^T f]``, and that wrench applied to
    it holds it."""
    model = _native_body()
    assert model.frame_parent_joint_ids[model.frame_id("heel")] == model.joint_id("left_knee")
    assert model.frame_parent_joint_ids[model.frame_id("spine")] == model.joint_id("spine")
    mass, com = 2.0, _tensor(0.3, 0.1, 0.0)
    builder = ModelBuilder("floating", gravity=UP)
    builder.add_body("body", mass=mass, com=com)
    builder.add_free_flyer_root("root", child="body")
    floating = build_model(builder.finalize(), dtype=F64)
    state = forward_kinematics(floating, _tensor(0.0, 0.0, 0.0, *QUARTER_TURN_Z))
    still = torch.zeros(6, dtype=F64)
    force = _quaternion_matrix(_tensor(*QUARTER_TURN_Z)).mT @ _tensor(0.0, mass * 9.81, 0.0)
    wrench = torch.cat((force, torch.linalg.cross(com, force)))
    torch.testing.assert_close(inverse_dynamics(floating, state, v=still, a=still), wrench)
    applied = torch.zeros(floating.njoints, 6, dtype=F64)
    applied[1] = wrench
    torch.testing.assert_close(inverse_dynamics(floating, state, v=still, a=still, external_forces=applied), still)


@cache
def _human(family: str) -> bh.HumanModel:
    """The family's body in float64 without inertias: SMPL from its random-weight archive, MHR LOD 6 and SOMA from the
    licensed weights."""
    specs: dict[str, Callable[[], bh.FamilySpec]] = {
        "smpl": lambda: random_weight_spec("smpl", dtype=F64),
        "mhr": lambda: bh.families.mhr(weight_path("MHR/converted", "mhr_lod6.npz"), lod=6, dtype=F64),
        "soma": lambda: bh.families.soma(weight_path("SOMA", "soma_mid.npz"), dtype=F64),
    }
    return bh.HumanModel.from_spec(specs[family](), density=None)


FAMILIES = [
    pytest.param("smpl", marks=pytest.mark.slow),
    pytest.param("mhr", marks=pytest.mark.licensed),
    pytest.param("soma", marks=pytest.mark.licensed),
]
#: MHR's authored inverse binds are float32 values: they carry its rest joints to 2.2e-7 m of the forward kinematics
#: (LOD 6, seed 4, measured 2026-11-10), as in ``tests/core/test_rig_properties.py``. The others compose exact inverses.
CARRY_TOLERANCES = {"mhr": {"rtol": 0.0, "atol": 1e-6}}


def _posed(family: str) -> bh.HumanState:
    """The family's zero identity at a seeded random configuration ``(2, nq)``."""
    human = _human(family)
    return human.bake().forward(_random_q(human.model, 2, seed=4))


@pytest.mark.parametrize("family", FAMILIES)
def test_the_root_is_a_free_flyer_placed_without_rotation(family: str) -> None:
    """Joint 1 is a free flyer whose placement has no rotation: adding ``d`` to ``q[..., :3]`` moves every joint,
    frame and vertex by ``d`` in the world, and ``q[..., 3:7]`` is joint 1's world rotation."""
    state = _posed(family)
    model = state.model
    assert (type(model.joints[1]), model.idx_qs[1]) == (JointFreeFlyer, 0)
    torch.testing.assert_close(model.joint_placements[1, 3:], _tensor(0.0, 0.0, 0.0, 1.0))
    torch.testing.assert_close(state.kinematics.joint_rotations[..., 1, :, :], _quaternion_matrix(state.q[..., 3:7]))
    shift = _tensor(0.2, 1.0, -0.3)
    q = state.q.clone()
    q[..., :3] += shift
    moved = state.identity.forward(q)
    torch.testing.assert_close(
        moved.kinematics.joint_translations[..., 1:, :], state.kinematics.joint_translations[..., 1:, :] + shift
    )
    torch.testing.assert_close(moved.frame_poses[..., 1:, :3], state.frame_poses[..., 1:, :3] + shift)
    torch.testing.assert_close(moved.vertices, state.vertices + shift)


@pytest.mark.parametrize("family", FAMILIES)
def test_gravity_points_down_y(family: str) -> None:
    """Every family's model carries ``gravity (0, -9.81, 0)``: y is up."""
    torch.testing.assert_close(_human(family).model.gravity, _tensor(*UP, 0.0, 0.0, 0.0))


@pytest.mark.parametrize("family", FAMILIES)
def test_joint_ids_name_the_body_of_each_native_joint(family: str) -> None:
    """``joint_ids (J,)``: joint ``joint_ids[j]`` moves the body named after native joint ``j``, or is joint 0 for a
    native world anchor (native joint 0, without a parent, at the identity in every state)."""
    human = _human(family)
    names, ids = human.spec.joint_names, human.joint_ids.tolist()
    assert len(ids) == human.spec.num_joints == len(names)
    anchors = [native for native, joint in enumerate(ids) if joint == 0]
    assert anchors in ([], [0])
    assert [human.model.body_names[ids[native]] for native in range(len(anchors), len(names))] == list(
        names[len(anchors) :]
    )
    if anchors:
        assert human.spec.parents[0] == -1
        torch.testing.assert_close(
            _posed(family).joint_transforms[..., 0, :, :], torch.eye(3, 4, dtype=F64).expand(2, 3, 4)
        )


@pytest.mark.parametrize("family", FAMILIES)
def test_state_joint_outputs_are_native_placements(family: str) -> None:
    """``joint_transforms (..., J, 3, 4)`` are rigid ``[R | p]`` in native order, ``joint_poses (..., J, 7)`` the same
    with a scalar-last quaternion, ``joint_translations (..., J, 3)`` BetterRobot's joint translations at
    ``joint_ids``."""
    state = _posed(family)
    human = state.identity.human
    count = human.spec.num_joints
    transforms, poses, translations = state.joint_transforms, state.joint_poses, state.joint_translations
    assert (transforms.shape, poses.shape, translations.shape) == ((2, count, 3, 4), (2, count, 7), (2, count, 3))
    rotations = transforms[..., :3]
    torch.testing.assert_close(rotations.mT @ rotations, torch.eye(3, dtype=F64).expand_as(rotations))
    torch.testing.assert_close(torch.linalg.det(rotations), torch.ones(2, count, dtype=F64))
    torch.testing.assert_close(translations, state.kinematics.joint_translations[..., human.joint_ids, :])
    torch.testing.assert_close(transforms[..., 3], translations)
    torch.testing.assert_close(poses[..., :3], translations)
    torch.testing.assert_close(_quaternion_matrix(poses[..., 3:]), rotations)


@pytest.mark.parametrize("family", FAMILIES)
def test_skin_affines_carry_the_rest_joints_to_the_world(family: str) -> None:
    """``skin_affines (..., J, 3, 4)`` applied to ``rest_joint_translations (..., J, 3)`` give
    ``joint_translations``."""
    state = _posed(family)
    affines, rest = state.skin_affines, state.rest_joint_translations
    count = state.identity.human.spec.num_joints
    assert (affines.shape, rest.shape) == ((2, count, 3, 4), (count, 3))
    carried = (affines[..., :3] @ rest[..., None]).squeeze(-1) + affines[..., 3]
    torch.testing.assert_close(carried, state.joint_translations, **CARRY_TOLERANCES.get(family, {}))


@pytest.mark.parametrize("family", FAMILIES)
def test_vertices_are_the_posed_mesh_in_vertex_order(family: str) -> None:
    """``vertices (..., V, 3)``: vertex ``i`` is the point ``i`` of the family's mesh that ``points`` poses alone."""
    state = _posed(family)
    count = state.identity.human.spec.num_vertices
    assert state.vertices.shape == (2, count, 3)
    some = torch.tensor([0, count // 3, count - 1])
    torch.testing.assert_close(state.points(some), state.vertices[..., some, :])


@pytest.mark.slow
def test_frame_poses_are_the_model_frames_with_the_contact_frames() -> None:
    """``frame_poses (..., nframes, 7)`` are the model's frames in frame order, contact frames among them; a contact
    frame sits on its vertex at the bind configuration."""
    spec = random_weight_spec("smpl", dtype=F64)
    human = bh.HumanModel.from_spec(spec, density=None, contact_frames={"heel": (3458, None), "tip": (412, "neck")})
    identity = human.bake()
    state = identity.forward(_random_q(human.model, 2, seed=5))
    assert state.frame_poses.shape == (2, human.model.nframes, 7)
    torch.testing.assert_close(state.frame_poses, state.kinematics.frame_poses)
    rest = identity.forward(human.model.q_neutral)
    for name, (vertex, _) in {"heel": (3458, None), "tip": (412, "neck")}.items():
        torch.testing.assert_close(rest.frame_poses[human.model.frame_id(name), :3], rest.vertices[vertex])


@pytest.mark.slow
def test_identity_batch_broadcasts_against_the_pose_batch() -> None:
    """Identity values ``I = (2,)`` against ``q`` with ``B = (3, 1)`` give ``(3, 2, ...)`` outputs, each slice one
    identity at one pose; a ``B`` that does not broadcast with ``I`` raises ``ShapeError``."""
    human = _human("smpl")
    betas = torch.randn(2, 10, generator=torch.Generator().manual_seed(6), dtype=F64)
    identity = human.bake(betas=betas)
    assert identity.model.joint_placements.shape == (2, human.model.njoints, 7)
    q = _random_q(human.model, 3, 1, seed=7)
    state = identity.forward(q)
    count, vertices = human.spec.num_joints, human.spec.num_vertices
    assert (state.joint_poses.shape, state.vertices.shape) == ((3, 2, count, 7), (3, 2, vertices, 3))
    for index in range(2):
        one = human.bake(betas=betas[index])
        for sample in range(3):
            alone = one.forward(q[sample, 0])
            torch.testing.assert_close(state.joint_poses[sample, index], alone.joint_poses)
            torch.testing.assert_close(state.vertices[sample, index], alone.vertices)
    with pytest.raises(ShapeError):
        identity.forward(q.expand(3, 5, human.model.nq))


#: Every format the page lists, with its direction.
FORMATS = {
    "better_human.mhr": "read",
    "better_human.soma": "read",
    "better_human.frame_set": "read, write",
    "better_human.segmentation": "read",
    "better_human.density_tables": "read",
    "better_human.studio.segmentation": "read, write",
    "better_human.studio.pose": "read, write",
}


def test_the_files_are_every_format_betterhuman_versions() -> None:
    """The page's formats are exactly those ``io.formats`` versions; ``test_formats.py`` checks each one's promise."""
    assert set(LATEST) == set(FORMATS)


def _page_lines() -> set[str]:
    """Every non-blank line of the page's ``text`` blocks, with each run of whitespace as one space."""
    blocks = re.findall(r"^```text\n(.*?)^```$", PAGE.read_text(), re.MULTILINE | re.DOTALL)
    return {" ".join(line.split()) for block in blocks for line in block.splitlines() if line.strip()}


def _frozen_lines() -> list[str]:
    return FROZEN.read_text().splitlines()


def test_page_only_grows() -> None:
    """No frozen line leaves the page: removing or changing a line of the standard is the owner's decision."""
    page = _page_lines()
    gone = [line for line in _frozen_lines() if line not in page]
    assert not gone, (
        "these lines of the coupling standard left docs/contracts/coupling-standard.md; put them back, or change the "
        "standard by the owner's decision in a commit that also edits coupling_standard_lines.txt:\n  "
        + "\n  ".join(gone)
    )


def test_lines_added_to_the_page_are_frozen() -> None:
    """A line added to the page is added to ``coupling_standard_lines.txt`` in the same commit."""
    added = sorted(_page_lines() - set(_frozen_lines()))
    assert not added, "add these lines to tests/contract/coupling_standard_lines.txt:\n  " + "\n  ".join(added)
