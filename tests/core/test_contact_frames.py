"""Contact frames baked onto SMPL-X: parents, rest placements, shape rebake."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from better_robot.lie import so3

import better_human as bh
from better_human.core.contact_frames import parse_frame_set, resolve_parents, vertex_normals

REPO_ROOT = Path(__file__).parents[2]
MODEL_PATH = REPO_ROOT / "models" / "smplx" / "SMPLX_NEUTRAL.npz"
CONTACTS_PATH = REPO_ROOT / "studio_data" / "frames" / "smplx_neutral" / "contacts.json"
SMPLX_BODY_FRAMES = 53


@pytest.fixture(scope="module")
def body() -> bh.SMPLX:
    """Neutral SMPL-X with hands, no face, and the authored contact frames."""
    if not MODEL_PATH.is_file():
        pytest.skip(f"licensed SMPL-X fixture is absent: {MODEL_PATH}")
    return bh.SMPLX(
        MODEL_PATH,
        use_face=False,
        use_hands=True,
        num_betas=10,
        contact_frames=CONTACTS_PATH,
    )


def _frame_poses(body: bh.SMPLX, q: torch.Tensor) -> torch.Tensor:
    """World poses ``(..., 12, 7)`` of the body's contact frames."""
    data = body.fk(q, compute_frames=True)
    frame_ids = torch.tensor(body.contact_frames.frame_ids, dtype=torch.long)
    return data.frame_pose_world.index_select(-2, frame_ids)


def test_frames_are_appended_in_authored_order(body: bh.SMPLX) -> None:
    authored = list(json.loads(CONTACTS_PATH.read_text()))
    contact_frames = body.contact_frames
    assert body.robot.nframes == SMPLX_BODY_FRAMES + len(authored)
    assert contact_frames.names == tuple(authored)
    assert body.robot.frame_names[SMPLX_BODY_FRAMES:] == tuple(authored)
    assert contact_frames.frame_ids == tuple(range(SMPLX_BODY_FRAMES, SMPLX_BODY_FRAMES + len(authored)))
    # The ids are assigned by formula at build time; they must match the robot's own table.
    assert contact_frames.frame_ids == tuple(body.robot.frame_name_to_id[name] for name in authored)


def test_explicit_joint_name_wins_over_argmax_skinning(body: bh.SMPLX) -> None:
    joint_names = body.structure.joint_names
    parents = dict(zip(body.contact_frames.names, body.contact_frames.parent_joints))
    vertices = dict(zip(body.contact_frames.names, body.contact_frames.vertex_ids))
    # "knee_left" is the one authored override, and it disagrees with argmax.
    assert joint_names[parents["knee_left"]] == "left_knee"
    assert joint_names[int(body.structure.part_ids[vertices["knee_left"]])] == "left_hip"


@pytest.mark.parametrize("name", ["toe_under_left", "heel_right", "palm_left", "finger_left"])
def test_null_joint_falls_back_to_argmax_skinning(body: bh.SMPLX, name: str) -> None:
    parents = dict(zip(body.contact_frames.names, body.contact_frames.parent_joints))
    vertices = dict(zip(body.contact_frames.names, body.contact_frames.vertex_ids))
    assert parents[name] == int(body.structure.part_ids[vertices[name]])


def test_rest_pose_frames_sit_on_their_vertex_with_the_normal_as_z(body: bh.SMPLX) -> None:
    poses = _frame_poses(body, body.robot.q_neutral)
    vertices = torch.tensor(body.contact_frames.vertex_ids, dtype=torch.long)
    # Both sides are pelvis-centered: q_neutral puts the free-flyer root at the origin.
    assert torch.allclose(poses[:, :3], body.values.v_shaped[vertices], atol=1e-5)
    normals = vertex_normals(body.values.v_shaped, body.structure.faces)[vertices]
    assert torch.allclose(so3.to_matrix(poses[:, 3:])[..., :, 2], normals, atol=1e-5)


def test_shape_rebake_moves_frames_onto_the_new_vertices(body: bh.SMPLX) -> None:
    torch.manual_seed(0)
    shaped = body.with_shape(betas=torch.randn(10) * 0.5)
    vertices = torch.tensor(shaped.contact_frames.vertex_ids, dtype=torch.long)
    targets = shaped.values.v_shaped[vertices]
    # The shape must actually move these vertices, or the test proves nothing.
    assert (targets - body.values.v_shaped[vertices]).abs().max() > 1e-3

    poses = _frame_poses(shaped, shaped.robot.q_neutral)
    assert torch.allclose(poses[:, :3], targets, atol=1e-5)
    normals = vertex_normals(shaped.values.v_shaped, shaped.structure.faces)[vertices]
    assert torch.allclose(so3.to_matrix(poses[:, 3:])[..., :, 2], normals, atol=1e-5)


def test_shape_rebake_is_batched(body: bh.SMPLX) -> None:
    torch.manual_seed(1)
    shaped = body.with_shape(betas=torch.randn(3, 10) * 0.5)
    assert shaped.robot.values.frame_placements.shape == (3, body.robot.nframes, 7)

    poses = _frame_poses(shaped, body.robot.q_neutral.expand(3, -1))
    vertices = torch.tensor(shaped.contact_frames.vertex_ids, dtype=torch.long)
    assert torch.allclose(poses[..., :3], shaped.values.v_shaped[:, vertices], atol=1e-5)


def test_body_without_contact_frames_keeps_only_body_frames() -> None:
    if not MODEL_PATH.is_file():
        pytest.skip(f"licensed SMPL-X fixture is absent: {MODEL_PATH}")
    plain = bh.SMPLX(MODEL_PATH, use_face=False, use_hands=True, num_betas=10, compute_mass=False)
    assert plain.contact_frames is None
    assert plain.robot.nframes == SMPLX_BODY_FRAMES
    assert plain.with_shape(betas=torch.zeros(10)).robot.nframes == SMPLX_BODY_FRAMES


def test_unknown_explicit_joint_name_raises(body: bh.SMPLX) -> None:
    with pytest.raises(ValueError, match="'tip'.*unknown parent joint 'not_a_joint'"):
        resolve_parents(
            {"tip": (0, "not_a_joint")},
            body.structure.joint_names,
            body.structure.part_ids,
        )
    with pytest.raises(ValueError, match="unknown parent joint"):
        bh.SMPLX(
            MODEL_PATH,
            use_face=False,
            use_hands=True,
            num_betas=10,
            compute_mass=False,
            contact_frames={"tip": (0, "not_a_joint")},
        )


def test_parse_frame_set_reads_the_authored_file() -> None:
    entries = parse_frame_set(CONTACTS_PATH, num_vertices=10475)
    assert entries["knee_left"] == (3673, "left_knee")
    assert entries["heel_right"] == (8712, None)
