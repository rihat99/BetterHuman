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


@pytest.fixture(scope="module")
def body() -> bh.Body:
    """Neutral SMPL-X with hands, no face, and the authored contact frames."""
    if not MODEL_PATH.is_file():
        pytest.skip(f"licensed SMPL-X fixture is absent: {MODEL_PATH}")
    return bh.Body(bh.families.smplx(MODEL_PATH, face=False, num_betas=10), contact_frames=CONTACTS_PATH)


def _frame_poses(identity: bh.Identity) -> torch.Tensor:
    """World poses ``(..., 12, 7)`` of the body's contact frames at the identity's neutral pose."""
    frame_ids = torch.tensor(identity.body.contact_frames.frame_ids, dtype=torch.long)
    return identity.forward(identity.q_neutral).frame_poses.index_select(-2, frame_ids)


def test_frames_are_appended_in_authored_order(body: bh.Body) -> None:
    authored = list(json.loads(CONTACTS_PATH.read_text()))
    contact_frames = body.contact_frames
    assert body.model.nframes == body.model.nbodies + len(authored)
    assert contact_frames.names == tuple(authored)
    assert body.model.frame_names[body.model.nbodies:] == tuple(authored)
    assert contact_frames.frame_ids == tuple(range(body.model.nbodies, body.model.nbodies + len(authored)))
    # The ids are assigned by formula at build time; they must match the robot's own table.
    assert contact_frames.frame_ids == tuple(body.model.frame_id(name) for name in authored)


def test_explicit_joint_name_wins_over_argmax_skinning(body: bh.Body) -> None:
    joint_names = body.joint_names
    parents = dict(zip(body.contact_frames.names, body.contact_frames.parent_joints))
    vertices = dict(zip(body.contact_frames.names, body.contact_frames.vertex_ids))
    # "knee_left" is the one authored override, and it disagrees with argmax.
    assert joint_names[parents["knee_left"]] == "left_knee"
    assert joint_names[int(body.spec.parts.ids[vertices["knee_left"]])] == "left_hip"


@pytest.mark.parametrize("name", ["toe_under_left", "heel_right", "palm_left", "finger_left"])
def test_null_joint_falls_back_to_argmax_skinning(body: bh.Body, name: str) -> None:
    parents = dict(zip(body.contact_frames.names, body.contact_frames.parent_joints))
    vertices = dict(zip(body.contact_frames.names, body.contact_frames.vertex_ids))
    assert parents[name] == int(body.spec.parts.ids[vertices[name]])


def test_rest_pose_frames_sit_on_their_vertex_with_the_normal_as_z(body: bh.Body) -> None:
    identity = body.bake()
    poses = _frame_poses(identity)
    vertices = torch.tensor(body.contact_frames.vertex_ids, dtype=torch.long)
    # Both sides are pelvis-centered: q_neutral puts the free-flyer root at the origin.
    assert torch.allclose(poses[:, :3], identity.v_shaped[vertices], atol=1e-5)
    normals = vertex_normals(identity.v_shaped, body.spec.faces)[vertices]
    assert torch.allclose(so3.to_matrix(poses[:, 3:])[..., :, 2], normals, atol=1e-5)


def test_shape_rebake_moves_frames_onto_the_new_vertices(body: bh.Body) -> None:
    torch.manual_seed(0)
    shaped = body.bake(betas=torch.randn(10) * 0.5)
    vertices = torch.tensor(body.contact_frames.vertex_ids, dtype=torch.long)
    targets = shaped.v_shaped[vertices]
    # The shape must actually move these vertices, or the test proves nothing.
    assert (targets - body.bake().v_shaped[vertices]).abs().max() > 1e-3

    poses = _frame_poses(shaped)
    assert torch.allclose(poses[:, :3], targets, atol=1e-5)
    normals = vertex_normals(shaped.v_shaped, body.spec.faces)[vertices]
    assert torch.allclose(so3.to_matrix(poses[:, 3:])[..., :, 2], normals, atol=1e-5)


def test_shape_rebake_is_batched(body: bh.Body) -> None:
    torch.manual_seed(1)
    shaped = body.bake(betas=torch.randn(3, 10) * 0.5)
    assert shaped.model.values.frame_placements.shape == (3, body.model.nframes, 7)

    poses = _frame_poses(shaped)
    vertices = torch.tensor(body.contact_frames.vertex_ids, dtype=torch.long)
    assert torch.allclose(poses[..., :3], shaped.v_shaped[:, vertices], atol=1e-5)


def test_body_without_contact_frames_keeps_only_body_frames() -> None:
    if not MODEL_PATH.is_file():
        pytest.skip(f"licensed SMPL-X fixture is absent: {MODEL_PATH}")
    plain = bh.Body(bh.families.smplx(MODEL_PATH, face=False, num_betas=10), mass=False)
    assert plain.contact_frames is None
    assert plain.model.nframes == plain.model.nbodies
    assert plain.bake(betas=torch.zeros(10)).model.nframes == plain.model.nbodies


def test_unknown_explicit_joint_name_raises(body: bh.Body) -> None:
    with pytest.raises(ValueError, match="'tip'.*unknown parent joint 'not_a_joint'"):
        resolve_parents(
            {"tip": (0, "not_a_joint")},
            body.joint_names,
            body.spec.parts.ids,
        )
    with pytest.raises(ValueError, match="unknown parent joint"):
        bh.Body(bh.families.smplx(MODEL_PATH, face=False), mass=False, contact_frames={"tip": (0, "not_a_joint")})


def test_parse_frame_set_reads_the_authored_file() -> None:
    entries = parse_frame_set(CONTACTS_PATH, num_vertices=10475)
    assert entries["knee_left"] == (3673, "left_knee")
    assert entries["heel_right"] == (8712, None)


def test_rig_corrective_features_receives_the_baked_identity(body: bh.Body, monkeypatch: pytest.MonkeyPatch) -> None:
    rig_type = type(body.spec.rig)
    original = rig_type.corrective_features
    seen = []

    def spy(self, spec, model, q, *, identity=None):  # noqa: ANN001, ANN202
        seen.append(identity)
        return original(self, spec, model, q, identity=identity)

    monkeypatch.setattr(rig_type, "corrective_features", spy)
    identity = body.bake()
    state = identity.forward(identity.q_neutral)
    assert state.vertices.shape[-2:] == (body.spec.num_vertices, 3)
    assert len(seen) == 1 and seen[0] is identity.params
