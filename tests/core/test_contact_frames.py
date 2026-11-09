"""Contact frames baked onto SMPL-X: parents, rest placements, shape rebake."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from better_robot.lie import so3

import better_human as bh
from better_human.core.contact_frames import _resolve_parents, _vertex_normals
from better_human.io import weight_path

REPO_ROOT = Path(__file__).parents[2]
CONTACTS_PATH = REPO_ROOT / "tests" / "support" / "contacts.json"


@pytest.fixture(scope="module")
def body() -> bh.HumanModel:
    """Neutral SMPL-X with hands, no face, and the authored contact frames."""
    return bh.HumanModel.from_spec(
        bh.families.smplx(weight_path("smplx", "SMPLX_NEUTRAL.npz"), face=False, num_betas=10),
        contact_frames=bh.io.read_frame_set(CONTACTS_PATH),
    )


def _frame_poses(identity: bh.Identity) -> torch.Tensor:
    """World poses ``(..., 12, 7)`` of the body's contact frames at the identity's neutral pose."""
    frame_ids = torch.tensor(identity.human.contact_frames.frame_ids, dtype=torch.long)
    return identity.forward(identity.model.q_neutral).frame_poses.index_select(-2, frame_ids)


@pytest.mark.licensed
def test_frames_are_appended_in_authored_order(body: bh.HumanModel) -> None:
    authored = list(json.loads(CONTACTS_PATH.read_text()))
    contact_frames = body.contact_frames
    assert body.model.nframes == body.model.nbodies + len(authored)
    assert contact_frames.names == tuple(authored)
    assert body.model.frame_names[body.model.nbodies :] == tuple(authored)
    assert contact_frames.frame_ids == tuple(range(body.model.nbodies, body.model.nbodies + len(authored)))
    # The ids are assigned by formula at build time; they must match the robot's own table.
    assert contact_frames.frame_ids == tuple(body.model.frame_id(name) for name in authored)


@pytest.mark.licensed
def test_explicit_joint_name_wins_over_argmax_skinning(body: bh.HumanModel) -> None:
    joint_names = body.spec.joint_names
    parents = dict(zip(body.contact_frames.names, body.contact_frames.parent_joints))
    vertices = dict(zip(body.contact_frames.names, body.contact_frames.vertex_ids))
    # "knee_left" is the one authored override, and it disagrees with argmax.
    assert joint_names[parents["knee_left"]] == "left_knee"
    assert joint_names[int(body.spec.parts.ids[vertices["knee_left"]])] == "left_hip"


@pytest.mark.licensed
@pytest.mark.parametrize("name", ["toe_under_left", "heel_right", "palm_left", "finger_left"])
def test_null_joint_falls_back_to_argmax_skinning(body: bh.HumanModel, name: str) -> None:
    parents = dict(zip(body.contact_frames.names, body.contact_frames.parent_joints))
    vertices = dict(zip(body.contact_frames.names, body.contact_frames.vertex_ids))
    assert parents[name] == int(body.spec.parts.ids[vertices[name]])


@pytest.mark.licensed
def test_rest_pose_frames_sit_on_their_vertex_with_the_normal_as_z(body: bh.HumanModel) -> None:
    identity = body.bake()
    poses = _frame_poses(identity)
    vertices = torch.tensor(body.contact_frames.vertex_ids, dtype=torch.long)
    # Both sides are pelvis-centered: q_neutral puts the free-flyer root at the origin.
    assert torch.allclose(poses[:, :3], identity.rest_vertices[vertices], atol=1e-5)
    normals = _vertex_normals(identity.rest_vertices, body.spec.faces)[vertices]
    assert torch.allclose(so3.to_matrix(poses[:, 3:])[..., :, 2], normals, atol=1e-5)


@pytest.mark.licensed
def test_shape_rebake_moves_frames_onto_the_new_vertices(body: bh.HumanModel) -> None:
    torch.manual_seed(0)
    shaped = body.bake(betas=torch.randn(10) * 0.5)
    vertices = torch.tensor(body.contact_frames.vertex_ids, dtype=torch.long)
    targets = shaped.rest_vertices[vertices]
    # The shape must actually move these vertices, or the test proves nothing.
    assert (targets - body.bake().rest_vertices[vertices]).abs().max() > 1e-3

    poses = _frame_poses(shaped)
    assert torch.allclose(poses[:, :3], targets, atol=1e-5)
    normals = _vertex_normals(shaped.rest_vertices, body.spec.faces)[vertices]
    assert torch.allclose(so3.to_matrix(poses[:, 3:])[..., :, 2], normals, atol=1e-5)


@pytest.mark.licensed
def test_shape_rebake_is_batched(body: bh.HumanModel) -> None:
    torch.manual_seed(1)
    shaped = body.bake(betas=torch.randn(3, 10) * 0.5)
    assert shaped.model.frame_placements.shape == (3, body.model.nframes, 7)

    poses = _frame_poses(shaped)
    vertices = torch.tensor(body.contact_frames.vertex_ids, dtype=torch.long)
    assert torch.allclose(poses[..., :3], shaped.rest_vertices[:, vertices], atol=1e-5)


@pytest.mark.licensed
def test_body_without_contact_frames_keeps_only_body_frames() -> None:
    plain = bh.HumanModel.from_spec(
        bh.families.smplx(weight_path("smplx", "SMPLX_NEUTRAL.npz"), face=False, num_betas=10), density=None
    )
    assert plain.contact_frames is None
    assert plain.model.nframes == plain.model.nbodies
    assert plain.bake(betas=torch.zeros(10)).model.nframes == plain.model.nbodies


@pytest.mark.licensed
def test_unknown_explicit_joint_name_raises(body: bh.HumanModel) -> None:
    with pytest.raises(ValueError, match="'tip'.*unknown parent joint 'not_a_joint'"):
        _resolve_parents(
            {"tip": (0, "not_a_joint")},
            body.spec.joint_names,
            body.spec.parts.ids,
        )
    with pytest.raises(ValueError, match="unknown parent joint"):
        bh.HumanModel.from_spec(
            bh.families.smplx(weight_path("smplx", "SMPLX_NEUTRAL.npz"), face=False),
            density=None,
            contact_frames={"tip": (0, "not_a_joint")},
        )


@pytest.mark.parametrize("vertex", [-1, 4, True, np.int64(-1)], ids=["negative", "past_the_mesh", "bool", "numpy"])
def test_resolve_parents_refuses_a_vertex_outside_the_mesh(vertex: int) -> None:
    # The frame-set reader cannot know the mesh; the first place that does checks every id.
    with pytest.raises(ValueError, match=r"'tip': vertex id .* is not an int in \[0, 4\)"):
        _resolve_parents({"tip": (vertex, None)}, ["root"], torch.zeros(4, dtype=torch.long))


def test_resolve_parents_takes_any_integer_vertex_id() -> None:
    part_ids = torch.tensor([0, 1, 1, 0])
    assert _resolve_parents({"a": (np.int64(1), None), "b": (np.int32(3), "j1")}, ["j0", "j1"], part_ids) == [1, 1]


@pytest.mark.licensed
def test_rig_corrective_features_receives_the_baked_identity(
    body: bh.HumanModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    rig_type = type(body.spec.rig)
    original = rig_type.corrective_features
    seen = []

    def spy(self, spec, model, *, q, identity):  # noqa: ANN001, ANN202
        seen.append(identity)
        return original(self, spec, model, q=q, identity=identity)

    monkeypatch.setattr(rig_type, "corrective_features", spy)
    identity = body.bake()
    state = identity.forward(identity.model.q_neutral)
    assert state.vertices.shape[-2:] == (body.spec.num_vertices, 3)
    assert len(seen) == 1 and seen[0] is identity.values
