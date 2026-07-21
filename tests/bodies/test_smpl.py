"""SMPL composition, conversion, batching, and differentiability tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import better_human as bh
import better_human.core.body as body_module
from better_human.bodies import SMPLClassic
from better_human.bodies.config_smpl import SMPL_CONFIG

MODEL_PATH = Path(__file__).parents[2] / "models" / "smpl" / "SMPL_NEUTRAL.npz"
GRADCHECK_VERTEX_INDICES = torch.tensor([0, 1723, 3445, 5167, 6889], dtype=torch.long)


@pytest.fixture(scope="module")
def body() -> bh.SMPL:
    """Load the licensed local neutral fixture when available."""
    if not MODEL_PATH.is_file():
        pytest.skip(f"licensed SMPL fixture is absent: {MODEL_PATH}")
    return bh.SMPL(MODEL_PATH, compute_mass=False)


def test_registry_and_preserved_native_alignment(body: bh.SMPL) -> None:
    loaded = bh.load("smpl", model_path=MODEL_PATH, compute_mass=False)
    assert isinstance(loaded, bh.SMPL)
    assert body.robot.body_names[1:] == body.structure.joint_names
    assert body.robot.joint_names[1:] == ("root", *body.structure.joint_names[1:])
    assert body.robot.njoints == len(body.structure.joint_names) + 1


def test_with_shape_is_functional_batched_and_broadcasts_q(body: bh.SMPL) -> None:
    betas = torch.stack((torch.zeros(10), torch.linspace(-0.5, 0.5, 10)))
    shaped = body.with_shape(betas=betas)
    assert shaped is not body
    assert body.values.v_shaped.shape == (6890, 3)
    assert shaped.values.v_shaped.shape == (2, 6890, 3)
    assert shaped.robot.joint_placements.shape == (2, 25, 7)
    q = shaped.from_classic(SMPLClassic())
    assert q.shape == (2, shaped.robot.nq)
    vertices = shaped.vertices(q[0])
    assert vertices.shape == (2, 6890, 3)
    torch.testing.assert_close(shaped.values.rest_joints[..., 0, :], torch.zeros(2, 3))


def test_classic_round_trips_both_directions_with_root_wart(body: bh.SMPL) -> None:
    shaped = body.with_shape(betas=torch.linspace(-1.0, 1.0, 10))
    classic = SMPLClassic(
        global_orient=torch.tensor([0.4, -0.3, 0.2]),
        body_pose=torch.linspace(-0.5, 0.5, 69),
        transl=torch.tensor([1.2, -0.7, 0.4]),
    )
    q = shaped.from_classic(classic)
    recovered = shaped.to_classic(q)
    torch.testing.assert_close(recovered.global_orient, classic.global_orient, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(recovered.body_pose, classic.body_pose, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(recovered.transl, classic.transl, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(shaped.from_classic(recovered), q, rtol=2e-6, atol=2e-6)


def test_to_classic_preserves_independent_quaternion_signs(body: bh.SMPL) -> None:
    q = body.from_classic(
        SMPLClassic(
            global_orient=torch.tensor([4.0, 0.0, 0.0]),
            body_pose=torch.cat((torch.tensor([0.0, -4.0, 0.0]), torch.zeros(66))),
        )
    )
    q = q.clone()
    q[3:7] = -q[3:7]
    q[7:11] = -q[7:11]
    torch.testing.assert_close(body.from_classic(body.to_classic(q)), q, rtol=2e-6, atol=2e-6)


def test_scaled_joint_quaternion_does_not_change_vertices(body: bh.SMPL) -> None:
    q = body.from_classic(SMPLClassic(body_pose=torch.linspace(-0.2, 0.2, 69)))
    scaled_q = q.clone()
    scaled_q[7:11] *= 2.0
    torch.testing.assert_close(body.vertices(scaled_q), body.vertices(q), rtol=2e-6, atol=2e-6)


def test_vertices_calls_fk_once(body: bh.SMPL, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0
    original = body_module.br.forward_kinematics

    def counted_forward_kinematics(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(body_module.br, "forward_kinematics", counted_forward_kinematics)
    body.vertices(body.from_classic(SMPLClassic()))
    assert calls == 1


def test_float32_gradcheck_betas_to_selected_vertices(body: bh.SMPL) -> None:
    betas = (torch.randn(10) * 0.1).requires_grad_()
    q = body.from_classic(SMPLClassic(body_pose=torch.linspace(-0.2, 0.2, 69)))

    def selected_vertices(shape: torch.Tensor) -> torch.Tensor:
        return body.with_shape(betas=shape).vertices(q)[GRADCHECK_VERTEX_INDICES]

    assert torch.autograd.gradcheck(selected_vertices, (betas,), eps=1e-3, atol=2e-2, rtol=5e-2, fast_mode=True)


def test_float32_gradcheck_q_to_selected_vertices(body: bh.SMPL) -> None:
    q = body.from_classic(
        SMPLClassic(
            global_orient=torch.tensor([0.1, -0.2, 0.15]),
            body_pose=torch.linspace(-0.15, 0.15, 69),
            transl=torch.tensor([0.2, -0.1, 0.3]),
        )
    ).requires_grad_()

    def selected_vertices(configuration: torch.Tensor) -> torch.Tensor:
        return body.vertices(configuration)[GRADCHECK_VERTEX_INDICES]

    assert torch.autograd.gradcheck(selected_vertices, (q,), eps=1e-3, atol=3e-2, rtol=8e-2, fast_mode=True)


def test_landmark_positions_gather_matches_config(body: bh.SMPL) -> None:
    q = body.from_classic(SMPLClassic(body_pose=torch.linspace(-0.2, 0.2, 69)))
    vertices = body.vertices(q)
    frame_vertex_ids = body.structure.landmarks["frame_vertex_ids"]
    assert frame_vertex_ids.tolist() == list(SMPL_CONFIG.frame_vertex_ids)

    landmarks = body.landmark_positions(vertices)
    assert landmarks.shape == (len(SMPL_CONFIG.frame_names), 3)
    torch.testing.assert_close(landmarks, vertices[frame_vertex_ids])
    nose = SMPL_CONFIG.frame_names.index("nose")
    torch.testing.assert_close(landmarks[nose], vertices[332])


def test_landmark_positions_preserves_leading_batch(body: bh.SMPL) -> None:
    shaped = body.with_shape(betas=torch.stack((torch.zeros(10), torch.linspace(-0.5, 0.5, 10))))
    vertices = shaped.vertices(shaped.from_classic(SMPLClassic()))
    landmarks = shaped.landmark_positions(vertices)
    assert landmarks.shape == (2, len(SMPL_CONFIG.frame_names), 3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_fk_and_vertices_smoke() -> None:
    body = bh.SMPL(MODEL_PATH, compute_mass=False, device=torch.device("cuda"))
    q = body.from_classic(SMPLClassic())
    assert body.vertices(q).is_cuda
