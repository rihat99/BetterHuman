"""SMPL-X conversion, reduction, batching, and differentiability tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import better_human as bh
from better_human.bodies import SMPLXClassic
from better_human.bodies.config_smplx import SMPLX_CONFIG

MODEL_PATH = Path(__file__).parents[2] / "models" / "smplx" / "SMPLX_NEUTRAL.npz"
GRADCHECK_VERTEX_INDICES = torch.tensor([0, 2618, 5237, 7856, 10474], dtype=torch.long)
PARTIAL_ATOL = 5e-7


@pytest.fixture(scope="module")
def body() -> bh.SMPLX:
    """Load the licensed local neutral fixture when available."""
    if not MODEL_PATH.is_file():
        pytest.skip(f"licensed SMPL-X fixture is absent: {MODEL_PATH}")
    return bh.SMPLX(MODEL_PATH, compute_mass=False)


def test_registry_and_preserved_native_alignment(body: bh.SMPLX) -> None:
    loaded = bh.load("smplx", model_path=MODEL_PATH, compute_mass=False)
    assert isinstance(loaded, bh.SMPLX)
    assert body.robot.body_names[1:] == body.structure.joint_names
    assert body.robot.joint_names[1:] == ("root", *body.structure.joint_names[1:])
    assert body.structure.native_joint_indices == tuple(range(55))


def test_full_axis_angle_round_trip_in_native_jaw_eyes_hands_order(body: bh.SMPLX) -> None:
    expression = torch.linspace(-0.4, 0.4, 10)
    shaped = body.with_shape(betas=torch.linspace(-0.5, 0.5, 16), expression=expression)
    classic = SMPLXClassic(
        global_orient=torch.tensor([0.3, -0.2, 0.1]),
        body_pose=torch.linspace(-0.25, 0.25, 63),
        jaw_pose=torch.tensor([0.1, -0.05, 0.02]),
        leye_pose=torch.tensor([0.02, 0.03, -0.01]),
        reye_pose=torch.tensor([-0.03, 0.01, 0.02]),
        left_hand_pose=torch.linspace(-0.2, 0.2, 45),
        right_hand_pose=torch.linspace(0.2, -0.2, 45),
        expression=expression,
        transl=torch.tensor([0.5, -0.3, 0.2]),
    )
    q = shaped.from_classic(classic)
    recovered = shaped.to_classic(q)
    for name in (
        "global_orient",
        "body_pose",
        "jaw_pose",
        "leye_pose",
        "reye_pose",
        "left_hand_pose",
        "right_hand_pose",
        "expression",
        "transl",
    ):
        torch.testing.assert_close(getattr(recovered, name), getattr(classic, name), rtol=3e-6, atol=3e-6)
    torch.testing.assert_close(shaped.from_classic(recovered), q, rtol=3e-6, atol=3e-6)

    signed_q = q.clone()
    root_quaternion = slice(3, 7)
    jaw_start = shaped.robot.idx_qs[shaped.structure.joint_names.index("jaw") + 1]
    hand_start = shaped.robot.idx_qs[shaped.structure.joint_names.index("left_index1") + 1]
    signed_q[root_quaternion] = -signed_q[root_quaternion]
    signed_q[jaw_start : jaw_start + 4] = -signed_q[jaw_start : jaw_start + 4]
    signed_q[hand_start : hand_start + 4] = -signed_q[hand_start : hand_start + 4]
    torch.testing.assert_close(
        shaped.from_classic(shaped.to_classic(signed_q)),
        signed_q,
        rtol=3e-6,
        atol=3e-6,
    )


@pytest.mark.parametrize("flat_hand_mean", [False, True])
def test_pca_hand_round_trip_uses_scaled_components(flat_hand_mean: bool) -> None:
    body = bh.SMPLX(MODEL_PATH, flat_hand_mean=flat_hand_mean, compute_mass=False)
    classic = SMPLXClassic(
        left_hand_pose=torch.linspace(-0.5, 0.5, 6),
        right_hand_pose=torch.linspace(0.4, -0.4, 6),
        num_pca_comps=6,
    )
    q = body.from_classic(classic)
    recovered = body.to_classic(q, num_pca_comps=6)
    torch.testing.assert_close(recovered.left_hand_pose, classic.left_hand_pose, rtol=5e-5, atol=5e-5)
    torch.testing.assert_close(recovered.right_hand_pose, classic.right_hand_pose, rtol=5e-5, atol=5e-5)
    torch.testing.assert_close(body.from_classic(recovered), q, rtol=5e-6, atol=5e-6)


def test_zero_hand_delta_uses_mean_unless_flat_hand_mean() -> None:
    natural = bh.SMPLX(MODEL_PATH, flat_hand_mean=False, compute_mass=False)
    flat = bh.SMPLX(MODEL_PATH, flat_hand_mean=True, compute_mass=False)
    natural_q = natural.from_classic(SMPLXClassic())
    flat_q = flat.from_classic(SMPLXClassic())
    assert not torch.allclose(natural_q[103:], flat_q[103:])
    assert (natural.vertices(natural_q) - flat.vertices(flat_q)).abs().max() > 1e-3


@pytest.mark.parametrize(
    ("use_hands", "use_face", "expected_joints"),
    [(False, True, 25), (True, False, 52), (False, False, 22)],
)
@pytest.mark.parametrize("flat_hand_mean", [False, True])
def test_partial_spawn_matches_full_body_with_dropped_dofs_at_rest(
    use_hands: bool,
    use_face: bool,
    expected_joints: int,
    flat_hand_mean: bool,
) -> None:
    full = bh.SMPLX(
        MODEL_PATH,
        num_betas=5,
        num_expression=3,
        flat_hand_mean=flat_hand_mean,
        compute_mass=False,
    )
    partial = bh.SMPLX(
        MODEL_PATH,
        num_betas=5,
        num_expression=3,
        use_hands=use_hands,
        use_face=use_face,
        flat_hand_mean=flat_hand_mean,
        compute_mass=False,
    )
    assert len(partial.structure.joint_names) == expected_joints
    betas = torch.linspace(-0.6, 0.6, 5)
    expression = torch.tensor([0.2, -0.1, 0.3]) if use_face else None
    full_shaped = full.with_shape(
        betas=betas,
        expression=expression if expression is not None else torch.zeros(3),
    )
    partial_shaped = (
        partial.with_shape(betas=betas, expression=expression)
        if expression is not None
        else partial.with_shape(betas=betas)
    )

    body_pose = torch.linspace(-0.3, 0.3, 63)
    left_hand = torch.linspace(-0.15, 0.15, 45) if use_hands else -full._left_hand_mean
    right_hand = torch.linspace(0.15, -0.15, 45) if use_hands else -full._right_hand_mean
    jaw = torch.tensor([0.1, -0.05, 0.02]) if use_face else torch.zeros(3)
    leye = torch.tensor([0.01, 0.02, -0.01]) if use_face else torch.zeros(3)
    reye = torch.tensor([-0.02, 0.01, 0.01]) if use_face else torch.zeros(3)
    shared = {
        "global_orient": torch.tensor([0.2, -0.1, 0.15]),
        "body_pose": body_pose,
        "transl": torch.tensor([0.3, -0.2, 0.1]),
    }
    full_classic = SMPLXClassic(
        **shared,
        jaw_pose=jaw,
        leye_pose=leye,
        reye_pose=reye,
        left_hand_pose=left_hand,
        right_hand_pose=right_hand,
        expression=expression if expression is not None else torch.zeros(3),
    )
    partial_classic = SMPLXClassic(
        **shared,
        jaw_pose=jaw if use_face else None,
        leye_pose=leye if use_face else None,
        reye_pose=reye if use_face else None,
        left_hand_pose=left_hand if use_hands else None,
        right_hand_pose=right_hand if use_hands else None,
        expression=expression,
    )
    expected = full_shaped.vertices(full_shaped.from_classic(full_classic))
    actual = partial_shaped.vertices(partial_shaped.from_classic(partial_classic))
    torch.testing.assert_close(actual, expected, rtol=0, atol=PARTIAL_ATOL)


def test_expression_is_baked_shape_state_and_batches_with_q(body: bh.SMPLX) -> None:
    expressions = torch.stack((torch.zeros(10), torch.linspace(-0.2, 0.2, 10)))
    shaped = body.with_shape(expression=expressions)
    q = shaped.from_classic(SMPLXClassic())
    assert shaped.vertices(q[0]).shape == (2, 10475, 3)
    with pytest.raises(ValueError, match="shape state"):
        shaped.from_classic(SMPLXClassic(expression=torch.ones(10)))


def test_float32_gradcheck_expression_to_selected_vertices(body: bh.SMPLX) -> None:
    expression = (torch.randn(10) * 0.05).requires_grad_()

    def selected_vertices(value: torch.Tensor) -> torch.Tensor:
        shaped = body.with_shape(expression=value)
        q = shaped.from_classic(SMPLXClassic())
        return shaped.vertices(q)[GRADCHECK_VERTEX_INDICES]

    assert torch.autograd.gradcheck(
        selected_vertices,
        (expression,),
        eps=1e-3,
        atol=3e-2,
        rtol=8e-2,
        fast_mode=True,
    )


def test_landmark_positions_gather_matches_config(body: bh.SMPLX) -> None:
    q = body.from_classic(SMPLXClassic(body_pose=torch.linspace(-0.2, 0.2, 63)))
    vertices = body.vertices(q)
    frame_vertex_ids = body.structure.landmarks["frame_vertex_ids"]
    assert frame_vertex_ids.tolist() == list(SMPLX_CONFIG.frame_vertex_ids)

    landmarks = body.landmark_positions(vertices)
    assert landmarks.shape == (len(SMPLX_CONFIG.frame_names), 3)
    torch.testing.assert_close(landmarks, vertices[frame_vertex_ids])
    nose = SMPLX_CONFIG.frame_names.index("nose")
    torch.testing.assert_close(landmarks[nose], vertices[9120])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_fk_and_vertices_smoke() -> None:
    body = bh.SMPLX(MODEL_PATH, compute_mass=False, device=torch.device("cuda"))
    q = body.from_classic(SMPLXClassic())
    assert body.vertices(q).is_cuda
