"""Behaviour of the SOMA family: sizes, bake against upstream, round trip, root frame, batching, gradients."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from better_robot.lie import se3, so3

from better_human import Body, families, load
from better_human.families.soma import SOMAClassic, soma_from_classic, soma_to_classic

ROOT = Path(__file__).parents[2]
MODEL_PATH = ROOT / "models" / "SOMA" / "soma_mid.npz"
GOLDEN_FILES = tuple(sorted((ROOT / "tests" / "parity" / "goldens" / "soma_mid").glob("*.npz")))
# Rest mesh [m], bind positions [m], bind rotation entries, float32 and float64: 2x the worst error over the 14
# goldens and their identity batch, rounded up; never below the float32 oracle's 5.2e-6 m joint regression rounding.
# Single reflect_00, a 5-sigma identity, sets its own rest mesh gate and not the general one: its float32 joint
# regression rounds further (5.9e-6 m; 3.2e-6 m with that regression in float64), as upstream's does (7.3e-6 m from
# its float64).
BAKE_ATOLS = (6e-6, 1e-5, 1.1e-6)
BAKE_ATOLS_BY_CASE = {"reflect_00": (1.2e-5, 1e-5, 1.1e-6)}


def _model_path() -> Path:
    if not MODEL_PATH.is_file():
        pytest.skip(f"converted SOMA fixture is absent: {MODEL_PATH}")
    return MODEL_PATH


def _random_classic(generator: torch.Generator, batch: tuple[int, ...] = ()) -> SOMAClassic:
    """Moderate identity ``(*batch, 128)``, pose ``(*batch, 77, 3)`` and root position ``(*batch, 3)``."""
    return SOMAClassic(
        0.5 * torch.randn(*batch, 128, generator=generator),
        0.3 * torch.randn(*batch, 77, 3, generator=generator),
        0.5 * torch.randn(*batch, 3, generator=generator),
    )


@pytest.fixture(scope="module")
def body() -> Body:
    return load("soma", str(_model_path()))


def test_sizes_parts_and_mass(body: Body) -> None:
    spec = body.spec
    assert (spec.num_vertices, spec.num_joints, spec.identity["shape"].shape[-1]) == (18056, 110, 128)
    assert spec.joint_names[:2] == ("Root", "Hips")
    assert (body.model.nq, body.model.nv) == (311, 234)
    assert set(spec.parts.joints) <= set(spec.rig.public_ids.tolist())  # twist helpers fold into their parents
    masses = body.bake().model.body_inertias[..., 0]
    assert torch.isfinite(masses).all()
    assert (masses[body.part_body_ids] > 0).all()


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64), ids=("float32", "float64"))
def test_bake_matches_upstream_rest_shape_and_bind(dtype: torch.dtype) -> None:
    """Skeleton fit and repose for every golden identity: the skinning rest mesh and the bind poses."""
    assert GOLDEN_FILES, "SOMA goldens are missing"
    body = Body(families.soma(_model_path(), correctives=False, dtype=dtype), mass=False)
    for path in GOLDEN_FILES:
        rest_atol, position_atol, rotation_atol = BAKE_ATOLS_BY_CASE.get(path.stem, BAKE_ATOLS)
        with np.load(path, allow_pickle=False) as golden:
            names = ("identity_coeffs", "rest_shape", "bind_transforms")
            coefficients, rest_shape, expected = (torch.from_numpy(golden[name]).to(dtype) for name in names)
        identity = body.bake(shape=coefficients)
        bind = se3.to_matrix(se3.inverse(identity.inverse_bind_poses))
        rest_error = (identity.v_shaped - rest_shape).abs().amax().item()
        positions = torch.stack((identity.rest_joints, bind[:, :3, 3]))
        position_error = (positions - expected[:, :3, 3]).abs().amax().item()
        rotation_error = (bind[:, :3, :3] - expected[:, :3, :3]).abs().amax().item()
        assert rest_error <= rest_atol, f"{path.stem}: rest mesh max error {rest_error:.3e}"
        assert position_error <= position_atol, f"{path.stem}: bind position max error {position_error:.3e}"
        assert rotation_error <= rotation_atol, f"{path.stem}: bind rotation max error {rotation_error:.3e}"


@pytest.mark.parametrize(("dtype", "atol"), ((torch.float32, 1e-6), (torch.float64, 1e-12)), ids=("float32", "float64"))
def test_classic_round_trip(dtype: torch.dtype, atol: float) -> None:
    body = Body(families.soma(_model_path(), correctives=False, dtype=dtype), mass=False)
    random = _random_classic(torch.Generator().manual_seed(0), (4,))
    classic = SOMAClassic(random.identity_coeffs.to(dtype), random.poses.to(dtype), random.transl.to(dtype))
    identity, q, stream = soma_from_classic(body, classic)
    assert q.shape == (4, 311)
    assert stream == {}
    restored = soma_to_classic(identity, q)
    torch.testing.assert_close(restored.identity_coeffs, classic.identity_coeffs, rtol=0, atol=0)
    torch.testing.assert_close(restored.poses, classic.poses, rtol=0, atol=atol)
    torch.testing.assert_close(restored.transl, classic.transl, rtol=0, atol=atol)


def test_free_flyer_is_world_aligned(body: Body) -> None:
    """``q[:3]`` moves the body along the world axes; ``q[3:7]`` turns it about the Hips, which start at the origin."""
    identity = body.bake()
    neutral = identity.forward(identity.q_neutral)
    q = identity.q_neutral.clone()
    q[0] += 1.0
    moved = identity.forward(q)
    torch.testing.assert_close(moved.vertices, neutral.vertices + q.new_tensor((1.0, 0.0, 0.0)), rtol=0, atol=1e-6)
    q = identity.q_neutral.clone()
    q[3:7] = so3.exp(q.new_tensor((0.0, 0.5, 0.0)))
    turned, turn = identity.forward(q), so3.to_matrix(q[3:7])
    torch.testing.assert_close(turned.joints, neutral.joints @ turn.mT, rtol=0, atol=1e-6)
    expected = turn @ neutral.joint_transforms[1:, :, :3]  # Root, the universe, does not turn
    torch.testing.assert_close(turned.joint_transforms[1:, :, :3], expected, rtol=0, atol=1e-6)


def test_identity_batch_broadcasts_against_pose_batch(body: Body) -> None:
    generator = torch.Generator().manual_seed(1)
    shapes = 0.5 * torch.randn(2, 1, 128, generator=generator)
    poses = _random_classic(generator, (3,))
    identity, q, _ = soma_from_classic(body, SOMAClassic(shapes, poses.poses, poses.transl))
    state = identity.forward(q)
    assert q.shape == (3, 311)
    assert state.vertices.shape == (2, 3, 18056, 3)
    # Batched float32 sums round in another order: 8.3e-7 m for these inputs (seed 1), not a bound; the joint
    # regression alone rounds by up to 5.2e-6 m.
    for i in range(2):
        single = body.bake(shape=shapes[i, 0])
        for j in range(3):
            expected = single.forward(q[j])
            torch.testing.assert_close(state.vertices[i, j], expected.vertices, rtol=0, atol=2e-6)
            torch.testing.assert_close(state.joint_poses[i, j], expected.joint_poses, rtol=0, atol=2e-6)


def test_gradients_match_finite_differences() -> None:
    """Shape, pose and root gradients of a vertex and joint loss are finite and match central differences."""
    body = Body(families.soma(_model_path(), dtype=torch.float64), mass=False)
    generator = torch.Generator().manual_seed(2)
    classic = _random_classic(generator)
    weights = torch.randn(18056, 3, generator=generator, dtype=torch.float64)
    inputs = tuple(
        value.double().requires_grad_() for value in (classic.identity_coeffs, classic.poses, classic.transl)
    )
    public = [body.spec.joint_names[joint] for joint in body.spec.rig.public_ids[1:].tolist()]

    def loss(*values: torch.Tensor) -> torch.Tensor:
        identity, q, _ = soma_from_classic(body, SOMAClassic(*values))
        state = identity.forward(q)
        return (state.vertices * weights).sum() + state.joints.sum()

    gradients = torch.autograd.grad(loss(*inputs), inputs)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    step = 1e-5
    entries = ((0, (0,)), (0, (100,)), (1, (0, 1)), (1, (public.index("LeftForeArm"), 0)), (2, (1,)))
    for which, index in entries:  # shape, Hips, forearm twist (drives the twist helpers), root height
        plus, minus = ([value.detach().clone() for value in inputs] for _ in range(2))
        plus[which][index] += step
        minus[which][index] -= step
        with torch.no_grad():
            numeric = (loss(*plus) - loss(*minus)) / (2 * step)
        torch.testing.assert_close(gradients[which][index], numeric, rtol=1e-6, atol=1e-7)


def test_spec_and_classic_validation(body: Body) -> None:
    with pytest.raises(TypeError, match="torch.float32 or torch.float64"):
        families.soma(_model_path(), dtype=torch.float16)
    with pytest.raises(TypeError, match="must be bool"):
        families.soma(_model_path(), correctives=1)
    zero = SOMAClassic(torch.zeros(128), torch.zeros(77, 3), torch.zeros(3))
    with pytest.raises(ValueError, match=r"\(\.\.\., 77, 3\)"):
        soma_from_classic(body, SOMAClassic(zero.identity_coeffs, torch.zeros(78, 3), zero.transl))
    with pytest.raises(TypeError, match="floating"):
        soma_from_classic(body, SOMAClassic(zero.identity_coeffs, zero.poses, None))
    identity, q, _ = soma_from_classic(body, zero)
    with pytest.raises(ValueError, match="311"):
        soma_to_classic(identity, q[:-1])
