"""Behaviour of the SOMA family: sizes, root frame, batching, gradients.

The classic round trip (float32 and float64) runs through tests/families/test_classic_formats.py.
"""

from __future__ import annotations

import pytest
import torch
from better_robot.exceptions import ArgumentError, DtypeMismatchError, ShapeError
from better_robot.lie import so3

from better_human import HumanModel, families, from_classic, load, to_classic
from better_human.families import SOMAClassic
from better_human.io import weight_path

pytestmark = pytest.mark.licensed


def _random_classic(generator: torch.Generator, batch: tuple[int, ...] = ()) -> SOMAClassic:
    """Moderate identity ``(*batch, 128)``, pose ``(*batch, 77, 3)`` and root position ``(*batch, 3)``."""
    return SOMAClassic(
        identity_coeffs=0.5 * torch.randn(*batch, 128, generator=generator),
        poses=0.3 * torch.randn(*batch, 77, 3, generator=generator),
        transl=0.5 * torch.randn(*batch, 3, generator=generator),
    )


@pytest.fixture(scope="module")
def body() -> HumanModel:
    return load("soma", str(weight_path("SOMA", "soma_mid.npz")))


def test_sizes_parts_and_mass(body: HumanModel) -> None:
    spec = body.spec
    assert (spec.num_vertices, spec.num_joints, spec.identity_bases["shape"].shape[-1]) == (18056, 110, 128)
    assert spec.joint_names[:2] == ("Root", "Hips")
    assert (body.model.nq, body.model.nv) == (311, 234)
    assert set(spec.parts.joints) <= set(spec.rig.public_ids.tolist())  # twist helpers fold into their parents
    masses = body.bake().model.body_inertias[..., 0]
    assert torch.isfinite(masses).all()
    assert (masses[body.part_body_ids] > 0).all()


def test_free_flyer_is_world_aligned(body: HumanModel) -> None:
    """``q[:3]`` moves the body along the world axes; ``q[3:7]`` turns it about the Hips, which start at the origin."""
    identity = body.bake()
    neutral = identity.forward(identity.model.q_neutral)
    q = identity.model.q_neutral.clone()
    q[0] += 1.0
    moved = identity.forward(q)
    torch.testing.assert_close(moved.vertices, neutral.vertices + q.new_tensor((1.0, 0.0, 0.0)), rtol=0, atol=1e-6)
    q = identity.model.q_neutral.clone()
    q[3:7] = so3.exp(q.new_tensor((0.0, 0.5, 0.0)))
    turned, turn = identity.forward(q), so3.to_matrix(q[3:7])
    torch.testing.assert_close(turned.joint_translations, neutral.joint_translations @ turn.mT, rtol=0, atol=1e-6)
    expected = turn @ neutral.joint_transforms[1:, :, :3]  # Root, the universe, does not turn
    torch.testing.assert_close(turned.joint_transforms[1:, :, :3], expected, rtol=0, atol=1e-6)


def test_identity_batch_broadcasts_against_pose_batch(body: HumanModel) -> None:
    generator = torch.Generator().manual_seed(1)
    shapes = 0.5 * torch.randn(2, 1, 128, generator=generator)
    poses = _random_classic(generator, (3,))
    identity, q, _ = from_classic(body, SOMAClassic(identity_coeffs=shapes, poses=poses.poses, transl=poses.transl))
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
    body = HumanModel.from_spec(families.soma(weight_path("SOMA", "soma_mid.npz"), dtype=torch.float64), density=None)
    generator = torch.Generator().manual_seed(2)
    classic = _random_classic(generator)
    weights = torch.randn(18056, 3, generator=generator, dtype=torch.float64)
    inputs = tuple(
        value.double().requires_grad_() for value in (classic.identity_coeffs, classic.poses, classic.transl)
    )
    public = [body.spec.joint_names[joint] for joint in body.spec.rig.public_ids[1:].tolist()]

    def loss(*values: torch.Tensor) -> torch.Tensor:
        identity, q, _ = from_classic(body, SOMAClassic(identity_coeffs=values[0], poses=values[1], transl=values[2]))
        state = identity.forward(q)
        return (state.vertices * weights).sum() + state.joint_translations.sum()

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


def test_spec_and_classic_validation(body: HumanModel) -> None:
    with pytest.raises(DtypeMismatchError, match="torch.float32 or torch.float64"):
        families.soma(weight_path("SOMA", "soma_mid.npz"), dtype=torch.float16)
    with pytest.raises(ArgumentError, match="must be bool"):
        families.soma(weight_path("SOMA", "soma_mid.npz"), correctives=1)
    with pytest.raises(ShapeError, match=r"\(\.\.\., 77, 3\)"):
        from_classic(body, SOMAClassic(poses=torch.zeros(78, 3)))
    with pytest.raises(TypeError, match="floating"):
        from_classic(body, SOMAClassic(transl=[0.0, 0.0, 0.0]))
    identity, q, _ = from_classic(body, SOMAClassic())
    with pytest.raises(ValueError, match="311"):
        to_classic(identity, q[:-1])
