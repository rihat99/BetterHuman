"""Every rig's bake and forward kinematics agree through the one skinning path.

Oracle: the bake itself. At ``q_neutral``, ``skin_affines`` (forward kinematics composed with the baked
``inverse_bind_affines``) carry every baked rest joint (``RigBake.rest_joint_translations``, the native joints in the
skinning rest frame) to its forward-kinematics position, for every rig and identity (joint form). At the bind
configuration, where every skin affine is the identity, forward kinematics reproduces ``rest_vertices`` (mesh form).
The SMPL family and MHR bind at ``q_neutral``; SOMA binds in its authored A-pose (``q_neutral`` is the T-pose), whose
configuration the test reads from the bake's inverse binds. ``NOT_HELD`` lists the cases where the mesh form does not
hold, each with its measured gap and reason; they are left out, never loosened.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
import torch

from better_robot.lie import so3

import better_human as bh
from better_human.io import weight_path
from tests.support.synthetic import flame_spec

#: Float64 rounding of the SMPL-family and SOMA bakes, which compose exact inverses.
EXACT = 1e-12
#: MHR's authored inverse bind matrices are float32 values: at zero proportions they invert the forward kinematics to
#: 1.5e-7 m (LOD 6, measured 2026-11-10), not to float64 rounding.
MHR_AUTHORED_BINDS = 1e-6

#: ``(family, identity)`` cases where the mesh form does not hold: gap measured 2026-11-10 (LOD 1, float64), reason.
NOT_HELD = {
    ("mhr", "random"): "0.18 m: MHR's proportions scale its skin affines (joint scales s != 1): none is I at any q",
}

LICENSED: dict[str, Callable[[], bh.FamilySpec]] = {
    "smpl": lambda: bh.families.smpl(weight_path("smpl", "SMPL_NEUTRAL.npz"), dtype=torch.float64),
    "smplx": lambda: bh.families.smplx(weight_path("smplx", "SMPLX_NEUTRAL.npz"), dtype=torch.float64),
    "mhr": lambda: bh.families.mhr(weight_path("MHR/converted", "mhr_lod1.npz"), dtype=torch.float64),
    "soma": lambda: bh.families.soma(weight_path("SOMA", "soma_mid.npz"), correctives=False, dtype=torch.float64),
}


def _identities(human: bh.HumanModel) -> dict[str, bh.Identity]:
    """The zero identity and one seeded random identity (seed 0) of ``human``."""
    generator = torch.Generator().manual_seed(0)
    values = {
        name: 0.5 * torch.randn(width, generator=generator, dtype=torch.float64)
        for name, width in human.spec.identity_groups.items()
    }
    return {"zero": human.bake(), "random": human.bake(**values)}


def _held(family: str, identities: dict[str, bh.Identity]) -> dict[str, bh.Identity]:
    """The ``identities`` of ``family`` whose mesh form is not in ``NOT_HELD``."""
    held = {name: identity for name, identity in identities.items() if (family, name) not in NOT_HELD}
    assert held, f"every {family} identity is in NOT_HELD: leave {family} out of the parameters"
    return held


def _joint_error(identity: bh.Identity) -> float:
    """Largest distance between ``skin_affines`` applied to the rest joints and the joints at ``q_neutral``."""
    state = identity.forward(identity.model.q_neutral)
    affines, rest = state.skin_affines, identity.rest_joint_translations
    carried = (affines[..., :3] @ rest[..., None]).squeeze(-1) + affines[..., 3]
    return float((carried - state.joint_translations).abs().max())


def _soma_bind_configuration(identity: bh.Identity) -> torch.Tensor:
    """``q (nq,)`` that puts every SOMA joint at its bind frame, read from the bake's inverse binds.

    Native joint ``j`` is BR joint ``j`` (``Root`` the universe). A joint's bind rotation and position invert its
    inverse bind affine; its local rotation is the bind rotation seen from its parent's bind frame and its placement.
    The BR Hips body frame is the upstream Hips frame turned back by its T-pose rotation (``SOMARig``).
    """
    model = identity.model
    inverse_binds = identity.inverse_bind_affines
    rotations = inverse_binds[..., :3].mT.clone()
    positions = -(rotations @ inverse_binds[..., 3:]).squeeze(-1)
    rotations[1] = rotations[1] @ identity.human.spec.rig.t_pose_rotations[1].mT
    placements = so3.to_matrix(model.joint_placements[..., 3:])
    q = model.q_neutral.clone()
    for joint in range(1, model.njoints):
        if model.nqs[joint] == 0:  # twist helpers follow their limb segments
            continue
        parent = model.parent_joint_ids[joint]
        parent_rotation = rotations[parent] if parent > 0 else torch.eye(3, dtype=q.dtype)
        local = so3.from_matrix(placements[joint].mT @ parent_rotation.mT @ rotations[joint])
        start = model.idx_qs[joint]
        if model.nqs[joint] == 7:
            q[start : start + 3] = positions[joint] - model.joint_placements[joint, :3]
            q[start + 3 : start + 7] = local
        else:
            q[start : start + 4] = local
    return q


def _mesh_error(identity: bh.Identity, q: torch.Tensor) -> float:
    """Largest distance between the vertices at ``q`` and ``rest_vertices``."""
    return float((identity.forward(q).vertices - identity.rest_vertices).abs().max())


def _weight_defect(identity: bh.Identity) -> float:
    """How far identity affines move a rest vertex because its skin weights do not sum to exactly one.

    The SMPL-X file stores float32 weights: their rows sum to one within about 4e-8.
    """
    defect = (identity.human.skin_matrix.sum(-1) - 1.0).abs()
    return float((defect[:, None] * identity.rest_vertices.abs()).max())


def test_regressed_rig_reproduces_its_rest_mesh_and_joints(tmp_path: Path) -> None:
    human = bh.HumanModel.from_spec(flame_spec(tmp_path), density=None)
    for name, identity in _identities(human).items():
        mesh, joints = _mesh_error(identity, identity.model.q_neutral), _joint_error(identity)
        assert mesh <= EXACT + _weight_defect(identity), f"{name}: vertices at q_neutral differ from rest by {mesh} m"
        assert joints <= EXACT, f"{name}: skin_affines move the rest joints off the joints by {joints} m"


@pytest.mark.licensed
@pytest.mark.parametrize("family", list(LICENSED))
def test_skin_affines_carry_the_rest_joints_to_the_joints(family: str) -> None:
    human = bh.HumanModel.from_spec(LICENSED[family](), density=None)
    tolerance = MHR_AUTHORED_BINDS if family == "mhr" else EXACT
    for name, identity in _identities(human).items():
        error = _joint_error(identity)
        assert error <= tolerance, f"{name}: skin_affines move the rest joints off the joints by {error} m"


@pytest.mark.licensed
@pytest.mark.parametrize("family", list(LICENSED))
def test_forward_kinematics_at_the_bind_configuration_reproduces_the_rest_mesh(family: str) -> None:
    human = bh.HumanModel.from_spec(LICENSED[family](), density=None)
    tolerance = MHR_AUTHORED_BINDS if family == "mhr" else EXACT
    for name, identity in _held(family, _identities(human)).items():
        q = _soma_bind_configuration(identity) if family == "soma" else identity.model.q_neutral
        error = _mesh_error(identity, q)
        bound = tolerance + _weight_defect(identity)
        assert error <= bound, f"{name}: vertices at the bind configuration differ from rest_vertices by {error} m"
