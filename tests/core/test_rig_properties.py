"""Decision records 0002 and 0003: every rig's bake and forward kinematics agree through the one skinning path.

Oracle: the bake itself. At ``q_neutral``, ``skin_affines`` (forward kinematics composed with the baked
``inverse_bind_affines``) carry every baked rest joint to its forward-kinematics position, for every rig and identity.
Where a rig binds at its neutral configuration (the SMPL family, and MHR at zero proportions), they also carry the
rest mesh onto itself: forward kinematics at ``q_neutral`` reproduces ``rest_vertices``. ``NOT_HELD`` lists the cases
where a form does not hold, each with its measured gap and reason; they are left out, never loosened.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
import torch

import better_human as bh
from better_human.io import weight_path
from tests.support.synthetic import flame_spec

#: Float64 rounding of the SMPL-family and SOMA bakes, which compose exact inverses.
EXACT = 1e-12
#: MHR's authored inverse bind matrices are float32 values: at zero proportions they invert the forward kinematics to
#: 1.5e-7 m (LOD 6, measured 2026-11-10), not to float64 rounding.
MHR_AUTHORED_BINDS = 1e-6

#: ``(form, family, identity)`` cases where the property does not hold: gap measured 2026-11-10 (LOD 1, float64), reason.
NOT_HELD = {
    ("joints", "mhr", "random"): "0.31 m: MHR's proportions scale the joints through the skinning",
    ("mesh", "mhr", "random"): "0.18 m: MHR's proportions scale the mesh through the skinning",
    ("mesh", "soma", "zero"): "0.97 m: SOMA binds in its authored A-pose and q_neutral is the T-pose",
    ("mesh", "soma", "random"): "0.89 m: SOMA binds in its authored A-pose and q_neutral is the T-pose",
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


def _held(form: str, family: str, identities: dict[str, bh.Identity]) -> dict[str, bh.Identity]:
    """The ``identities`` of ``family`` whose ``form`` (``"joints"`` or ``"mesh"``) is not in ``NOT_HELD``."""
    held = {name: identity for name, identity in identities.items() if (form, family, name) not in NOT_HELD}
    assert held, f"every {family} identity is in NOT_HELD for the {form} form: leave {family} out of the parameters"
    return held


def _joint_error(identity: bh.Identity) -> float:
    """Largest distance between ``skin_affines`` applied to the rest joints and the joints at ``q_neutral``."""
    state = identity.forward(identity.model.q_neutral)
    affines, rest = state.skin_affines, identity.rest_joint_translations
    carried = (affines[..., :3] @ rest[..., None]).squeeze(-1) + affines[..., 3]
    return float((carried - state.joint_translations).abs().max())


def _mesh_error(identity: bh.Identity) -> float:
    """Largest distance between the vertices at ``q_neutral`` and ``rest_vertices``."""
    return float((identity.forward(identity.model.q_neutral).vertices - identity.rest_vertices).abs().max())


def _weight_defect(identity: bh.Identity) -> float:
    """How far identity affines move a rest vertex because its skin weights do not sum to exactly one.

    The SMPL-X file stores float32 weights: their rows sum to one within about 4e-8.
    """
    defect = (identity.human.skin_matrix.sum(-1) - 1.0).abs()
    return float((defect[:, None] * identity.rest_vertices.abs()).max())


def test_regressed_rig_reproduces_its_rest_mesh_and_joints(tmp_path: Path) -> None:
    human = bh.HumanModel.from_spec(flame_spec(tmp_path), density=None)
    for name, identity in _identities(human).items():
        mesh, joints = _mesh_error(identity), _joint_error(identity)
        assert mesh <= EXACT + _weight_defect(identity), f"{name}: vertices at q_neutral differ from rest by {mesh} m"
        assert joints <= EXACT, f"{name}: skin_affines move the rest joints off the joints by {joints} m"


@pytest.mark.licensed
@pytest.mark.parametrize("family", list(LICENSED))
def test_skin_affines_carry_the_rest_joints_to_the_joints(family: str) -> None:
    human = bh.HumanModel.from_spec(LICENSED[family](), density=None)
    tolerance = MHR_AUTHORED_BINDS if family == "mhr" else EXACT
    for name, identity in _held("joints", family, _identities(human)).items():
        error = _joint_error(identity)
        assert error <= tolerance, f"{name}: skin_affines move the rest joints off the joints by {error} m"


@pytest.mark.licensed
@pytest.mark.parametrize("family", ["smpl", "smplx", "mhr"])  # soma: both identities in NOT_HELD
def test_forward_kinematics_at_q_neutral_reproduces_the_rest_mesh(family: str) -> None:
    human = bh.HumanModel.from_spec(LICENSED[family](), density=None)
    tolerance = MHR_AUTHORED_BINDS if family == "mhr" else EXACT
    for name, identity in _held("mesh", family, _identities(human)).items():
        error = _mesh_error(identity)
        bound = tolerance + _weight_defect(identity)
        assert error <= bound, f"{name}: vertices at q_neutral differ from rest_vertices by {error} m"
