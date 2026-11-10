"""Terms and states raise library errors that name the fix, on a synthetic FLAME model (no weights needed)."""

from __future__ import annotations

import dataclasses

import pytest
import torch
from better_robot.exceptions import ArgumentError, UnknownNameError

import better_human as bh
from better_human import terms
from better_human.core.rig import gravity
from tests.support.synthetic import flame_spec


@pytest.fixture(scope="module")
def human(tmp_path_factory: pytest.TempPathFactory) -> bh.HumanModel:
    return bh.HumanModel.from_spec(flame_spec(tmp_path_factory.mktemp("flame")), density=None)


def test_a_family_without_landmarks_raises_an_argument_error(human: bh.HumanModel) -> None:
    assert human.spec.landmarks is None
    state = human.bake().forward(human.model.q_neutral)
    fix = r"the flame family has no landmarks.*state\.points\(vertex_ids\).*joints=True"
    with pytest.raises(ArgumentError, match=fix):
        state.landmarks()
    with pytest.raises(ArgumentError, match=fix):
        terms.landmarks(target="t").fn(state, torch.zeros(0, 3, dtype=human.spec.dtype))
    camera = torch.eye(3, 4, dtype=human.spec.dtype)
    with pytest.raises(ArgumentError, match=fix):
        terms.reprojection(target="pixels").fn(state, torch.zeros(0, 2, dtype=human.spec.dtype), camera)


def test_misspelled_groups_raise_when_the_term_is_built(human: bh.HumanModel) -> None:
    with pytest.raises(UnknownNameError, match=r"unknown identity group 'beta'; valid identity groups: \['betas'\]"):
        terms.baked_identity(human, "beta")
    with pytest.raises(UnknownNameError, match=r"unknown stream group 'expresion'.*\['expression'\]"):
        terms.human_state(human.bake(), streams=["expresion"])
    assert terms.human_state(identity_name="identity", streams=["expresion"]).reads == ("identity", "q", "expresion")


def test_ground_defaults_to_the_family_up_axis(human: bh.HumanModel) -> None:
    z_up = bh.HumanModel.from_spec(dataclasses.replace(human.spec, up_axis="+z"), density=None)
    for model, up in ((human, (0.0, 1.0, 0.0)), (z_up, (0.0, 0.0, 1.0))):
        state = model.bake().forward(model.model.q_neutral)
        torch.testing.assert_close(terms.ground(hinge=False).fn(state), terms.ground(up=up, hinge=False).fn(state))


@pytest.mark.parametrize("up_axis", ["y", "+w", "up", ""])
def test_gravity_of_an_unknown_up_axis_is_an_argument_error(up_axis: str) -> None:
    with pytest.raises(ArgumentError, match=r"is not one of \['\+x', '-x', '\+y'"):
        gravity(up_axis)
