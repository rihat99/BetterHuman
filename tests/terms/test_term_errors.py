"""Terms and states raise library errors that name the fix, on a synthetic FLAME model (no weights needed)."""

from __future__ import annotations

import pytest
import torch
from better_robot.exceptions import ArgumentError, UnknownNameError

import better_human as bh
from better_human import terms
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
