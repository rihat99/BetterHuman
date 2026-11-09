"""Decision record 0001: the reprs of ``HumanModel``, ``Identity`` and ``HumanState`` are one short line each."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import better_human as bh
from tests.support.synthetic import flame_spec


@pytest.mark.slow
def test_reprs_are_one_short_line(tmp_path: Path) -> None:
    human = bh.HumanModel.from_spec(flame_spec(tmp_path), density=None)
    identity = human.bake(betas=torch.zeros(2, 100, dtype=torch.float64))
    state = identity.forward(identity.model.q_neutral.expand(3, 2, identity.model.nq))
    texts = {
        "HumanModel": (repr(human), "HumanModel(flame, vertices=50, joints=5, dtype=torch.float64, device=cpu)"),
        "Identity": (repr(identity), "Identity(flame, batch=(2,))"),
        "HumanState": (repr(state), "HumanState(flame, batch=(3, 2))"),
    }
    for name, (text, expected) in texts.items():
        assert "\n" not in text and len(text) < 120, name
        assert text == expected, name
