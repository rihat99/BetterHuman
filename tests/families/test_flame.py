"""FLAME loader on the new core, against a synthetic FLAME-shaped file (no FLAME asset ships).

Classic round trips run through tests/families/test_classic_formats.py.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch


import better_human as bh
from better_human.families import FLAMEClassic
from tests.support.synthetic import write_family


@pytest.fixture
def flame_path(tmp_path: Path) -> Path:
    return write_family(tmp_path, "flame")


def _body(path: Path, **options: int) -> bh.HumanModel:
    return bh.HumanModel.from_spec(bh.families.flame(path, dtype=torch.float64, **options), density=None)


def test_load(flame_path: Path) -> None:
    body = _body(flame_path)
    assert body.spec.joint_names == ("global", "neck", "jaw", "left_eye", "right_eye")
    assert body.spec.num_vertices == 5023
    assert body.spec.identity_groups == {"betas": 100}
    assert body.spec.stream_groups == {"expression": 50}


def test_widths_slice_and_validate(flame_path: Path) -> None:
    body = _body(flame_path, num_betas=7, num_expression=3)
    assert body.spec.identity_groups == {"betas": 7}
    assert body.spec.stream_groups == {"expression": 3}
    with pytest.raises(ValueError):
        bh.families.flame(flame_path, num_betas=301)
    with pytest.raises(ValueError):
        bh.families.flame(flame_path, num_expression=101)


def test_expression_stream_moves_vertices_and_joints(flame_path: Path) -> None:
    expression = torch.full((50,), 2.0, dtype=torch.float64)
    identity, q, stream = bh.from_classic(_body(flame_path), FLAMEClassic(expression=expression))
    torch.testing.assert_close(q[3:], identity.model.q_neutral[3:], atol=0, rtol=0)
    neutral = identity.forward(q)
    expressive = identity.forward(q, stream)
    assert (expressive.vertices - neutral.vertices).abs().max() > 1e-4
    assert (expressive.joint_translations - neutral.joint_translations).abs().max() > 1e-6
