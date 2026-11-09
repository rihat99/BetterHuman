"""Part topologies: closed parts from authored faces or from a vertex labelling, checked when they are built."""

from __future__ import annotations

import pytest
import torch
from better_robot.exceptions import ArgumentError

from better_human.mass import PartTopology


def test_an_authored_part_indexing_past_the_joints_is_an_argument_error() -> None:
    vertices, joints = torch.eye(4, 3, dtype=torch.float64), torch.zeros(1, 3, dtype=torch.float64)
    faces = {0: torch.tensor([[0, 1, 2], [0, 2, 4], [0, 4, 1], [1, 4, 2]]), 1: torch.tensor([[0, 1, 5]])}
    with pytest.raises(ArgumentError, match=r"closed part 1 indexes outside \[0, 5\)"):
        PartTopology.from_closed_faces(faces, reference_vertices=vertices, reference_joints=joints)
