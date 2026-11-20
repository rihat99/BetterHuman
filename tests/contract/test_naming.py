"""BetterRobot's vocabulary on BetterHuman's three public records: their field tables, field by field.

Plural fields for per-joint and per-vertex tensors, ``_translations`` for positions, ``_ids`` for integer ids,
``_affines`` for ``(…, 3, 4)`` maps. A field added, renamed or removed changes this table in the same commit.
"""

from __future__ import annotations

import dataclasses

import pytest

import better_human as bh

FIELDS = {
    bh.HumanModel: (
        "spec",
        "model",
        "skin_matrix",
        "joint_ids",
        "part_body_ids",
        "part_joint_ids",
        "density",
        "part_topology",
        "contact_frames",
        "stream_joint_bases",
    ),
    bh.Identity: (
        "human",
        "values",
        "rest_vertices",
        "rest_joint_translations",
        "inverse_bind_affines",
        "root_offset",
        "model",
        "_frame_placements",
    ),
    bh.HumanState: (
        "identity",
        "q",
        "streams",
        "model",
        "kinematics",
        "rest_joint_translations",
        "inverse_bind_affines",
        "root_offset",
        "stream_offsets",
        "_memo",
    ),
}


@pytest.mark.parametrize("cls", list(FIELDS), ids=lambda cls: cls.__name__)
def test_field_table(cls: type) -> None:
    assert tuple(field.name for field in dataclasses.fields(cls)) == FIELDS[cls]
