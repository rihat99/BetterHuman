"""A misspelled joint name raises UnknownNameError listing the family's joints; ``drop_joints`` takes a list of
distinct names (ArgumentError). Synthetic FLAME, no weights."""

from __future__ import annotations

from pathlib import Path

import pytest
from better_robot.exceptions import ArgumentError, UnknownNameError

import better_human as bh
from tests.support.synthetic import flame_spec


def test_an_unknown_joint_name_raises_unknown_name_error(tmp_path: Path) -> None:
    spec = flame_spec(tmp_path)
    assert spec.joint_id("jaw") == 2
    with pytest.raises(UnknownNameError, match=r"unknown joint 'jwa'.*'jaw'"):
        spec.joint_id("jwa")
    with pytest.raises(UnknownNameError, match=r"unknown joint 'jwa'"):
        bh.families.drop_joints(spec, ["jwa"])


def test_drop_joints_takes_a_list_of_distinct_names(tmp_path: Path) -> None:
    spec = flame_spec(tmp_path)
    with pytest.raises(ArgumentError, match=r"got the string 'jaw'; pass \['jaw'\]"):
        bh.families.drop_joints(spec, "jaw")
    with pytest.raises(ArgumentError, match=r"names \['jaw'\] more than once"):
        bh.families.drop_joints(spec, ["jaw", "left_eye", "jaw"])


def test_an_unknown_contact_frame_joint_raises_at_from_spec(tmp_path: Path) -> None:
    with pytest.raises(UnknownNameError, match=r"unknown parent joint 'nekc' \(valid names: \['global'"):
        bh.HumanModel.from_spec(flame_spec(tmp_path), density=None, contact_frames={"tip": (0, "nekc")})
