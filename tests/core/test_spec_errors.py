"""A misspelled joint name raises UnknownNameError listing the family's joints (synthetic FLAME, no weights)."""

from __future__ import annotations

from pathlib import Path

import pytest
from better_robot.exceptions import UnknownNameError

import better_human as bh
from tests.support.synthetic import flame_spec


def test_an_unknown_joint_name_raises_unknown_name_error(tmp_path: Path) -> None:
    spec = flame_spec(tmp_path)
    assert spec.joint_id("jaw") == 2
    with pytest.raises(UnknownNameError, match=r"unknown joint 'jwa'.*'jaw'"):
        spec.joint_id("jwa")
    with pytest.raises(UnknownNameError, match=r"unknown joint 'jwa'"):
        bh.families.drop_joints(spec, ["jwa"])
