"""Frame-set files (decision record 0013): the v1 round trip, the v0 migration, refusals, and the shipped set."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from better_robot.exceptions import UnknownNameError

import better_human as bh
from better_human.io import FormatError

FIXTURE = Path(__file__).parents[1] / "support" / "contacts.json"


def test_round_trip_keeps_order_and_joints(tmp_path: Path) -> None:
    frames = {"reye": (6260, "head"), "nose": (331, None)}
    path = tmp_path / "set.json"
    bh.io.write_frame_set(path, frames)
    payload = json.loads(path.read_text())
    assert (payload["format"], payload["format_version"]) == ("better_human.frame_set", 1)
    assert list(bh.io.read_frame_set(path).items()) == list(frames.items())
    assert not (tmp_path / "set.json.tmp").exists()


def test_unversioned_file_reads_through_its_migration(tmp_path: Path) -> None:
    path = tmp_path / "v0.json"
    path.write_text(
        json.dumps(
            {"legacy": 331, "default": {"vertex": 100, "joint": None}, "named": {"vertex": 200, "joint": "left_wrist"}}
        )
    )
    assert bh.io.read_frame_set(path) == {"legacy": (331, None), "default": (100, None), "named": (200, "left_wrist")}


def test_the_authored_fixture_reads() -> None:
    frames = bh.io.read_frame_set(FIXTURE)
    assert frames["knee_left"] == (3673, "left_knee")
    assert frames["heel_right"] == (8712, None)


def test_unknown_joint_names_pass_the_reader(tmp_path: Path) -> None:
    # The reader does not know the body; Body resolves (and refuses) joint names.
    path = tmp_path / "set.json"
    bh.io.write_frame_set(path, {"tip": (5, "not_a_real_joint")})
    assert bh.io.read_frame_set(path) == {"tip": (5, "not_a_real_joint")}


@pytest.mark.parametrize(
    ("frames", "match"),
    [
        pytest.param({"tip": {"vertex": 5, "joint": 7}}, r"'tip': joint 7 is not a joint name or null", id="joint"),
        pytest.param({"tip": {"vertex": True, "joint": None}}, r"'tip': vertex id True is not an int", id="bool"),
        pytest.param({"tip": {"vertex": 1.5, "joint": None}}, r"'tip': vertex id 1.5 is not an int", id="float"),
        pytest.param({"tip": [5, None]}, r"frame 'tip' must be \{vertex, joint\}", id="list"),
    ],
)
def test_malformed_entries_are_refused(tmp_path: Path, frames: dict, match: str) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"format": "better_human.frame_set", "format_version": 1, "frames": frames}))
    with pytest.raises(FormatError, match=match):
        bh.io.read_frame_set(path)


def test_writer_refuses_what_the_reader_refuses(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    with pytest.raises(FormatError, match="'tip': vertex id '5' is not an int"):
        bh.io.write_frame_set(path, {"tip": ("5", None)})
    assert not path.exists()


def test_shipped_frame_set_reads_package_data() -> None:
    frames = bh.io.shipped_frame_set("smplx_contacts")
    assert len(frames) == 35
    assert frames["heel_left"] == (8924, None)


def test_unknown_shipped_name_lists_the_shipped_names() -> None:
    with pytest.raises(UnknownNameError, match=r"unknown shipped frame set 'nope'; choose one of \['smplx_contacts'\]"):
        bh.io.shipped_frame_set("nope")
