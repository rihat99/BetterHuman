"""Frame-set files (decision record 0013): the v1 round trip, the v0 migration, refusals, and the shipped set."""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path

import numpy as np
import pytest
from better_robot.exceptions import ArgumentError, UnknownNameError

import better_human as bh
from better_human.families import _tables as tables
from better_human.io import FormatError

FIXTURE = Path(__file__).parents[1] / "support" / "contacts.json"
SHIPPED = sorted(
    path.stem for path in (resources.files("better_human.io") / "assets").iterdir() if path.suffix == ".json"
)


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


def test_an_added_key_inside_an_entry_is_ignored(tmp_path: Path) -> None:
    path = tmp_path / "set.json"
    entry = {"vertex_id": 5, "joint_name": None, "color": "red"}
    path.write_text(json.dumps({"format": "better_human.frame_set", "format_version": 1, "frames": {"tip": entry}}))
    assert bh.io.read_frame_set(path) == {"tip": (5, None)}


def test_unknown_joint_names_pass_the_reader(tmp_path: Path) -> None:
    # The reader does not know the family; HumanModel.from_spec resolves (and refuses) joint names.
    path = tmp_path / "set.json"
    bh.io.write_frame_set(path, {"tip": (5, "not_a_real_joint")})
    assert bh.io.read_frame_set(path) == {"tip": (5, "not_a_real_joint")}


@pytest.mark.parametrize(
    ("frames", "match"),
    [
        pytest.param(
            {"tip": {"vertex_id": 5, "joint_name": 7}}, r"'tip': joint_name 7 is not a string or null", id="joint"
        ),
        pytest.param(
            {"tip": {"vertex_id": True, "joint_name": None}}, r"'tip': vertex id True is not an int", id="bool"
        ),
        pytest.param(
            {"tip": {"vertex_id": 1.5, "joint_name": None}}, r"'tip': vertex id 1.5 is not an int", id="float"
        ),
        pytest.param({"tip": [5, None]}, r"frame 'tip' must be \{vertex_id, joint_name\}", id="list"),
    ],
)
def test_malformed_entries_are_refused(tmp_path: Path, frames: dict, match: str) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"format": "better_human.frame_set", "format_version": 1, "frames": frames}))
    with pytest.raises(FormatError, match=match):
        bh.io.read_frame_set(path)


FIX = r"Write frame sets with better_human\.io\.write_frame_set"


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        pytest.param([1, 2], rf"a frame set is a JSON object, got list\. {FIX}", id="list"),
        pytest.param({"tip": "5"}, rf"names no format and is not a version-0 frame set .*\. {FIX}", id="v0_string"),
        pytest.param({"tip": 5, "note": None}, rf"is not a version-0 frame set .*\. {FIX}", id="v0_null"),
        pytest.param(
            {"format": "better_human.frame_set", "frames": {"tip": {"vertex_id": 5, "joint_name": None}}},
            r"a better_human\.frame_set file without format_version",
            id="no_version",
        ),
        pytest.param(
            {"format": "better_human.frame_set", "format_version": 1, "frames": [5]},
            rf"'frames' must map .*{FIX}",
            id="frames",
        ),
    ],
)
def test_a_file_that_is_not_frame_shaped_is_refused(tmp_path: Path, payload: object, match: str) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(FormatError, match=match):
        bh.io.read_frame_set(path)


def test_writer_refuses_what_the_reader_refuses(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    with pytest.raises(FormatError, match="'tip': vertex id '5' is not an int"):
        bh.io.write_frame_set(path, {"tip": ("5", None)})
    assert not path.exists()


@pytest.mark.parametrize("content", [b'{"tip": 5,', bytes(range(256))], ids=["truncated", "binary"])
def test_a_file_that_is_not_json_is_a_format_error(tmp_path: Path, content: bytes) -> None:
    path = tmp_path / "set.json"
    path.write_bytes(content)
    with pytest.raises(FormatError, match=rf"set\.json: not a JSON file .*{FIX}"):
        bh.io.read_frame_set(path)


def test_a_missing_file_is_an_argument_error_naming_the_path(tmp_path: Path) -> None:
    with pytest.raises(ArgumentError, match=r"absent\.json is not a file; pass the path of a frame-set file"):
        bh.io.read_frame_set(tmp_path / "absent.json")


@pytest.mark.parametrize("value", [5, (5,), (5, None, "x"), "ab"])
def test_writer_refuses_a_value_that_is_not_a_pair(tmp_path: Path, value: object) -> None:
    with pytest.raises(ArgumentError, match=r"frame 'tip' must be a pair \(vertex_id, joint_name \| None\)"):
        bh.io.write_frame_set(tmp_path / "set.json", {"tip": value})


def test_writer_takes_any_integer_vertex_id(tmp_path: Path) -> None:
    path = tmp_path / "set.json"
    bh.io.write_frame_set(path, {"tip": (np.int64(5), None)})
    assert bh.io.read_frame_set(path) == {"tip": (5, None)}


def test_writer_refuses_a_negative_vertex_id(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    with pytest.raises(FormatError, match=r"'tip': vertex id -1 is not an int >= 0"):
        bh.io.write_frame_set(path, {"tip": (-1, None)})
    assert not path.exists()


def test_shipped_frame_set_reads_package_data() -> None:
    frames = bh.io.shipped_frame_set("smplx_contacts")
    assert len(frames) == 35
    assert frames["heel_left"] == (8924, None)


def test_shipped_frame_names_are_not_body_names() -> None:
    """BetterRobot refuses a frame named like a body (its record 0018), and every family body carries its joint's name.

    A shipped set's family is its name's first word (``smplx_contacts`` -> SMPL-X).
    """
    for name in SHIPPED:
        table = getattr(tables, name.split("_")[0].upper())
        clashes = sorted(set(bh.io.shipped_frame_set(name)) & set(table.joint_names))
        assert not clashes, f"shipped frame set {name!r} names joints of {table.name}: {clashes}"


def test_unknown_shipped_name_lists_the_shipped_names() -> None:
    with pytest.raises(
        UnknownNameError, match=r"unknown shipped frame set 'nope' \(valid names: \['smplx_contacts'\]\)"
    ):
        bh.io.shipped_frame_set("nope")
