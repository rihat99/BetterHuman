"""The studio's files: round trips, refused files, vertex range checks and the built-in segmentations (record 0009).

The format contract (versions, other formats, spec pages, fixtures) is ``tests/contract/test_formats.py``'s.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from better_robot.exceptions import ArgumentError, UnknownNameError

from better_human.io import FormatError
from better_human.studio import persistence

SMPL_NUM_VERTICES = 6890
SMPLX_NUM_VERTICES = 10475


def test_model_key_variants() -> None:
    assert persistence.model_key("smpl", gender="neutral") == "smpl_neutral"
    assert persistence.model_key("smplx", gender="female") == "smplx_female"
    assert persistence.model_key("mhr", lod=1) == "mhr_lod1"
    assert persistence.model_key("smpl") == "smpl"


def test_studio_paths_layout(tmp_path: Path) -> None:
    paths = persistence.StudioPaths(root=tmp_path)
    assert paths.segmentation_dir("smpl_neutral") == tmp_path / "segmentations" / "smpl_neutral"
    assert paths.frame_dir("smpl_neutral") == tmp_path / "frames" / "smpl_neutral"
    assert paths.pose_dir("mhr_lod1") == tmp_path / "poses" / "mhr_lod1"


def test_segmentation_round_trip_creates_the_folder(tmp_path: Path) -> None:
    path = tmp_path / "segmentations" / "smpl_neutral" / "custom.json"
    persistence.write_segmentation(path, {"head": [0, 1, 2], "torso": (3, 4)})
    assert persistence.read_segmentation(path) == {"head": [0, 1, 2], "torso": [3, 4]}
    assert json.loads(path.read_text())["format"] == "better_human.studio.segmentation"


def test_pose_round_trip_keeps_joint_names(tmp_path: Path) -> None:
    path = tmp_path / "poses" / "smpl_neutral" / "wave.json"
    pose = persistence.Pose(joint_names=("root", "left_hip"), q=(0.0, 1.0, 0.5, 0.0, 0.0, 0.0, 1.0))
    persistence.write_pose(path, pose)
    assert persistence.read_pose(path) == pose


def test_list_saved_sorted_and_missing(tmp_path: Path) -> None:
    assert persistence.list_saved(tmp_path / "nope") == {}
    directory = tmp_path / "poses" / "smpl_neutral"
    for stem in ("charlie", "alpha", "bravo"):
        persistence.write_pose(directory / f"{stem}.json", persistence.Pose(joint_names=("root",), q=(0.0,)))
    saved = persistence.list_saved(directory)
    assert list(saved) == ["alpha", "bravo", "charlie"]
    assert saved["alpha"] == directory / "alpha.json"


SEGMENTATION = {"format": "better_human.studio.segmentation", "format_version": 1}
POSE = {"format": "better_human.studio.pose", "format_version": 1}


@pytest.mark.parametrize(
    ("payload", "match", "read"),
    [
        pytest.param(
            {**SEGMENTATION, "parts": {"torso": [0, 1.5]}}, "'torso'", persistence.read_segmentation, id="float"
        ),
        pytest.param(
            {**SEGMENTATION, "parts": {"arm": []}},
            "'arm' must be a non-empty",
            persistence.read_segmentation,
            id="empty",
        ),
        pytest.param({**SEGMENTATION, "parts": {"leg": [-1]}}, "'leg'", persistence.read_segmentation, id="negative"),
        pytest.param({"head": 3}, "is not a version-0 segmentation", persistence.read_segmentation, id="not_version_0"),
        pytest.param({**SEGMENTATION}, "'parts' must map", persistence.read_segmentation, id="no_parts"),
        pytest.param(
            {"model_key": "smpl_neutral", "q": [0.0]},
            "format_version 0 has no migration",
            persistence.read_pose,
            id="pose_version_0",
        ),
        pytest.param(
            {**POSE, "joint_names": ["a"], "q": ["x"]}, "'q' must be a list of numbers", persistence.read_pose, id="q"
        ),
        pytest.param(
            {**POSE, "joint_names": "a", "q": [0.0]}, "'joint_names' must be", persistence.read_pose, id="names"
        ),
    ],
)
def test_a_malformed_file_is_refused(tmp_path: Path, payload: dict, match: str, read) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(FormatError, match=rf"{match}.*re-save it in the studio"):
        read(path)


@pytest.mark.parametrize("payload", [[1, 2, 3], "abc", 5, None], ids=["list", "string", "number", "null"])
@pytest.mark.parametrize("read", [persistence.read_pose, persistence.read_segmentation], ids=["pose", "segmentation"])
def test_json_that_is_not_an_object_is_a_format_error(tmp_path: Path, payload: object, read) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(FormatError, match=rf"bad\.json: .*{type(payload).__name__}.*re-save it in the studio"):
        read(path)


def test_a_segmentation_of_the_pre_format_studio_reads_as_version_1(tmp_path: Path) -> None:
    """Version 0 is the bare mapping the studio wrote before its files named a format; it migrates on read."""
    path = tmp_path / "old.json"
    path.write_text(json.dumps({"head": [0, 1, 2], "torso": [3, 4]}, indent=2))
    assert persistence.read_segmentation(path) == {"head": [0, 1, 2], "torso": [3, 4]}


def test_a_missing_file_is_an_argument_error(tmp_path: Path) -> None:
    with pytest.raises(ArgumentError, match="is not a file"):
        persistence.read_pose(tmp_path / "missing.json")


def test_vertex_ids_outside_the_mesh_raise_naming_the_source() -> None:
    persistence.check_vertex_ids([0, 99], 100, source="seg.json")
    with pytest.raises(ArgumentError, match=r"seg\.json: vertex ids \[100, 999\] are out of range .* 100 vertices"):
        persistence.check_vertex_ids([0, 999, 100], 100, source="seg.json")


@pytest.mark.parametrize(
    ("family", "num_vertices", "num_parts"),
    [pytest.param("smpl", SMPL_NUM_VERTICES, 24, id="smpl"), pytest.param("smplx", SMPLX_NUM_VERTICES, 27, id="smplx")],
)
def test_builtin_segmentation(family: str, num_vertices: int, num_parts: int) -> None:
    parts = persistence.builtin_segmentation(family)
    assert len(parts) == num_parts
    persistence.check_vertex_ids((vertex for ids in parts.values() for vertex in ids), num_vertices, source=family)


def test_builtin_segmentation_of_another_family_raises() -> None:
    with pytest.raises(
        UnknownNameError, match=r"unknown built-in segmentation 'mhr' \(valid names: \['smpl', 'smplx'\]\)"
    ):
        persistence.builtin_segmentation("mhr")


def test_frame_config_snippet_golden() -> None:
    snippet = persistence.frame_config_snippet({"nose": (331, None), "reye": (6260, "head")})
    expected = "contact_frames={\n    'nose': (331, None),\n    'reye': (6260, 'head'),\n}"
    assert snippet == expected


def test_frame_config_snippet_is_valid_python() -> None:
    frames = {"nose": (331, None), "reye": (6260, None), "leye": (2800, "head")}
    namespace: dict[str, object] = {}
    exec(persistence.frame_config_snippet(frames), {}, namespace)  # noqa: S102
    assert namespace["contact_frames"] == frames
    assert list(namespace["contact_frames"]) == list(frames)
