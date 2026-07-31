"""Persistence round-trips, validation, and built-in asset loading for the studio."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from better_human.viz.studio import persistence

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


def test_segmentation_round_trip(tmp_path: Path) -> None:
    parts = {"head": [0, 1, 2], "left_hand": [10, 11]}
    path = tmp_path / "seg" / "custom.json"
    persistence.save_segmentation(path, parts)
    assert path.is_file()
    assert persistence.load_segmentation(path, num_vertices=100) == parts


def test_segmentation_save_accepts_tuple_values(tmp_path: Path) -> None:
    path = tmp_path / "custom.json"
    persistence.save_segmentation(path, {"torso": (3, 4, 5)})
    assert persistence.load_segmentation(path, num_vertices=100) == {"torso": [3, 4, 5]}


def test_frame_set_round_trip(tmp_path: Path) -> None:
    frames = {"nose": 331, "reye": 6260}
    path = tmp_path / "frames" / "config.json"
    persistence.save_frame_set(path, frames)
    assert path.is_file()
    assert persistence.load_frame_set(path, num_vertices=SMPL_NUM_VERTICES) == frames


def test_pose_round_trip(tmp_path: Path) -> None:
    q = torch.linspace(-1.0, 1.0, 99, dtype=torch.float32)
    path = tmp_path / "poses" / "wave.json"
    persistence.save_pose(path, q, model_key="smpl_neutral")
    loaded = persistence.load_pose(path, model_key="smpl_neutral", nq=99)
    assert loaded.dtype == torch.float32
    torch.testing.assert_close(loaded, q)


def test_saved_json_uses_indent(tmp_path: Path) -> None:
    path = tmp_path / "seg.json"
    persistence.save_segmentation(path, {"head": [0, 1]})
    assert "\n  " in path.read_text()


def test_list_saved_sorted_and_missing(tmp_path: Path) -> None:
    assert persistence.list_saved(tmp_path / "nope") == {}
    directory = tmp_path / "poses"
    for stem in ("charlie", "alpha", "bravo"):
        persistence.save_pose(directory / f"{stem}.json", torch.zeros(3), model_key="smpl_neutral")
    saved = persistence.list_saved(directory)
    assert list(saved) == ["alpha", "bravo", "charlie"]
    assert saved["alpha"] == directory / "alpha.json"


def test_segmentation_out_of_range_id_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"head": [0, 999]}))
    with pytest.raises(ValueError, match="'head'.*out of range"):
        persistence.load_segmentation(path, num_vertices=100)


def test_segmentation_non_int_id_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"torso": [0, 1.5]}))
    with pytest.raises(ValueError, match="'torso'.*not an int"):
        persistence.load_segmentation(path, num_vertices=100)


def test_segmentation_empty_part_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"left_arm": []}))
    with pytest.raises(ValueError, match="'left_arm'.*empty"):
        persistence.load_segmentation(path, num_vertices=100)


def test_frame_set_out_of_range_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"nose": 10000}))
    with pytest.raises(ValueError, match="'nose'.*out of range"):
        persistence.load_frame_set(path, num_vertices=SMPL_NUM_VERTICES)


def test_pose_model_key_mismatch_raises(tmp_path: Path) -> None:
    path = tmp_path / "pose.json"
    persistence.save_pose(path, torch.zeros(99), model_key="smpl_neutral")
    with pytest.raises(ValueError, match="model_key.*does not match"):
        persistence.load_pose(path, model_key="smplx_female", nq=99)


def test_pose_nq_mismatch_raises(tmp_path: Path) -> None:
    path = tmp_path / "pose.json"
    persistence.save_pose(path, torch.zeros(99), model_key="smpl_neutral")
    with pytest.raises(ValueError, match="expected nq=132"):
        persistence.load_pose(path, model_key="smpl_neutral", nq=132)


def test_builtin_smpl_segmentation() -> None:
    parts = persistence.load_builtin_segmentation("smpl_vert_segmentation.json", SMPL_NUM_VERTICES)
    assert len(parts) == 24
    ids = [vid for part in parts.values() for vid in part]
    assert all(isinstance(vid, int) and 0 <= vid < SMPL_NUM_VERTICES for vid in ids)


def test_builtin_smplx_segmentation() -> None:
    parts = persistence.load_builtin_segmentation("smplx_vert_segmentation.json", SMPLX_NUM_VERTICES)
    assert len(parts) == 27
    ids = [vid for part in parts.values() for vid in part]
    assert all(isinstance(vid, int) and 0 <= vid < SMPLX_NUM_VERTICES for vid in ids)


def test_builtin_smpl_range_check_rejects_too_small_v() -> None:
    with pytest.raises(ValueError, match="out of range"):
        persistence.load_builtin_segmentation("smpl_vert_segmentation.json", num_vertices=100)


def test_frame_config_snippet_golden() -> None:
    snippet = persistence.frame_config_snippet({"nose": 331, "reye": 6260})
    expected = (
        "frame_names=(\n"
        '    "nose",\n'
        '    "reye",\n'
        ")\n"
        "frame_vertex_ids=(\n"
        "    331,\n"
        "    6260,\n"
        ")"
    )
    assert snippet == expected


def test_frame_config_snippet_is_valid_python() -> None:
    frames = {"nose": 331, "reye": 6260, "leye": 2800}
    namespace: dict[str, object] = {}
    exec(persistence.frame_config_snippet(frames), {}, namespace)  # noqa: S102
    assert namespace["frame_names"] == tuple(frames.keys())
    assert namespace["frame_vertex_ids"] == tuple(frames.values())
