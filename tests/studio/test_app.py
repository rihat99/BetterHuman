"""Headless HumanStudio tests on BetterRobot's stand-in viser server, ``better_robot.testing.FakeServer``.

The unmarked tests run the app on the random-weight SMPL archive in the full tier, found through
``$BETTERHUMAN_MODELS_DIR`` as a user's weights are; the ``licensed`` ones on the real weights.
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
import viser

from better_robot.exceptions import ArgumentError, ShapeError, UnknownNameError
from better_robot.kinematics import integrate
from better_robot.lie import so3
from better_robot.testing import FakeServer

from better_human.studio import app, launch
from better_human.studio import layers as studio_layers
from better_human.studio import state as studio_state
from better_human.studio.__main__ import main
from better_human.studio.app import HumanStudio


@pytest.fixture
def studio(tmp_path: Path):
    return HumanStudio(server=FakeServer(), data_root=tmp_path, generator=np.random.default_rng(0))


SMPL_RANDOM = Path(__file__).parents[1] / "support" / "random_weights" / "smpl_random.npz"


@pytest.fixture
def random_models(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The random SMPL archive placed where ``families.smpl`` looks for the neutral and female models."""
    models = tmp_path / "models"
    (models / "smpl").mkdir(parents=True)
    for gender in ("NEUTRAL", "FEMALE"):
        shutil.copy(SMPL_RANDOM, models / "smpl" / f"SMPL_{gender}.npz")
    monkeypatch.setenv("BETTERHUMAN_MODELS_DIR", str(models))
    return models


@pytest.fixture
def random_studio(tmp_path: Path, random_models: Path) -> HumanStudio:
    """The app on the random SMPL archive."""
    return HumanStudio(server=FakeServer(), data_root=tmp_path / "data", generator=np.random.default_rng(0))


def test_the_studio_runs_on_the_stand_in_server(random_studio: HumanStudio) -> None:
    """Smoke test: load, pose, shape, segment, frame and save on the random-weight archive."""
    studio = random_studio
    assert studio.status.startswith("smpl_neutral loaded")
    joint = studio.session.state.joint_translations.detach().reshape(-1, 3)[5].numpy()
    _click(studio, joint + np.array([0.0, 0.0, 2.0]), joint)
    assert studio.gizmo.joint == 5
    before = studio.session.vertices.clone()
    studio.random_shape("betas")
    assert not torch.equal(before, studio.session.vertices)
    studio.segment.load_builtin()
    assert len(studio.session.part_names) == 24
    studio.segment.save("builtin")
    studio.frames.add_at(studio.session.vertices[100].detach().numpy())
    studio.frames.save("frames")
    path = studio.save_pose("pose")
    saved = sorted(file.relative_to(studio.paths.root).parts for file in studio.paths.root.rglob("*.json"))
    assert saved == [
        ("frames", "smpl_neutral", "frames.json"),
        ("poses", "smpl_neutral", "pose.json"),
        ("segmentations", "smpl_neutral", "builtin.json"),
    ]
    assert json.loads(path.read_text())["joint_names"] == list(studio.session.identity.model.joint_names)


def test_every_control_drives_the_state_it_names(random_studio: HumanStudio) -> None:
    """Smoke test: every GUI control and the pose gizmo is fired as the browser fires it, and the
    session field, layer, scene node or file it drives is checked. A control with a callback that is neither removed
    nor fired fails.
    """
    studio = random_studio
    fired: list[Any] = []
    fire = partial(_fire, studio, fired)
    _body_section(studio, fire)
    _gizmo_and_selection_row(studio, fire)
    _display_section(studio, fire)
    _segment_section(studio, fire)
    _frames_section(studio, fire)
    _layers_folder(studio, fire)
    _model_section(studio, fire, fired)


def _fire(studio: HumanStudio, fired: list[Any], control: Any, *value: object) -> Any:
    """Set ``value`` on the control (a label or a handle), if given, fire it as the browser does, and add it to
    ``fired``."""
    handle = _control(studio, control) if isinstance(control, str) else control
    if value:
        handle.value = value[0]
    handle.fire()
    fired.append(handle)
    return handle


def _control(studio: HumanStudio, label: str) -> Any:
    """The one control labelled ``label`` that was not removed: a rebuilt layer leaves its removed controls behind."""
    (control,) = [
        control for control in studio.viewer.server.gui.controls if control.name == label and not control.removed
    ]
    return control


def _body_section(studio: HumanStudio, fire: Any) -> None:
    """The Body and Pose sections: shape sliders and buttons, density, saved poses."""
    session = studio.session
    for k in range(10):
        fire(f"betas[{k}]", 0.5 * (k + 1))
        assert session.shape["betas"][k].item() == 0.5 * (k + 1)
    fire("Reset betas")
    assert not session.shape["betas"].any() and _control(studio, "betas[0]").value == 0.0
    fire("Random betas")
    drawn = [_control(studio, f"betas[{k}]").value for k in range(10)]
    assert any(drawn) and session.shape["betas"][:10].tolist() == pytest.approx(drawn)
    _control(studio, "Density table").value = "Dempster"
    fire("Apply density")
    assert studio.density == "Dempster" and session.body.density.numel() > 1
    tangent = torch.zeros(session.identity.model.nv)
    tangent[6 + 3 * 15] = 0.7
    posed = integrate(session.identity.model, session.q, tangent)
    session.set_q(posed)
    _control(studio, "Pose name").value = "p1"
    fire("Save pose")
    assert (studio.paths.pose_dir("smpl_neutral") / "p1.json").is_file() and studio.saved_poses.value == "p1"
    fire("Reset pose")
    assert torch.equal(session.q, session.identity.model.q_neutral)
    _control(studio, "Saved poses").value = "p1"
    fire("Load pose")
    assert torch.allclose(session.q, posed)


def _gizmo_and_selection_row(studio: HumanStudio, fire: Any) -> None:
    """The pose gizmo (picked by a click) and the Status section's joint buttons."""
    session = studio.session
    joint = session.state.joint_translations.detach().reshape(-1, 3)[5].numpy()
    _click(studio, joint + np.array([0.0, 0.0, 2.0]), joint)
    assert studio.gizmo.joint == 5
    before = session.vertices.clone()
    handle = studio.gizmo.gizmo.handle
    handle.wxyz = (np.cos(0.2), np.sin(0.2), 0.0, 0.0)
    fire(handle)
    studio.gizmo.gizmo.wait()
    assert (session.vertices - before).abs().max() > 1e-3
    fire("Deselect joint")
    assert studio.gizmo.joint is None
    fire("Reset pose")


def _display_section(studio: HumanStudio, fire: Any) -> None:
    """The Display section: opacity, joint radius, explode (the colour modes are fired with the Segment section)."""
    scene = studio.viewer.server.scene
    fire("Opacity", 0.6)
    mesh_node = scene.nodes["/body/mesh/mesh"]
    assert studio.opacity == 0.6 and studio.layers.mesh.opacity == pytest.approx(0.6)
    assert mesh_node.opacity == pytest.approx(0.6)
    fire("Joint radius", 0.03)
    assert studio.layers.radius == 0.03 and np.all(studio.layers.joints.radii == np.float32(0.03))
    fire("Explode", 0.5)
    assert studio.layers.explode == 0.5
    fire("Explode", 0.0)


def _segment_section(studio: HumanStudio, fire: Any) -> None:
    """The Segment section and the Display section's colour modes, as the scene draws them: parts, painting, files,
    marquee, bodies."""
    session = studio.session
    scene = studio.viewer.server.scene
    fire("Load built-in")
    assert len(session.part_names) == 24
    fire("Colour", "segmentation")
    assert studio.layers.mode == "segmentation" and scene.nodes["/body/mesh/mesh"].kind == "GlbHandle"
    assert len(np.unique(studio.layers.mesh.colors, axis=0)) == 24
    fire("Colour", "parts")
    assert studio.layers.mode == "parts"
    neutral = session.identity.forward(session.identity.model.q_neutral).vertices.detach()
    for index, (part, used, _, _) in enumerate(studio_layers.part_meshes(session, neutral)[0]):
        # A part mesh rides its joint rigidly: at the neutral pose it lies on the posed mesh.
        node = scene.nodes[f"/body/mesh/mesh_{index}"]
        wxyz = np.asarray(node.wxyz, dtype=np.float32)
        rotation = so3.to_matrix(torch.from_numpy(wxyz[[1, 2, 3, 0]])).numpy()
        world = node.props["vertices"][: len(used)] @ rotation.T + np.asarray(node.position)
        np.testing.assert_allclose(world, neutral[used].numpy(), atol=1e-4, err_msg=f"part {part}")
    fire("Colour", "skin")
    fire("Part", session.part_names[3])
    assert studio.segment.part == session.part_names[3]
    _control(studio, "New part").value = "blob"
    fire("Add part")
    assert session.part_names[-1] == "blob" and studio.segment.part == "blob"
    fire("Paint (click mesh)", True)
    assert studio.segment.painting and studio.mode == "segment"
    _control(studio, "Brush radius").value = 0.05
    center = session.vertices[0].detach().numpy()
    _click(studio, center + np.array([0.0, 0.0, 2.0]), center)
    assert int((session.part_ids == session.part_names.index("blob")).sum()) > 0
    fire("Paint (click mesh)", False)
    assert not studio.segment.painting and studio.mode == "pose"
    studio.segment.save_name.value = "seg1"
    fire("Save segmentation")
    assert (studio.paths.segmentation_dir("smpl_neutral") / "seg1.json").is_file()
    fire("Reset to skinning")
    assert "blob" not in session.part_names
    studio.segment.saved.value = "seg1"
    fire("Load segmentation")
    assert session.part_names[-1] == "blob"
    fire("Marquee select")
    assert _pointer_events(studio) == ["rect-select"]
    camera = SimpleNamespace(position=(0.0, -0.3, 4.0), wxyz=(0.0, 0.0, 1.0, 0.0), fov=1.0, aspect=1.0)
    rect = SimpleNamespace(client=SimpleNamespace(camera=camera), screen_min=(0.0, 0.0), screen_max=(1.0, 1.0))
    _pointer(studio, "rect-select")(rect)
    assert bool((session.part_ids == session.part_names.index("blob")).all())
    assert _pointer_events(studio) == ["click"]
    fire("Load built-in")
    fire("Build bodies")
    assert "| total |" in studio.segment.mass_table.content


def _frames_section(studio: HumanStudio, fire: Any) -> None:
    """The Frames section and the Status section's frame button."""
    session = studio.session
    fire("Add frames (click mesh)", True)
    assert studio.frames.adding and studio.mode == "frames"
    for vertex in (100, 2000):
        point = session.vertices[vertex].detach().numpy()
        _click(studio, point + np.array([0.0, 0.0, 2.0]), point)
    first, second = session.frame_set
    fire("Add frames (click mesh)", False)
    assert not studio.frames.adding and studio.mode == "pose"
    fire("Frame", first)
    assert studio.frames.active == first
    fire("Parent joint", session.body.spec.joint_names[1])
    assert session.frame_set[first].joint == 1
    _control(studio, "Name").value = "toe"
    fire("Rename")
    assert list(session.frame_set) == ["toe", second] and studio.frames.active == "toe"
    studio.frames.save_name.value = "set1"
    fire("Save frames")
    assert (studio.paths.frame_dir("smpl_neutral") / "set1.json").is_file()
    fire("Export as config")
    assert "'toe': (" in studio.frames.export_box.content
    fire("Delete")
    assert list(session.frame_set) == [second]
    studio.frames.saved.value = "set1"
    fire("Load frames")
    assert list(session.frame_set) == ["toe", second]
    fire("Frame", second)
    fire("Deselect frame")
    assert studio.frames.active is None


def _layers_folder(studio: HumanStudio, fire: Any) -> None:
    """The viewer's Layers checkboxes, one per tag."""
    gui = studio.viewer.server.gui
    for tag in ("body", "skeleton", "selection", "frames"):
        (box,) = [c for c in gui.controls if c.name == tag.capitalize() and c.kind == "GuiCheckboxHandle"]
        tagged = [layer for layer in studio.viewer.layers if tag in layer.tags]
        fire(box, False)
        assert tagged and not any(layer.visible for layer in tagged)
        fire(box, True)
        assert all(layer.visible for layer in tagged)


def _model_section(studio: HumanStudio, fire: Any, fired: list[Any]) -> None:
    """The Model section; Load rebuilds the sections after it, so it goes last, once every other control was fired."""
    session = studio.session
    gui = studio.viewer.server.gui
    fire("Family", "mhr")
    assert _control(studio, "Gender").disabled and not _control(studio, "LOD").disabled
    fire("Family", "smpl")
    assert not _control(studio, "Gender").disabled and _control(studio, "LOD").disabled
    unfired = [
        control.name for control in gui.controls if not control.removed and control.callbacks and control not in fired
    ]
    assert unfired == ["Load"]
    _control(studio, "Gender").value = "female"
    _control(studio, "LOD").value = 3
    fire("Load")
    assert (session.model_key, studio.lod) == ("smpl_female", 3)


def test_an_error_that_is_not_the_users_escapes_the_callback(random_studio: HumanStudio) -> None:
    """``locked`` shows a BetterRobotError on the status line and lets any other exception through."""
    with pytest.raises(ZeroDivisionError):
        random_studio.locked(lambda _event: 1 / 0)(None)


def test_the_same_seed_draws_the_same_shape(random_studio: HumanStudio, tmp_path: Path) -> None:
    other = HumanStudio(server=FakeServer(), data_root=tmp_path / "other", generator=np.random.default_rng(0))
    random_studio.random_shape("betas")
    other.random_shape("betas")
    assert torch.equal(random_studio.session.vertices, other.session.vertices)


def test_a_pose_saved_on_other_joints_is_refused_on_the_status_line(random_studio: HumanStudio) -> None:
    path = random_studio.save_pose("pose")
    payload = json.loads(path.read_text())
    path.write_text(json.dumps({**payload, "joint_names": payload["joint_names"][:-1]}))
    with pytest.raises(ArgumentError, match="follows other joints than the smpl_neutral model's"):
        random_studio.load_pose("pose")
    random_studio.locked(lambda _event: random_studio.load_pose("pose"))(None)
    assert random_studio.status.startswith("ArgumentError: ") and "other joints" in random_studio.status


def test_a_pose_of_another_length_is_refused_and_leaves_the_session_unchanged(random_studio: HumanStudio) -> None:
    studio = random_studio
    nq = studio.session.identity.model.nq
    path = studio.save_pose("short")
    payload = json.loads(path.read_text())
    path.write_text(json.dumps({**payload, "q": payload["q"][:-1]}))
    q = studio.session.q.clone()
    studio.locked(lambda _event: studio.load_pose("short"))(None)
    assert studio.status.startswith("ArgumentError: ") and f"holds {nq - 1} values of q" in studio.status
    assert "re-save it in the studio" in studio.status
    assert torch.equal(studio.session.q, q) and torch.equal(studio.session.state.q, q)
    assert len(json.loads(studio.save_pose("after").read_text())["q"]) == nq


def test_a_refused_pose_edit_leaves_the_session_unchanged(random_studio: HumanStudio) -> None:
    session = random_studio.session
    q, state = session.q.clone(), session.state
    with pytest.raises(ShapeError):
        session.set_q(q[:-1])
    assert torch.equal(session.q, q) and session.state is state


def test_a_segmentation_of_the_pre_format_studio_loads_and_saves_as_version_1(random_studio: HumanStudio) -> None:
    studio = random_studio
    directory = studio.paths.segmentation_dir(studio.session.model_key)
    directory.mkdir(parents=True)
    (directory / "old.json").write_text(json.dumps({"head": [0, 1, 2], "rest": list(range(3, 6890))}, indent=2))
    studio.locked(lambda _event: studio.segment.load("old"))(None)
    assert studio.session.part_names == ("head", "rest")
    studio.segment.save("old")
    saved = json.loads((directory / "old.json").read_text())
    assert (saved["format"], saved["format_version"], list(saved["parts"])) == (
        "better_human.studio.segmentation",
        1,
        ["head", "rest"],
    )


@pytest.mark.parametrize(
    ("folder", "payload", "load", "match"),
    [
        pytest.param("poses", [1, 2, 3], "pose", "FormatError: .*got list", id="pose_list"),
        pytest.param(
            "poses",
            {"model_key": "smpl_neutral", "q": []},
            "pose",
            "FormatError: .*no migration .*pose the model again in the studio and save it",
            id="pose_v0",
        ),
        pytest.param("segmentations", "abc", "segment", "FormatError: .*got str", id="segmentation_string"),
        pytest.param(
            "frames",
            {
                "format": "better_human.frame_set",
                "format_version": 1,
                "frames": {"toe": {"vertex_id": 1, "joint_name": "x"}},
            },
            "frames",
            "UnknownNameError: unknown parent joint in .*'x'",
            id="frame_set_unknown_joint",
        ),
    ],
)
def test_a_refused_file_shows_on_the_status_line(
    random_studio: HumanStudio, folder: str, payload: object, load: str, match: str
) -> None:
    studio = random_studio
    directory = studio.paths.root / folder / studio.session.model_key
    directory.mkdir(parents=True)
    (directory / "bad.json").write_text(json.dumps(payload))
    loader = {"pose": studio.load_pose, "segment": studio.segment.load, "frames": studio.frames.load}[load]
    studio.locked(lambda _event: loader("bad"))(None)
    assert re.match(match, studio.status), studio.status


@pytest.mark.parametrize("name", ["../x", "absolute", ""], ids=["parent", "absolute", "empty"])
def test_a_save_name_that_is_not_one_file_name_is_refused_and_writes_nothing(
    random_studio: HumanStudio, tmp_path: Path, name: str
) -> None:
    """Rule H-STUDIO-2: each Save button refuses a name that would leave ``<data root>/<kind>/<model key>/``."""
    studio = random_studio
    name = str(tmp_path / "outside" / "x") if name == "absolute" else name
    studio.segment.load_builtin()
    studio.frames.add_at(studio.session.vertices[100].detach().numpy())
    for field, button in (
        (studio.pose_name, "Save pose"),
        (studio.segment.save_name, "Save segmentation"),
        (studio.frames.save_name, "Save frames"),
    ):
        field.value = name
        studio.set_status("")
        _control(studio, button).fire()
        assert studio.status.startswith(f"ArgumentError: the name {name!r} is not one file name"), studio.status
    assert sorted(tmp_path.rglob("*.json")) == []


@pytest.mark.parametrize("family", ["soma", "mano", "nope"])
def test_a_family_the_studio_does_not_edit_is_an_unknown_name(
    random_studio: HumanStudio, tmp_path: Path, family: str
) -> None:
    with pytest.raises(UnknownNameError, match="unknown studio family"):
        HumanStudio(server=FakeServer(), data_root=tmp_path, generator=np.random.default_rng(0), family=family)
    random_studio.locked(lambda _event: random_studio.load_model(family))(None)
    assert random_studio.status.startswith("UnknownNameError: ") and random_studio.session.model_key == "smpl_neutral"


def test_launch_and_the_command_line_serve_the_studio(
    random_models: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """``studio.launch`` and ``python -m better_human.studio`` build the app, print its URL and block in ``run``; here
    on the stand-in server, with ``run`` returning at once."""
    built: list[HumanStudio] = []

    def on_the_stand_in(**arguments: Any) -> HumanStudio:
        built.append(HumanStudio(server=FakeServer(), **arguments))
        return built[-1]

    monkeypatch.setattr(app, "HumanStudio", on_the_stand_in)
    monkeypatch.setattr(HumanStudio, "run", lambda self: None)
    launch(data_root=tmp_path / "launched", generator=np.random.default_rng(0), gender="female")
    monkeypatch.setattr(sys, "argv", ["python -m better_human.studio", "--data-root", str(tmp_path / "cli")])
    main()
    assert [studio.session.model_key for studio in built] == ["smpl_female", "smpl_neutral"]
    assert [studio.paths.root for studio in built] == [tmp_path / "launched", tmp_path / "cli"]
    assert capsys.readouterr().out.splitlines() == [f"HumanStudio at {studio.viewer.url}" for studio in built]


def test_the_command_line_requires_a_data_root(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    monkeypatch.setattr(sys, "argv", ["python -m better_human.studio", "--family", "smpl"])
    with pytest.raises(SystemExit) as exit_info:
        main()
    assert exit_info.value.code == 2
    assert "the following arguments are required: --data-root" in capsys.readouterr().err


def _click(studio, origin, target) -> None:
    origin = np.asarray(origin, float)
    event = SimpleNamespace(ray_origin=origin, ray_direction=np.asarray(target, float) - origin)
    _pointer(studio, "click")(event)


def _pointer_events(studio: HumanStudio) -> list[str]:
    """The event types of the scene's pointer callbacks, in registration order."""
    return [event_type for event_type, _ in studio.viewer.server.scene.pointers]


def _pointer(studio: HumanStudio, event_type: str) -> Any:
    """The one scene callback registered for ``event_type`` (``"click"`` or ``"rect-select"``)."""
    (callback,) = [callback for held, callback in studio.viewer.server.scene.pointers if held == event_type]
    return callback


def test_imports_without_running_server() -> None:
    assert callable(HumanStudio) and hasattr(studio_state, "StudioSession")


@pytest.mark.licensed
def test_layers_session_and_sections_built(studio) -> None:
    assert studio.session.model_key == "smpl_neutral"
    assert _pointer_events(studio) == ["click"]
    assert _control(studio, "Build bodies").kind == "GuiButtonHandle"
    assert studio.status.startswith("smpl_neutral loaded")


@pytest.mark.licensed
def test_shape_slider_changes_vertices(studio) -> None:
    before = studio.session.vertices.clone()
    slider = (
        studio.shape_sliders["betas"][0]
        if "betas" in studio.shape_sliders
        else next(iter(studio.shape_sliders.values()))[0]
    )
    slider.value = 2.0
    slider.fire()
    assert not torch.allclose(before, studio.session.vertices)


@pytest.mark.licensed
def test_click_on_joint_selects_and_highlights(studio) -> None:
    joint = studio.session.state.joint_translations.detach().reshape(-1, 3)[5].numpy()
    _click(studio, joint + np.array([0.0, 0.0, 2.0]), joint)
    assert studio.gizmo.joint == 5


@pytest.mark.licensed
def test_pose_save_and_load_roundtrip(studio) -> None:
    q = studio.session.q.clone()
    q[..., -1] += 0.3
    studio.session.set_q(q)
    studio.save_pose("p")
    studio.session.reset_pose()
    studio.load_pose("p")
    assert torch.allclose(studio.session.q, q)


@pytest.mark.licensed
def test_segment_paint_and_build_bodies(studio) -> None:
    studio.segment.load_builtin()
    studio.segment.add_part("blob")
    studio.segment.set_painting(True)
    assert studio.mode == "segment"
    center = studio.session.vertices[0].detach().numpy()
    _click(studio, center + np.array([0.0, 0.0, 2.0]), center)
    rows = studio.segment.build_bodies()
    assert rows[-1][0] == "total" and rows[-1][1] > 10.0
    assert "blob" in dict(rows)


@pytest.mark.licensed
def test_frame_add_rename_delete_export(studio) -> None:
    studio.frames.set_adding(True)
    point = studio.session.vertices[100].detach().numpy()
    _click(studio, point + np.array([0.0, 0.0, 2.0]), point)
    name = studio.frames.active
    assert name in studio.session.frame_set
    studio.frames.rename(name, "toe")
    studio.frames.set_joint("toe", studio.session.body.spec.joint_names[1])
    snippet = studio.frames.export()
    namespace: dict[str, object] = {}
    exec(snippet, {}, namespace)  # noqa: S102
    assert namespace["contact_frames"]["toe"][1] == studio.session.body.spec.joint_names[1]
    studio.frames.save("set")
    studio.frames.delete("toe")
    assert "toe" not in studio.session.frame_set
    studio.frames.load("set")
    assert "toe" in studio.session.frame_set


@pytest.mark.licensed
def test_joint_selection_highlights_dims_and_reverts(studio) -> None:
    opacities: list[float] = []
    studio.layers.set_opacity = opacities.append
    selected = studio.layers.selected
    studio.select_joint(5)
    joint = studio.session.state.joint_translations[5].detach().numpy()
    assert studio.layers.highlight == 5 and selected.radii[0] > studio.layers.radius
    np.testing.assert_allclose(selected.centers[0, 0], joint, atol=1e-6)
    studio.set_opacity(0.8)
    assert opacities[-1] == pytest.approx(0.8 * 0.35)
    studio.select_joint(None)
    assert opacities[-1] == pytest.approx(0.8)
    assert studio.layers.highlight is None and selected.radii[0] == 0.0


@pytest.mark.licensed
def test_selection_row_buttons_and_label(studio) -> None:
    gui = studio.viewer.server.gui
    # Only the Status section's buttons carry these labels.
    for name in ("Reset pose", "Deselect joint"):
        assert [c for c in gui.controls if c.name == name] == [studio.actions[name]]
    joint_name = studio.session.body.spec.joint_names[5]
    studio.select_joint(5)
    assert f"joint: {joint_name}" in studio.selection_text.content
    studio.frames.add_at(studio.session.vertices[100].detach().numpy())  # adds and selects a frame
    assert "frame: —" not in studio.selection_text.content
    studio.gizmo._on_drag(studio.gizmo.handle_for(5), _rotated(studio, 5))
    for action in ("Deselect joint", "Deselect frame", "Reset pose"):
        studio.actions[action].fire()
    assert studio.selection_text.content.endswith("joint: — · frame: —")
    assert studio.frames.active is None
    assert torch.equal(studio.session.q, studio.session.identity.model.q_neutral)


def _rotated(studio, joint: int) -> np.ndarray:
    pose = studio.session.joint_poses[joint].detach().numpy().copy()
    pose[3:] = [np.sin(0.4), 0.0, 0.0, np.cos(0.4)]
    return pose


@pytest.mark.licensed
def test_gizmo_drag_rotates_session_vertices(studio) -> None:
    studio.select_joint(5)
    before = studio.session.vertices.clone()
    pose = studio.session.joint_poses[5].detach().numpy().copy()
    pose[3:] = [np.sin(0.4), 0.0, 0.0, np.cos(0.4)]
    studio.gizmo._on_drag(studio.gizmo.handle_for(5), pose)
    assert (studio.session.vertices - before).abs().max() > 1e-3


@pytest.mark.licensed
def test_frame_click_selects_and_syncs_editor(studio) -> None:
    studio.frames.set_adding(True)
    point = studio.session.vertices[100].detach().numpy()
    _click(studio, point + np.array([0.0, 0.0, 2.0]), point)
    name = studio.frames.active
    studio.frames.set_adding(False)
    studio.frames.active = None
    studio.frames.frame_dropdown.value = None
    _click(studio, point + np.array([0.0, 0.0, 2.0]), point)
    assert studio.frames.active == name
    assert studio.frames.frame_dropdown.value == name


@pytest.mark.licensed
def test_marquee_assigns_vertices_from_screen_rect(studio) -> None:
    studio.segment.add_part("blob")
    blob = studio.session.part_names.index("blob")
    studio.start_marquee()
    camera = SimpleNamespace(position=(0.0, -0.3, 4.0), wxyz=(0.0, 0.0, 1.0, 0.0), fov=1.0, aspect=1.0)
    event = SimpleNamespace(client=SimpleNamespace(camera=camera), screen_min=(0.0, 0.0), screen_max=(1.0, 1.0))
    _pointer(studio, "rect-select")(event)
    assert int((studio.session.part_ids == blob).sum()) == studio.session.vertices.shape[-2]
    assert _pointer_events(studio) == ["click"]


@pytest.mark.licensed
def test_build_bodies_total_matches_session_mass(studio) -> None:
    rows = studio.segment.build_bodies()
    assert rows[-1][1] == pytest.approx(float(studio.session.total_mass), rel=0.01)


@pytest.mark.licensed
def test_density_table_and_saved_pose_dropdown(studio) -> None:
    studio.density_input.value = "Dempster"
    assert studio.chosen_density() == "Dempster"
    studio.reload(studio.chosen_density())
    assert studio.session.body.density.numel() > 1
    studio.save_pose("first")
    assert studio.saved_poses.options == ("first",) and studio.saved_poses.value == "first"


@pytest.mark.licensed
def test_frames_layer_follows_frame_set(tmp_path: Path) -> None:
    frames = tmp_path / "frames" / "smplx_neutral"
    frames.mkdir(parents=True)
    shutil.copy(Path(__file__).parents[1] / "support" / "contacts.json", frames)
    studio = HumanStudio(family="smplx", server=FakeServer(), data_root=tmp_path, generator=np.random.default_rng(0))
    studio.frames.load("contacts")
    count = len(studio.session.frame_set)
    assert count > 0 and studio.layers.frames.poses.shape == (1, count, 7)
    studio.frames.add_at(studio.session.vertices[0].detach().numpy())
    assert studio.layers.frames.poses.shape == (1, count + 1, 7)


@pytest.mark.licensed
def test_model_switch_rebuilds_body_widgets(studio) -> None:
    for family, key in (("smplx", "smplx_neutral"), ("mhr", "mhr_lod1")):
        studio.load_model(family, gender="neutral", lod=1)
        body = studio.session.body
        assert studio.session.model_key == key and body.spec.name == family
        groups = body.spec.identity_groups
        assert set(studio.shape_sliders) == set(groups)
        assert all(len(studio.shape_sliders[g]) == min(w, 10) for g, w in groups.items())
        names = [layer.name for layer in studio.layers.layers.values()]
        assert len(names) == len(set(names)) and "frames" in names
        assert key in studio.status


@pytest.mark.licensed
def test_model_switch_clears_active_frame(studio) -> None:
    studio.frames.set_adding(True)
    point = studio.session.vertices[100].detach().numpy()
    _click(studio, point + np.array([0.0, 0.0, 2.0]), point)
    assert studio.frames.active in studio.session.frame_set
    studio.load_model("smpl", gender="female")
    assert studio.session.model_key == "smpl_female"
    assert studio.frames.active is None


@pytest.mark.licensed
def test_model_switch_on_real_viser_server(tmp_path: Path) -> None:
    """A body swap removes and rebuilds the sidebar sections and the scene's layers on a real viser server."""
    server = viser.ViserServer(host="127.0.0.1", port=8097, verbose=False)
    try:
        studio = HumanStudio(server=server, data_root=tmp_path, generator=np.random.default_rng(0))
        studio.load_model("smpl", gender="neutral")
        assert studio.status.startswith("smpl_neutral loaded")
    finally:
        server.stop()
