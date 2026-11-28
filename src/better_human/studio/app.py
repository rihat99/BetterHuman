"""``HumanStudio``: an interactive body editor built on ``better_robot.Viewer``."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, final

import numpy as np
import torch
from better_robot import Viewer
from better_robot.exceptions import ArgumentError, BetterRobotError, UnknownNameError
from better_robot.viewer.tools import CameraFrustum, nearest_on_ray, ray_hit, vertices_in_rect

import better_human as bh
from better_human.studio import persistence
from better_human.studio.frames_editor import FramesEditor
from better_human.studio.gizmo import GizmoController
from better_human.studio.layers import StudioLayers
from better_human.studio.segment_editor import SegmentEditor
from better_human.studio.state import FrameEntry, StudioSession
from better_human.viz import HUMAN_THEME

if TYPE_CHECKING:
    from viser import SceneClickEvent, SceneRectSelectEvent

_DIM = 0.35  #: mesh opacity factor while a joint is selected
_FAMILIES = ("smpl", "smplx", "mhr")  #: the Model section's choices
_GENDERS = ("neutral", "male", "female")
_MHR_LODS = 7
#: The sidebar sections a body swap builds again, in their order (Status and Model stay).
_BODY_SECTIONS = ("Display", "Body", "Pose", "Segment", "Frames")


def _family(name: str) -> str:
    """``name`` in lower case; raises :class:`~better_robot.exceptions.UnknownNameError` for a family the studio does
    not edit."""
    family = name.lower()
    if family not in _FAMILIES:
        raise UnknownNameError("studio family", name, valid=list(_FAMILIES))
    return family


@final
class HumanStudio:
    """Pose, shape, segment and frame-edit one body family in the browser.

    Click modes: ``"pose"`` (pick a joint, drag its gizmo), ``"segment"`` (sphere-brush painting) and
    ``"frames"`` (add a frame on the nearest vertex). ``data_root`` is the directory the studio reads and writes its
    segmentations, frame sets and poses in; the weights come from ``better_human.io.models_dir()``. ``family`` is
    ``"smpl"``, ``"smplx"`` or ``"mhr"``; another raises :class:`~better_robot.exceptions.UnknownNameError`.
    ``generator`` draws the random shapes. ``density`` is the body's at start and after a family switch: a table name
    (the library's default, ``"Dempster"``) or kg/m^3. ``host`` and ``port`` default to the viewer's
    (``127.0.0.1:8080``); ``server`` injects a viser-compatible server instead (tests).
    """

    def __init__(
        self,
        *,
        data_root: str | Path,
        family: str = "smpl",
        gender: str = "neutral",
        lod: int = 1,
        port: int | None = None,
        host: str | None = None,
        generator: np.random.Generator,
        density: str | float = "Dempster",
        server: Any = None,
    ) -> None:
        self.family, self.gender, self.lod = _family(family), gender, lod
        self.generator = generator
        self.density, self._start_density, self.opacity = density, density, 1.0
        self._uniform_density = 1000.0 if isinstance(density, str) else float(density)  # the dropdown's uniform entry
        self.paths = persistence.StudioPaths(Path(data_root).expanduser())
        self.mode = "pose"
        self.status = ""
        started = time.perf_counter()
        body = self._load_body()
        self.session = StudioSession(body, self._key(), self._default_frames(body))
        self.viewer = Viewer(host=host, port=port, theme=HUMAN_THEME, ground=False, server=server)
        self.sidebar = self.viewer.sidebar
        self.layers = StudioLayers(self.viewer, self.session, theme=self.viewer.theme)
        self.gizmo = GizmoController(self.viewer, self.session, on_status=self.set_status)
        self.segment = SegmentEditor(self)
        self.frames = FramesEditor(self)
        self.session.subscribe(self._on_change)
        self._build_sidebar()
        self.viewer.on_pointer("click", self.locked(self.on_click))
        self.set_status(f"{self._key()} loaded in {time.perf_counter() - started:.1f}s")

    def __repr__(self) -> str:
        return f"HumanStudio({self._key()}, mode={self.mode})"

    def _key(self) -> str:
        if self.family == "mhr":
            return persistence.model_key("mhr", lod=self.lod)
        return persistence.model_key(self.family, gender=self.gender)

    def _load_body(self) -> bh.HumanModel:
        if self.family == "mhr":
            spec = bh.families.mhr(lod=self.lod)
        elif self.family == "smplx":
            spec = bh.families.smplx(gender=self.gender)
        else:
            spec = bh.families.smpl(gender=self.gender)
        return bh.HumanModel.from_spec(spec, density=self.density)

    @staticmethod
    def _default_frames(body: bh.HumanModel) -> dict[str, FrameEntry]:
        frames = body.contact_frames
        if frames is None:
            return {}
        return {name: FrameEntry(int(vertex)) for name, vertex in zip(frames.names, frames.vertex_ids, strict=True)}

    def _build_sidebar(self) -> None:
        # Status and Model stay for the whole run; the sections after them depend on the body (_build_body_sections).
        sidebar, gui = self.sidebar, self.sidebar.gui
        with sidebar.section("Status"):
            self.status_text = gui.add_markdown("")
            self.selection_text = gui.add_markdown("")
        sidebar.button("Status", "Reset pose", on_click=self.locked(lambda _=None: self.session.reset_pose()))
        sidebar.button("Status", "Deselect joint", on_click=self.locked(lambda _=None: self.select_joint(None)))
        sidebar.button("Status", "Deselect frame", on_click=self.locked(lambda _=None: self.deselect_frame()))
        self.family_input = sidebar.dropdown(
            "Model", "Family", options=_FAMILIES, value=self.family, on_change=self.locked(self._sync_model_inputs)
        )
        with sidebar.section("Model"):
            self.gender_input = gui.add_dropdown(
                "Gender", options=_GENDERS, initial_value=self.gender, disabled=self.family == "mhr"
            )
            self.lod_input = gui.add_slider(
                "LOD", min=0, max=_MHR_LODS - 1, step=1, initial_value=self.lod, disabled=self.family != "mhr"
            )
        sidebar.button(
            "Model",
            "Load",
            on_click=self.locked(
                lambda _=None: self.load_model(
                    self.family_input.value, gender=self.gender_input.value, lod=int(self.lod_input.value)
                )
            ),
        )
        self._build_body_sections()
        self._show_selection()

    def _build_body_sections(self) -> None:
        sidebar, gui, layers = self.sidebar, self.sidebar.gui, self.layers
        sidebar.dropdown(
            "Display", "Colour", options=("skin", "segmentation", "parts"), value=layers.mode,
            on_change=self.locked(layers.set_mode),
        )  # fmt: skip
        sidebar.slider(
            "Display", "Opacity", low=0.1, high=1.0, step=0.05, value=self.opacity,
            on_change=self.locked(lambda value: self.set_opacity(float(value))),
        )  # fmt: skip
        sidebar.slider(
            "Display", "Joint radius", low=0.005, high=0.05, step=0.005, value=layers.radius,
            on_change=self.locked(lambda value: layers.set_radius(float(value))),
        )  # fmt: skip
        sidebar.slider(
            "Display", "Explode", low=0.0, high=1.0, step=0.05, value=layers.explode,
            on_change=self.locked(lambda value: layers.set_explode(float(value))),
        )  # fmt: skip
        self.shape_sliders: dict[str, list[Any]] = {}
        for group, width in self.session.body.spec.identity_groups.items():
            self.shape_sliders[group] = [
                sidebar.slider(
                    "Body",
                    f"{group}[{k}]",
                    low=-5.0,
                    high=5.0,
                    step=0.05,
                    value=0.0,
                    on_change=self.locked(lambda _, group=group: self.apply_shape(group)),
                )  # fmt: skip
                for k in range(min(width, 10))
            ]
            sidebar.button("Body", f"Reset {group}", on_click=self.locked(lambda _=None, g=group: self.reset_shape(g)))
            sidebar.button(
                "Body", f"Random {group}", on_click=self.locked(lambda _=None, g=group: self.random_shape(g))
            )
        tables = tuple(self.session.body.spec.parts.density_tables)
        with sidebar.section("Body"):
            if tables:
                self._uniform = f"uniform {self._uniform_density:g} kg/m^3"
                start = self.density if self.density in tables else self._uniform
                self.density_input = gui.add_dropdown(
                    "Density table", options=(self._uniform, *tables), initial_value=start
                )
            else:
                self.density_input = gui.add_number("Density (kg/m^3)", initial_value=self._uniform_density)
        sidebar.button("Body", "Apply density", on_click=self.locked(lambda _=None: self.reload(self.chosen_density())))
        with sidebar.section("Body"):
            self.mass_text = gui.add_markdown("")
        with sidebar.section("Pose"):
            self.pose_name = gui.add_text("Pose name", initial_value="pose")
            self.saved_poses = gui.add_dropdown("Saved poses", options=self._pose_options())
        sidebar.button("Pose", "Save pose", on_click=self.locked(lambda _=None: self.save_pose(self.pose_name.value)))
        sidebar.button("Pose", "Load pose", on_click=self.locked(lambda _=None: self.load_pose(self.saved_poses.value)))
        self.segment.build(sidebar)
        self.frames.build(sidebar)
        self._update_mass()

    def locked(self, callback: Callable[[Any], None]) -> Callable[..., None]:
        """Wrap a sidebar or scene callback so it runs under the viewer's lock and ``session.lock`` and shows a refused
        edit on the status line; the wrapper takes the control's value, the scene event, or nothing (a button).

        A ``BetterRobotError`` (a file of another format, data authored on another body, parts that do not close)
        is the user's to fix, so the app shows its message instead of stopping the callback thread; any other
        exception propagates.
        """

        def run(argument: Any = None) -> None:
            with self.viewer.lock, self.session.lock:
                try:
                    callback(argument)
                except BetterRobotError as error:
                    self.set_status(f"{type(error).__name__}: {error}")

        return run

    def _sync_model_inputs(self, family: str) -> None:
        mhr = family == "mhr"
        self.gender_input.disabled, self.lod_input.disabled = mhr, not mhr

    def load_model(self, family: str, *, gender: str = "neutral", lod: int = 1) -> None:
        """Switch to another body (``family``, SMPL-family ``gender`` or MHR ``lod``) and rebuild the sections that
        depend on it."""
        started = time.perf_counter()
        family = _family(family)
        self.set_mode("pose")
        self.family, self.gender, self.lod = family, gender, lod
        self.density = self._start_density
        body = self._load_body()
        self.session.load_body(body, self._key(), self._default_frames(body))
        self.frames.active = None  # frame names belong to the previous model
        for section in _BODY_SECTIONS:
            self.sidebar.remove(section)
        self._build_body_sections()
        self.set_status(f"{self._key()} loaded in {time.perf_counter() - started:.1f}s")

    def set_status(self, message: str) -> None:
        """Show ``message`` on the status line."""
        self.status = message
        self.status_text.content = message
        # The frames editor reports every frame selection through this line, so the selection label follows it here.
        self._show_selection()

    def _show_selection(self) -> None:
        model = f"{self.family} {self.gender if self.family != 'mhr' else f'lod {self.lod}'}"
        joint = self.gizmo.joint
        joint_name = "—" if joint is None else self.session.body.spec.joint_names[joint]
        frame = self.frames.active if self.frames.active in self.session.frame_set else "—"
        self.selection_text.content = f"{model} · joint: {joint_name} · frame: {frame}"

    def deselect_frame(self) -> None:
        """Drop the frame the Frames section acts on and its vertex highlight (as deleting a frame does)."""
        self.frames.active = None
        self.layers.set_edit_vertices([])
        self._show_selection()

    def set_mode(self, mode: str) -> None:
        """Switch the click mode (``"pose"``, ``"segment"``, ``"frames"``); entering an edit mode drops the gizmo."""
        self.mode = mode
        if mode != "pose":
            self.select_joint(None)
        if mode != "segment" and self.segment.painting:
            self.segment.painting = False
            self.sidebar.set_value("Segment", "Paint (click mesh)", value=False)
        if mode != "frames" and self.frames.adding:
            self.frames.adding = False
            self.sidebar.set_value("Frames", "Add frames (click mesh)", value=False)

    def select_joint(self, joint: int | None) -> None:
        """Attach the gizmo to ``joint`` (native index) and highlight its sphere; ``None`` clears both."""
        self.gizmo.select(joint)
        self.layers.set_highlight(joint)
        self.layers.set_opacity(self.opacity if joint is None else self.opacity * _DIM)
        self._show_selection()

    def set_opacity(self, opacity: float) -> None:
        """Set the mesh opacity the user chose; it stays dimmed while a joint is selected."""
        self.opacity = opacity
        self.layers.set_opacity(opacity if self.gizmo.joint is None else opacity * _DIM)

    def pick_joint(self, origin: np.ndarray, direction: np.ndarray) -> int | None:
        """Nearest joint (along the ray) whose sphere of radius ``layers.radius`` the ray crosses."""
        return nearest_on_ray(
            self.session.state.joint_translations.detach().reshape(-1, 3).cpu().numpy(),
            origin,
            direction=direction,
            radius=self.layers.radius,
        )

    def pick_frame(self, origin: np.ndarray, direction: np.ndarray) -> str | None:
        """Nearest frame (along the ray) whose vertex lies within ``layers.radius`` of the ray."""
        names = list(self.session.frame_set)
        if not names:
            return None
        points = self.session.vertices.detach().cpu().numpy()[[self.session.frame_set[name].vertex for name in names]]
        index = nearest_on_ray(points, origin, direction=direction, radius=self.layers.radius)
        return None if index is None else names[index]

    def on_click(self, event: SceneClickEvent) -> None:
        """Dispatch a scene click (``event.ray_origin``, ``event.ray_direction``) to the active mode."""
        if event.ray_origin is None or event.ray_direction is None:
            return
        origin, direction = np.asarray(event.ray_origin, float), np.asarray(event.ray_direction, float)
        if self.mode == "pose":
            joint = self.pick_joint(origin, direction)
            if joint is not None:
                self.select_joint(joint)
                return
            frame = self.pick_frame(origin, direction)
            if frame is not None:
                self.frames.select_frame(frame)
                return
        hit = ray_hit(
            origin,
            direction,
            vertices=self.session.vertices.detach().cpu().numpy(),
            faces=self.session.body.spec.faces.cpu().numpy(),
        )
        if hit is None:
            return
        point, face = hit
        if self.mode == "segment":
            self.segment.paint(point)
        elif self.mode == "frames":
            self.frames.add_at(point)
        else:
            vertex_joints = self.session.vertex_native_joints
            nearest = int(self.session.body.spec.faces[face, 0])
            self.select_joint(int(vertex_joints[nearest]))

    def start_marquee(self) -> None:
        """Arm one rectangle selection that assigns the enclosed vertices to the active segment part."""

        def _on_rect(event: SceneRectSelectEvent) -> None:
            self.viewer.clear_pointer()
            camera = CameraFrustum.from_camera(event.client.camera)
            ids = vertices_in_rect(
                self.session.vertices.detach().cpu().numpy(),
                camera,
                rect_min=np.asarray(event.screen_min, float),
                rect_max=np.asarray(event.screen_max, float),
            )
            self.segment.assign(ids)
            self.viewer.on_pointer("click", self.locked(self.on_click))

        self.viewer.clear_pointer()
        self.viewer.on_pointer("rect-select", self.locked(_on_rect))
        self.set_status("drag a rectangle to select vertices")

    def apply_shape(self, group: str) -> None:
        """Push the ``group`` sliders into the session (re-bakes the identity)."""
        width = self.session.body.spec.identity_groups[group]
        values = [float(slider.value) for slider in self.shape_sliders[group]] + [0.0] * width
        self.session.set_shape(**{group: self.session.shape_tensor(group, values[:width])})

    def _set_sliders(self, group: str, values: list[float]) -> None:
        """Show ``values`` on the ``group`` sliders without running their callbacks, then apply them once."""
        for k, value in enumerate(values):
            self.sidebar.set_value("Body", f"{group}[{k}]", value=value)
        self.apply_shape(group)

    def reset_shape(self, group: str) -> None:
        """Zero the ``group`` sliders and coefficients."""
        self._set_sliders(group, [0.0] * len(self.shape_sliders[group]))

    def random_shape(self, group: str) -> None:
        """Draw the ``group`` sliders from ``N(0, 1)``."""
        self._set_sliders(group, [round(float(self.generator.normal()), 2) for _ in self.shape_sliders[group]])

    def chosen_density(self) -> float | str:
        """The density control's value: kg/m^3, or a table name for SMPL / SMPL-H."""
        value = self.density_input.value
        if not self.session.body.spec.parts.density_tables:
            return float(value)
        return self._uniform_density if value == self._uniform else value

    def _pose_options(self) -> tuple[str, ...]:
        return tuple(persistence.list_saved(self.paths.pose_dir(self.session.model_key))) or ("(none)",)

    def save_pose(self, name: str) -> Path:
        """Save ``q`` to ``<data>/poses/<model_key>/<name>.json``; ``name`` is one file name."""
        path = persistence.saved_path(self.paths.pose_dir(self.session.model_key), name)
        model = self.session.identity.model
        persistence.write_pose(path, persistence.Pose(joint_names=model.joint_names, q=tuple(self.session.q.tolist())))
        self.saved_poses.options = self._pose_options()
        self.saved_poses.value = name
        self.set_status(f"saved pose {path}")
        return path

    def load_pose(self, name: str) -> None:
        """Load a saved pose by name."""
        path = persistence.saved_path(self.paths.pose_dir(self.session.model_key), name)
        if not path.is_file():
            self.set_status(f"no saved pose {name!r}")
            return
        pose = persistence.read_pose(path)
        model = self.session.identity.model
        if pose.joint_names != model.joint_names:
            raise ArgumentError(
                f"{path}: the pose follows other joints than the {self.session.model_key} model's; load a pose saved on it"
            )
        if len(pose.q) != model.nq:
            raise ArgumentError(
                f"{path}: the pose holds {len(pose.q)} values of q, the {self.session.model_key} model takes "
                f"nq={model.nq}; re-save it in the studio"
            )
        self.session.set_q(torch.tensor(pose.q))

    def reload(self, density: float | str) -> None:
        """Reload the body at a new density, keeping the pose and frames."""
        self.density = density
        q, frames = self.session.q.clone(), dict(self.session.frame_set)
        self.session.load_body(self._load_body(), self._key(), frames)
        self.session.set_q(q)

    def _on_change(self, kind: str) -> None:
        if kind in ("body", "shape"):
            self._update_mass()
        if kind in ("body", "segmentation"):
            self.segment.refresh()
        if kind in ("body", "frames"):
            self.frames.refresh()
            self._show_selection()

    def _update_mass(self) -> None:
        mass = self.session.total_mass
        self.mass_text.content = "mass: n/a" if mass is None else f"mass: {float(mass):.2f} kg"

    def run(self) -> None:
        """Print the address to open and serve the page until interrupted."""
        print(f"HumanStudio at {self.viewer.url}")  # noqa: T201 - the one line a user needs: where to open it
        self.viewer.serve()
