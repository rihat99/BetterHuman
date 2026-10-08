"""``HumanStudio``: an interactive body editor built on ``better_robot.viewer.Viewer``."""

from __future__ import annotations

import os
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from better_robot.viewer import LIGHT, Viewer
from better_robot.viewer.tools import CameraFrustum, ray_hit, vertices_in_rect

import better_human as bh
from better_human.families.assets import load_density_table
from better_human.viz.studio import persistence
from better_human.viz.studio.frames_editor import FramesEditor
from better_human.viz.studio.gizmo import GizmoController
from better_human.viz.studio.layers import StudioLayers
from better_human.viz.studio.segment_editor import SegmentEditor
from better_human.viz.studio.state import FrameEntry, StudioSession

_TABLE_FAMILIES = ("smpl", "smplh")
_DENSITY_TABLES = ("Naive", "Chandler", "Dempster", "Clauser", "Custom")  #: keys of body_densities.json
_DIM = 0.35  #: mesh opacity factor while a joint is selected
_FAMILIES = ("smpl", "smplx", "mhr")  #: Model-tab choices
_GENDERS = ("neutral", "male", "female")
_MHR_LODS = 7

class HumanStudio:
    """Pose, shape, segment and frame-edit one body family in the browser.

    Click modes: ``"pose"`` (pick a joint, drag its gizmo), ``"segment"`` (sphere-brush painting) and
    ``"frames"`` (add a frame on the nearest vertex). ``server`` injects a viser-compatible server (tests).
    """

    def __init__(
        self,
        *,
        family: str = "smpl",
        gender: str = "neutral",
        lod: int = 1,
        port: int = 8080,
        host: str = "127.0.0.1",
        models_dir: str | Path | None = None,
        data_dir: str | Path | None = None,
        density: float | None = None,
        server: Any = None,
    ) -> None:
        if models_dir is not None:
            os.environ["BETTERHUMAN_MODELS_DIR"] = str(Path(models_dir).expanduser())
        self.family, self.gender, self.lod = family.lower(), gender, lod
        self.density = 1000.0 if density is None else float(density)
        self._uniform_density, self.opacity = self.density, 1.0
        self.paths = persistence.StudioPaths(persistence.DEFAULT_ROOT if data_dir is None else Path(data_dir))
        self.mode = "pose"
        self.status = ""
        started = time.perf_counter()
        body = self._load_body()
        self.session = StudioSession(body, self._key(), self._default_frames(body))
        self.viewer = Viewer(host=host, port=port, theme=replace(LIGHT, up="+y"), server=server)
        self.layers = StudioLayers(self.viewer, self.session, theme=self.viewer.theme)
        self.gizmo = GizmoController(self.viewer, self.session, on_status=self.set_status)
        self.segment = SegmentEditor(self)
        self.frames = FramesEditor(self)
        self.session.subscribe(self._on_change)
        self._build_sidebar()
        self.viewer.server.scene.on_pointer_event(event_type="click")(self._locked(self.on_click))
        self.viewer.show()
        self.set_status(f"{self._key()} loaded in {time.perf_counter() - started:.1f}s")

    def _key(self) -> str:
        if self.family == "mhr":
            return persistence.model_key("mhr", lod=self.lod)
        return persistence.model_key(self.family, gender=self.gender)

    def _load_body(self) -> bh.Body:
        options = {"lod": self.lod} if self.family == "mhr" else {"gender": self.gender}
        density = self.density
        if isinstance(density, str):
            body = self.session.body
            names = [body.joint_names[j] for j in body.spec.parts.joints]
            density = load_density_table("body_densities.json", density, names, body.spec.vertices)
        return bh.load(self.family, mass=True, density=density, **options)

    @staticmethod
    def _default_frames(body: bh.Body) -> dict[str, FrameEntry]:
        frames = body.contact_frames
        if frames is None:
            return {}
        return {name: FrameEntry(int(vertex)) for name, vertex in zip(frames.names, frames.vertex_ids, strict=True)}

    def _build_sidebar(self) -> None:
        gui = self.viewer.server.gui
        self.status_text = gui.add_markdown("")
        # Built once above the tab group, so a model switch (which rebuilds only the tabs) keeps this row on top.
        self.selection_text = gui.add_markdown("")
        # Plain buttons: viser's button group never wraps and clips at the sidebar edge.
        self.actions: dict[str, Any] = {}
        for action in ("Reset pose", "Deselect joint", "Deselect frame"):
            self.actions[action] = gui.add_button(action)
            self.actions[action].on_click(self._locked(lambda _event, action=action: self.run_action(action)))
        self._show_selection()
        self._build_tabs()

    def _build_tabs(self) -> None:
        # Every tab depends on the body, so a model switch removes the tab group and builds it again.
        gui = self.viewer.server.gui
        self.tabs = tabs = gui.add_tab_group()
        with tabs.add_tab("Model"):
            families = tuple(dict.fromkeys((*_FAMILIES, self.family)))
            self.family_input = gui.add_dropdown("Family", options=families, initial_value=self.family)
            self.gender_input = gui.add_dropdown(
                "Gender", options=_GENDERS, initial_value=self.gender, disabled=self.family == "mhr"
            )
            self.lod_input = gui.add_slider(
                "LOD", min=0, max=_MHR_LODS - 1, step=1, initial_value=self.lod, disabled=self.family != "mhr"
            )
            self.family_input.on_update(self._locked(lambda _event: self._sync_model_inputs()))
            gui.add_button("Load").on_click(
                self._locked(
                    lambda _event: self.load_model(
                        self.family_input.value, gender=self.gender_input.value, lod=int(self.lod_input.value)
                    )
                )
            )
        with tabs.add_tab("Body"):
            gui.add_markdown(f"**{self.family}** {self.gender if self.family != 'mhr' else f'lod {self.lod}'}")
            self.shape_sliders: dict[str, list[Any]] = {}
            for group, width in self.session.body.identity_groups.items():
                sliders = [
                    gui.add_slider(f"{group}[{k}]", min=-5.0, max=5.0, step=0.05, initial_value=0.0)
                    for k in range(min(width, 10))
                ]
                for slider in sliders:
                    slider.on_update(self._locked(lambda _event, group=group: self.apply_shape(group)))
                self.shape_sliders[group] = sliders
                gui.add_button(f"Reset {group}").on_click(self._locked(lambda _event, group=group: self.reset_shape(group)))
                gui.add_button(f"Random {group}").on_click(self._locked(lambda _event, group=group: self.random_shape(group)))
            if self.family in _TABLE_FAMILIES:
                self._uniform = f"uniform {self.density:g} kg/m^3"
                self.density_input = gui.add_dropdown(
                    "Density table", options=(self._uniform, *_DENSITY_TABLES), initial_value=self._uniform
                )
            else:
                self.density_input = gui.add_number("Density (kg/m^3)", initial_value=self.density)
            gui.add_button("Apply density").on_click(self._locked(lambda _event: self.reload(self.chosen_density())))
            self.pose_name = gui.add_text("Pose name", initial_value="pose")
            gui.add_button("Save pose").on_click(self._locked(lambda _event: self.save_pose(self.pose_name.value)))
            self.saved_poses = gui.add_dropdown("Saved poses", options=self._pose_options())
            gui.add_button("Load pose").on_click(self._locked(lambda _event: self.load_pose(self.saved_poses.value)))
            self.mass_text = gui.add_markdown("")
        # Visibility lives only in the viewer's Layers folder (one checkbox per layer).
        with tabs.add_tab("Display"):
            opacity = gui.add_slider("Opacity", min=0.1, max=1.0, step=0.05, initial_value=1.0)
            opacity.on_update(self._locked(lambda _event: self.set_opacity(float(opacity.value))))
            radius = gui.add_slider("Joint radius", min=0.005, max=0.05, step=0.005, initial_value=self.layers.radius)
            radius.on_update(self._locked(lambda _event: self.layers.set_radius(float(radius.value))))
            explode = gui.add_slider("Explode", min=0.0, max=1.0, step=0.05, initial_value=0.0)
            explode.on_update(self._locked(lambda _event: self.layers.set_explode(float(explode.value))))
        with tabs.add_tab("Segment"):
            self.segment.build(gui)
        with tabs.add_tab("Frames"):
            self.frames.build(gui)
        self._update_mass()

    def _locked(self, callback: Any) -> Any:
        """Wrap a GUI or scene callback so it runs under ``session.lock``."""

        def run(event: Any) -> None:
            with self.session.lock:
                callback(event)

        return run

    def _sync_model_inputs(self) -> None:
        mhr = self.family_input.value == "mhr"
        self.gender_input.disabled, self.lod_input.disabled = mhr, not mhr

    def load_model(self, family: str, *, gender: str = "neutral", lod: int = 1) -> None:
        """Switch to another body (``family``, SMPL-family ``gender`` or MHR ``lod``) and rebuild the tabs."""
        started = time.perf_counter()
        self.set_mode("pose")
        self.family, self.gender, self.lod = family.lower(), gender, lod
        self.density = self._uniform_density  # density tables are per family; start uniform
        body = self._load_body()
        self.session.load_body(body, self._key(), self._default_frames(body))
        # viser's GuiTabGroupHandle.remove() marks the group removed before removing its tabs, and each tab
        # then writes to the removed group and raises; removing the tabs first avoids that.
        for tab in tuple(getattr(self.tabs, "_tab_handles", ())):
            tab.remove()
        self.tabs.remove()
        self._build_tabs()
        self.set_status(f"{self._key()} loaded in {time.perf_counter() - started:.1f}s")

    def set_status(self, message: str) -> None:
        """Show ``message`` on the status line."""
        self.status = message
        if hasattr(self, "status_text"):
            self.status_text.content = message
        # The frames editor reports every frame selection through this line, so the selection label follows it here.
        self._show_selection()

    def _show_selection(self) -> None:
        if not hasattr(self, "selection_text"):
            return
        model = f"{self.family} {self.gender if self.family != 'mhr' else f'lod {self.lod}'}"
        joint = self.gizmo.joint
        joint_name = "—" if joint is None else self.session.body.joint_names[joint]
        frame = self.frames.active if self.frames.active in self.session.frame_set else "—"
        self.selection_text.content = f"{model} · joint: {joint_name} · frame: {frame}"

    def run_action(self, action: str) -> None:
        """Run a selection-row button: ``"Reset pose"``, ``"Deselect joint"`` or ``"Deselect frame"``."""
        if action == "Reset pose":
            self.session.reset_pose()
        elif action == "Deselect joint":
            self.select_joint(None)
        elif action == "Deselect frame":
            self.deselect_frame()

    def deselect_frame(self) -> None:
        """Drop the frame the Frames tab acts on and its vertex highlight (as deleting a frame does)."""
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
            self.segment.paint_toggle.value = False
        if mode != "frames" and self.frames.adding:
            self.frames.adding = False
            self.frames.add_toggle.value = False

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
        return _nearest_on_ray(
            self.session.state.joints.detach().reshape(-1, 3).numpy(), origin, direction, self.layers.radius
        )

    def pick_frame(self, origin: np.ndarray, direction: np.ndarray) -> str | None:
        """Nearest frame (along the ray) whose vertex lies within ``layers.radius`` of the ray."""
        names = list(self.session.frame_set)
        if not names:
            return None
        points = self.session.vertices.detach().numpy()[[self.session.frame_set[name].vertex for name in names]]
        index = _nearest_on_ray(points, origin, direction, self.layers.radius)
        return None if index is None else names[index]

    def on_click(self, event: Any) -> None:
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
        hit = ray_hit(origin, direction, self.session.vertices.detach().numpy(), self.session.body.faces.numpy())
        if hit is None:
            return
        point, face = hit
        if self.mode == "segment":
            self.segment.paint(point)
        elif self.mode == "frames":
            self.frames.add_at(point)
        else:
            vertex_joints = self.session.vertex_native_joints
            nearest = int(self.session.body.faces[face, 0])
            self.select_joint(int(vertex_joints[nearest]))

    def start_marquee(self) -> None:
        """Arm one rectangle selection that assigns the enclosed vertices to the active segment part."""

        def _on_rect(event: Any) -> None:
            self.viewer.server.scene.remove_pointer_callback()
            camera = CameraFrustum.from_camera(event.client.camera)
            (x0, y0), (x1, y1) = event.screen_pos
            ids = vertices_in_rect(
                self.session.vertices.detach().numpy(), camera, np.array([x0, y0]), np.array([x1, y1])
            )
            self.segment.assign(ids)
            self.viewer.server.scene.on_pointer_event(event_type="click")(self._locked(self.on_click))

        self.viewer.server.scene.remove_pointer_callback()
        self.viewer.server.scene.on_pointer_event(event_type="rect-select")(self._locked(_on_rect))
        self.set_status("drag a rectangle to select vertices")

    def apply_shape(self, group: str) -> None:
        """Push the ``group`` sliders into the session (re-bakes the identity)."""
        width = self.session.body.identity_groups[group]
        values = [float(slider.value) for slider in self.shape_sliders[group]] + [0.0] * width
        self.session.set_shape(**{group: self.session.shape_tensor(group, values[:width])})

    def reset_shape(self, group: str) -> None:
        """Zero the ``group`` sliders and coefficients."""
        for slider in self.shape_sliders[group]:
            slider.value = 0.0
        self.apply_shape(group)

    def random_shape(self, group: str) -> None:
        """Draw the ``group`` sliders from ``N(0, 1)``."""
        for slider in self.shape_sliders[group]:
            slider.value = round(float(np.random.normal()), 2)
        self.apply_shape(group)

    def chosen_density(self) -> float | str:
        """The density control's value: kg/m^3, or a table name for SMPL / SMPL-H."""
        value = self.density_input.value
        if self.family not in _TABLE_FAMILIES:
            return float(value)
        return self._uniform_density if value == self._uniform else value

    def _pose_options(self) -> tuple[str, ...]:
        return tuple(persistence.list_saved(self.paths.pose_dir(self.session.model_key))) or ("(none)",)

    def save_pose(self, name: str) -> Path:
        """Save ``q`` to ``<data>/poses/<model_key>/<name>.json``."""
        path = self.paths.pose_dir(self.session.model_key) / f"{name}.json"
        persistence.save_pose(path, self.session.q, self.session.model_key)
        self.saved_poses.options = self._pose_options()
        self.saved_poses.value = name
        self.set_status(f"saved pose {path}")
        return path

    def load_pose(self, name: str) -> None:
        """Load a saved pose by name."""
        path = self.paths.pose_dir(self.session.model_key) / f"{name}.json"
        if not path.is_file():
            self.set_status(f"no saved pose {name!r}")
            return
        self.session.set_q(persistence.load_pose(path, self.session.model_key, self.session.q.shape[-1]))

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
        if hasattr(self, "mass_text"):
            self.mass_text.content = "mass: n/a" if mass is None else f"mass: {float(mass):.2f} kg"

    def run(self) -> None:
        """Block until interrupted."""
        while True:
            time.sleep(1.0)


def _nearest_on_ray(points: np.ndarray, origin: np.ndarray, direction: np.ndarray, radius: float) -> int | None:
    """Index of the point ``(N, 3)`` within ``radius`` of the ray nearest its origin, or ``None``."""
    direction = direction / np.linalg.norm(direction)
    along = (points - origin) @ direction
    distance = np.linalg.norm(points - origin - along[:, None] * direction, axis=-1)
    hits = np.flatnonzero((distance <= radius) & (along > 0))
    return None if hits.size == 0 else int(hits[np.argmin(along[hits])])


def launch(**kwargs: Any) -> HumanStudio:
    """Build a :class:`HumanStudio` with ``kwargs``, print its URL and block."""
    studio = HumanStudio(**kwargs)
    print(f"HumanStudio at http://{kwargs.get('host', '127.0.0.1')}:{kwargs.get('port', 8080)}")
    studio.run()
    return studio
