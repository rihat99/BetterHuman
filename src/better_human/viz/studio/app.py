"""HumanStudio: the viser front end that wires the scene, sidebar, and session.

``HumanStudio`` owns the :class:`~viser.ViserServer`, the shared
:class:`~better_human.viz.studio.state.StudioSession`, the scene layers, and the
sidebar tabs. It loads bodies, drives single-gizmo joint posing, exposes shape
and density controls, and delegates segmentation and frame authoring to the
editors. Slow rebuilds are wrapped so load failures land in the status line
instead of killing the server.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import torch
import viser

import better_human as bh
from better_human.bodies.assets import _ASSET_DIR
from better_human.core.mass import build_part_topology
from better_human.bodies.smpl_family.config_smpl import SMPL_CONFIG
from better_human.bodies.smpl_family.config_smplx import SMPLX_CONFIG
from better_human.core import ParametricBody

from . import densities, persistence
from .frames_editor import FramesEditor
from .gizmo import GizmoController
from .layers import FramesLayer, MeshLayer, PartsLayer, SkeletonLayer
from .layers.base import EDIT_POINT_COLOR, to_numpy
from .layers.mesh import fast_skinning_available
from .segment_editor import SegmentEditor
from .state import StudioSession, compact_segmentation

_GENDERS = ("neutral", "male", "female")
_LODS = tuple(str(lod) for lod in range(7))
_MODELS = ("smpl", "smplx", "mhr")


class HumanStudio:
    """Interactive viser studio for inspecting and authoring parametric bodies."""

    def __init__(
        self,
        *,
        model: str = "smpl",
        gender: str = "neutral",
        lod: int = 1,
        port: int = 8080,
        data_dir: str | Path = "studio_data",
        models_dir: str | Path | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        self._device = str(device)
        self._models_dir = self._resolve_models_dir(models_dir)
        self._paths = persistence.StudioPaths(Path(data_dir))
        self._server = viser.ViserServer(port=port)

        self._selected: int | None = None
        self._suppress_shape = False
        self._coloring_busy = False
        self._coloring_before_fast = False
        self._density_note = ""
        self._edit_pointcloud: object | None = None
        self._shape_sliders: dict[str, list] = {}
        self._visibility_checks: dict[str, object] = {}

        self._status = self._server.gui.add_markdown("HumanStudio starting...")
        body = self._load_body(model, gender, lod, None)
        self._session = StudioSession(body, self._current_key(model, gender, lod), self._default_frames(model))
        self._session.subscribe(self._on_change)
        self._gizmo_ctrl = GizmoController(self._server, self._session, self._set_status)

        self._skeleton = SkeletonLayer(self.select_joint, base_radius=0.015 if model == "mhr" else 0.02)
        self._layers = {
            "mesh": MeshLayer(),
            "skeleton": self._skeleton,
            "frames": FramesLayer(self._select_frame_from_click),
            "parts": PartsLayer(),
        }
        for layer in self._layers.values():
            layer.build(self._server, self._session)

        self._build_sidebar()
        self._refresh_status(load_seconds=0.0)

    # -- weight resolution ---------------------------------------------------

    @staticmethod
    def _resolve_models_dir(models_dir: str | Path | None) -> Path:
        if models_dir is not None:
            return Path(models_dir)
        env = os.environ.get("BETTERHUMAN_MODELS_DIR")
        if env:
            return Path(env)
        repo_models = Path(__file__).resolve().parents[4] / "models"
        if repo_models.is_dir():
            return repo_models
        raise FileNotFoundError(
            "No models directory found: pass models_dir, set $BETTERHUMAN_MODELS_DIR, "
            "or run inside a repo checkout with a models/ directory."
        )

    def _model_path(self, model: str, gender: str, lod: int) -> Path:
        if model == "smpl":
            return self._models_dir / "smpl" / f"SMPL_{gender.upper()}.npz"
        if model == "smplx":
            return self._models_dir / "smplx" / f"SMPLX_{gender.upper()}.npz"
        return self._models_dir / "MHR" / "converted" / f"mhr_lod{lod}.npz"

    def _load_body(self, model: str, gender: str, lod: int, density: object | None) -> ParametricBody:
        cfg: dict[str, object] = {
            "model_path": str(self._model_path(model, gender, lod)),
            "compute_mass": True,
            "device": self._device,
        }
        if model in ("smpl", "smplx"):
            cfg["gender"] = gender
        else:
            cfg["lod"] = lod
        if density is not None:
            cfg["density"] = density
        return bh.load(model, **cfg)

    @staticmethod
    def _current_key(model: str, gender: str, lod: int) -> str:
        if model == "mhr":
            return persistence.model_key(model, lod=lod)
        return persistence.model_key(model, gender=gender)

    @staticmethod
    def _default_frames(model: str) -> dict[str, int]:
        if model == "smpl":
            return dict(zip(SMPL_CONFIG.frame_names, SMPL_CONFIG.frame_vertex_ids, strict=True))
        if model == "smplx":
            return dict(zip(SMPLX_CONFIG.frame_names, SMPLX_CONFIG.frame_vertex_ids, strict=True))
        return {}

    # -- sidebar -------------------------------------------------------------

    def _build_sidebar(self) -> None:
        tabs = self._server.gui.add_tab_group()
        self._body_tab = tabs.add_tab("Body")
        with self._body_tab:
            self._build_body_static()
            self._build_dynamic()
            self._build_pose_controls()
        with tabs.add_tab("Layers"):
            self._build_layer_controls()
        self._segment = SegmentEditor(
            self._server,
            self._session,
            self._paths,
            self.set_segmentation_coloring,
            self._layers["parts"],
            self._set_status,
            self._set_edit_overlay,
        )
        self._frames_editor = FramesEditor(
            self._server,
            self._session,
            self._paths,
            self._set_status,
            lambda: self._default_frames(self._model.value),
            self._set_edit_overlay,
            self._layers["frames"].set_active,
        )
        with tabs.add_tab("Segment"):
            self._segment.build()
        with tabs.add_tab("Frames"):
            self._frames_editor.build()
        self._segment.on_edit_start = self._frames_editor.exit_edit
        self._frames_editor.on_edit_start = self._segment.exit_edit

    def _build_body_static(self) -> None:
        gui = self._server.gui
        self._model = gui.add_dropdown("Model", _MODELS, initial_value=self._session.model_key.split("_")[0])
        self._gender = gui.add_dropdown("Gender", _GENDERS, initial_value="neutral")
        self._lod = gui.add_dropdown("LOD", _LODS, initial_value="1")
        gui.add_button("Load model").on_click(lambda _event: self._reload(density=None))

    def _build_dynamic(self) -> None:
        """Create the reloadable shape, density, and mass section.

        The caller is responsible for making the Body tab the active container,
        so this never re-enters ``self._body_tab`` (whose context is not
        re-entrant).
        """
        gui = self._server.gui
        self._dynamic = gui.add_folder("Shape & density")
        with self._dynamic:
            self._build_shape_controls()
            self._build_density_controls()
            self._mass_md = gui.add_markdown("")
        self._update_mass()

    def _build_shape_controls(self) -> None:
        gui = self._server.gui
        self._shape_sliders = {}
        for group, width in self._session.body.structure.parameter_groups.items():
            with gui.add_folder(group, expand_by_default=width <= 12):
                sliders = []
                for index in range(width):
                    slider = gui.add_slider(f"{group}[{index}]", min=-3.0, max=3.0, step=0.05, initial_value=0.0)
                    slider.on_update(self._make_shape_handler(group))
                    sliders.append(slider)
                self._shape_sliders[group] = sliders
                gui.add_button(f"Reset {group}").on_click(self._make_shape_reset(group))
                gui.add_button(f"Random {group}").on_click(self._make_shape_random(group))

    def _build_density_controls(self) -> None:
        gui = self._server.gui
        tables = tuple(json.loads((_ASSET_DIR / "body_densities.json").read_text()))
        if self._model.value == "smpl":
            options = ("uniform (1000)", *tables)
        else:
            options = ("uniform (1000)", *(f"{table} (name-mapped)" for table in tables))
        self._density = gui.add_dropdown("Density table", options)
        gui.add_button("Apply density").on_click(lambda _event: self._reload(density=self._density_value()))

    def _density_value(self) -> object:
        """Resolve the density dropdown to a scalar, table name, or per-part tensor.

        SMPL keeps the exact table path (a name string); SMPL-X and MHR resolve
        the chosen table by mapping their joint names onto the 24 SMPL density
        categories, producing a per-native-joint tensor and a mapped/unmapped
        note surfaced on the next status refresh.
        """
        value = self._density.value
        self._density_note = ""
        if value == "uniform (1000)":
            return 1000.0
        if self._model.value == "smpl":
            return value
        table_name = value.removesuffix(" (name-mapped)")
        table = json.loads((_ASSET_DIR / "body_densities.json").read_text())[table_name]
        names = self._session.body.structure.joint_names
        mapped = sum(category is not None for category in densities.part_categories(names))
        self._density_note = f", {table_name}: {mapped}/{len(names)} name-mapped"
        exemplar = self._session.body.values.v_template
        return densities.density_for_parts(names, table).to(dtype=exemplar.dtype, device=exemplar.device)

    def _build_pose_controls(self) -> None:
        gui = self._server.gui
        with gui.add_folder("Pose"):
            self._pose_name = gui.add_text("Save pose as", initial_value="pose")
            gui.add_button("Save pose").on_click(lambda _event: self._save_pose())
            self._pose_dropdown = gui.add_dropdown("Saved pose", self._saved_poses())
            gui.add_button("Load pose").on_click(lambda _event: self._load_pose())
            gui.add_button("Reset pose").on_click(lambda _event: self._session.reset_pose())
            gui.add_button("Deselect joint").on_click(lambda _event: self.select_joint(None))

    def _build_layer_controls(self) -> None:
        gui = self._server.gui
        self._add_visibility("Mesh", "mesh", True)
        self._color_checkbox = gui.add_checkbox("Color by segmentation", initial_value=False)
        self._color_checkbox.on_update(lambda _event: self.set_segmentation_coloring(self._color_checkbox.value))
        self._fast_check = gui.add_checkbox(
            "Fast skinning (GPU, no pose correctives)",
            initial_value=False,
            hint="Deform on the GPU: uploads once, writes only bone transforms per pose. "
            "Skips pose correctives and segmentation coloring. MHR uses a top-4 "
            "skin-weight approximation (viser caps influences at 4 per vertex).",
        )
        self._fast_check.on_update(lambda _event: self._toggle_fast())
        self._wireframe_check = gui.add_checkbox(
            "Wireframe",
            initial_value=False,
            hint="Overlay a thin wireframe on top of the solid mesh. Disabled in fast-skinning mode.",
        )
        self._wireframe_check.on_update(lambda _event: self._layers["mesh"].set_wireframe(self._wireframe_check.value))
        self._opacity_slider = gui.add_slider("Mesh opacity", min=0.1, max=1.0, step=0.05, initial_value=0.9)
        self._opacity_slider.on_update(lambda _event: self._on_opacity_change())
        self._add_visibility("Skeleton", "skeleton", True)
        self._radius_slider = gui.add_slider("Joint radius (x)", min=0.2, max=3.0, step=0.05, initial_value=1.0)
        self._radius_slider.on_update(
            lambda _event: self._layers["skeleton"].set_radius_multiplier(self._radius_slider.value)
        )
        self._add_visibility("Frames", "frames", False)
        frame_radius = gui.add_slider("Frame radius", min=0.002, max=0.05, step=0.001, initial_value=0.012)
        frame_radius.on_update(lambda _event: self._layers["frames"].set_radius(frame_radius.value))
        self._add_visibility("Parts", "parts", False)
        explode = gui.add_slider("Explode", min=0.0, max=1.0, step=0.05, initial_value=0.0)
        explode.on_update(lambda _event: self._layers["parts"].set_explode(explode.value))
        self._refresh_fast_availability()

    def _toggle_fast(self) -> None:
        enabled = self._fast_check.value
        if enabled and not fast_skinning_available(self._session.body):
            self._fast_check.value = False
            self._set_status("fast skinning unavailable: body uses non-identity joint scales")
            return
        if enabled:
            # Remember the coloring choice so it survives a fast on/off cycle.
            self._coloring_before_fast = self._color_checkbox.value
            self.set_segmentation_coloring(False)
            self._layers["mesh"].set_fast(True)
        else:
            self._layers["mesh"].set_fast(False)
            self.set_segmentation_coloring(self._coloring_before_fast)
        # Wireframe overlay can't be represented cheaply while GPU-skinning, so
        # grey the checkbox out (keeping fast mode's per-pose cost minimal).
        self._wireframe_check.disabled = enabled
        self._set_status("fast skinning on (no correctives/segmentation colors)" if enabled else "fast skinning off")

    def _refresh_fast_availability(self) -> None:
        available = fast_skinning_available(self._session.body)
        self._fast_check.disabled = not available
        if not available and self._fast_check.value:
            self._fast_check.value = False
            self._layers["mesh"].set_fast(False)
        self._wireframe_check.disabled = self._fast_check.value

    def _on_opacity_change(self) -> None:
        # The slider is the authoritative opacity; a joint selection only dims
        # the mesh transiently (see :meth:`_dim_mesh_for_selection`).
        self._layers["mesh"].set_opacity(self._opacity_slider.value)

    def set_segmentation_coloring(self, enabled: bool) -> None:
        """Toggle mesh segmentation coloring, keeping checkbox and mesh in sync.

        The single entry point for both the Layers checkbox and the editors; a
        reentrancy guard absorbs the checkbox's callback echo.
        """
        if self._coloring_busy:
            return
        self._coloring_busy = True
        try:
            self._color_checkbox.value = enabled
            self._layers["mesh"].set_mode("segmentation" if enabled else "solid")
        finally:
            self._coloring_busy = False

    def _add_visibility(self, label: str, key: str, initial: bool) -> None:
        checkbox = self._server.gui.add_checkbox(label, initial_value=initial)
        checkbox.on_update(lambda _event: self._set_layer_visible(key, checkbox.value))
        self._visibility_checks[key] = checkbox
        self._layers[key].set_visible(initial)

    def _set_layer_visible(self, key: str, visible: bool) -> None:
        if key == "parts" and visible:
            try:
                self._ensure_parts_topology()
            except (ValueError, RuntimeError) as error:
                self._visibility_checks["parts"].value = False
                self._set_status(f"parts layer failed: {error}")
                return
        self._layers[key].set_visible(visible)
        if key == "skeleton":
            self._layers["frames"].set_skeleton_visible(visible)

    def _ensure_parts_topology(self) -> None:
        session = self._session
        part_ids, _names = compact_segmentation(session.body.structure.faces, session.part_ids, session.part_names)
        topology = build_part_topology(session.body.structure.faces, part_ids, len(_names))
        self._layers["parts"].set_topology(topology, part_ids)

    # -- shape handlers ------------------------------------------------------

    def _make_shape_handler(self, group: str):
        def _handler(_event: object) -> None:
            if self._suppress_shape:
                return
            self._apply_shape(group)

        return _handler

    def _make_shape_reset(self, group: str):
        def _handler(_event: object) -> None:
            self._set_group_values(group, [0.0] * len(self._shape_sliders[group]))

        return _handler

    def _make_shape_random(self, group: str):
        def _handler(_event: object) -> None:
            values = torch.empty(len(self._shape_sliders[group])).uniform_(-2.0, 2.0).tolist()
            self._set_group_values(group, values)

        return _handler

    def _set_group_values(self, group: str, values: list[float]) -> None:
        self._suppress_shape = True
        for slider, value in zip(self._shape_sliders[group], values, strict=True):
            slider.value = value
        self._suppress_shape = False
        self._apply_shape(group)

    def _apply_shape(self, group: str) -> None:
        values = [slider.value for slider in self._shape_sliders[group]]
        self._session.set_shape(**{group: self._session.shape_tensor(group, values)})

    # -- posing --------------------------------------------------------------

    @property
    def _gizmo(self) -> viser.TransformControlsHandle | None:
        """The live posing gizmo handle, or ``None`` (kept for test access)."""
        return self._gizmo_ctrl.gizmo

    def select_joint(self, native_index: int | None) -> None:
        """Attach the single posing gizmo to a joint, or detach on ``None``.

        Selecting a joint also dims the mesh so the gizmo handles are visible and
        grabbable even when the joint sits inside the body.
        """
        self._gizmo_ctrl.detach()
        if native_index is None:
            self._selected = None
            self._skeleton.set_selected(None)
            self._restore_mesh_opacity()
            self._set_status("")
            return
        handle = next((item for item in self._session.handles if item.native_index == native_index), None)
        name = self._session.body.structure.joint_names[native_index]
        if handle is None:
            self._selected = None
            self._skeleton.set_selected(None)
            self._restore_mesh_opacity()
            self._set_status(f"{name}: fixed joint - no DOF")
            return
        self._selected = native_index
        self._skeleton.set_selected(native_index)
        self._gizmo_ctrl.attach(handle, self._session.native_poses[native_index])
        self._dim_mesh_for_selection()
        self._set_status(f"selected {name} ({handle.kind})")

    def _select_frame_from_click(self, name: str) -> None:
        """Route a frame-sphere click to the Frames editor's selection."""
        self._frames_editor.select_frame(name)

    def _dim_mesh_for_selection(self) -> None:
        self._layers["mesh"].set_opacity(min(self._opacity_slider.value, 0.35))

    def _restore_mesh_opacity(self) -> None:
        self._layers["mesh"].set_opacity(self._opacity_slider.value)

    # -- edit-mode all-vertices point cloud ---------------------------------

    def _set_edit_overlay(self, active: bool) -> None:
        """Show or hide a neutral point cloud of every mesh vertex while editing."""
        self._clear_edit_pointcloud()
        if active:
            self._edit_pointcloud = self._server.scene.add_point_cloud(
                "/edit/points",
                points=to_numpy(self._session.vertices),
                colors=EDIT_POINT_COLOR,
                point_size=0.004,
            )

    def _update_edit_pointcloud(self) -> None:
        if self._edit_pointcloud is not None:
            self._edit_pointcloud.points = to_numpy(self._session.vertices)

    def _clear_edit_pointcloud(self) -> None:
        if self._edit_pointcloud is not None:
            self._edit_pointcloud.remove()
            self._edit_pointcloud = None

    # -- pose persistence ----------------------------------------------------

    def _saved_poses(self) -> tuple[str, ...]:
        return tuple(persistence.list_saved(self._paths.pose_dir(self._session.model_key))) or ("",)

    def _save_pose(self) -> None:
        name = self._pose_name.value.strip()
        if not name:
            self._set_status("provide a name to save the pose")
            return
        path = self._paths.pose_dir(self._session.model_key) / f"{name}.json"
        persistence.save_pose(path, self._session.q, self._session.model_key)
        self._pose_dropdown.options = self._saved_poses()
        self._set_status(f"saved pose to {path}")

    def _load_pose(self) -> None:
        name = self._pose_dropdown.value
        if not name:
            return
        path = self._paths.pose_dir(self._session.model_key) / f"{name}.json"
        try:
            q = persistence.load_pose(path, self._session.model_key, self._session.body.robot.nq)
        except (ValueError, OSError) as error:
            self._set_status(f"pose load failed: {error}")
            return
        self._session.set_q(q)
        self._set_status(f"loaded pose {name}")

    # -- reload --------------------------------------------------------------

    def _reload(self, *, density: object | None) -> None:
        model, gender, lod = self._model.value, self._gender.value, int(self._lod.value)
        start = time.perf_counter()
        try:
            body = self._load_body(model, gender, lod, density)
        except (FileNotFoundError, ValueError, RuntimeError) as error:
            self._set_status(f"load failed: {error}")
            return
        self._gizmo_ctrl.detach()
        self._selected = None
        self._clear_edit_pointcloud()
        self._session.load_body(body, self._current_key(model, gender, lod), self._default_frames(model), notify=False)
        for layer in self._layers.values():
            layer.clear()
        self._radius_slider.value = 1.0
        self._skeleton.set_base_radius(0.015 if model == "mhr" else 0.02)
        self._skeleton.set_radius_multiplier(1.0)
        for layer in self._layers.values():
            layer.build(self._server, self._session)
        self._layers["frames"].set_skeleton_visible(self._visibility_checks["skeleton"].value)
        self._refresh_fast_availability()
        self._restore_mesh_opacity()
        self._dynamic.remove()
        with self._body_tab:
            self._build_dynamic()
        self._pose_dropdown.options = self._saved_poses()
        self._segment.refresh()
        self._frames_editor.refresh()
        self._refresh_status(load_seconds=time.perf_counter() - start)

    # -- change dispatch and status -----------------------------------------

    def _on_change(self, kind: str) -> None:
        for layer in self._layers.values():
            layer.update(self._session, kind)
        if kind == "shape":
            self._gizmo_ctrl.reseat_external()
            self._update_mass()
        elif kind == "pose" and not self._gizmo_ctrl.applying:
            # Gizmo-driven pose edits never re-seat (that fought the drag); only
            # non-gizmo sources (reset/load) do.
            self._gizmo_ctrl.reseat_external()
        if kind in ("pose", "shape"):
            self._update_edit_pointcloud()

    def _update_mass(self) -> None:
        mass = self._session.total_mass
        self._mass_md.content = "Total mass: unavailable" if mass is None else f"**Total mass:** {mass:.2f} kg"

    def _refresh_status(self, *, load_seconds: float) -> None:
        mass = self._session.total_mass
        mass_text = "n/a" if mass is None else f"{mass:.2f} kg"
        self._set_status(
            f"loaded {self._session.model_key} in {load_seconds:.2f}s, mass {mass_text}{self._density_note}"
        )

    def _set_status(self, message: str) -> None:
        self._status.content = f"**Status:** {message}" if message else "**Status:** ready"

    # -- lifecycle -----------------------------------------------------------

    def run(self, *, block: bool = True) -> None:
        """Serve the studio, optionally blocking the calling thread."""
        if block:
            self._server.sleep_forever()

    def stop(self) -> None:
        """Stop the gizmo worker and the underlying viser server."""
        self._gizmo_ctrl.shutdown()
        self._server.stop()


def launch(**kwargs: object) -> HumanStudio:
    """Construct a :class:`HumanStudio` and run it (blocking)."""
    studio = HumanStudio(**kwargs)
    studio.run()
    return studio


__all__ = ["HumanStudio", "launch"]
