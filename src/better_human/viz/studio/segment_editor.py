"""Segmentation editor: pick, paint, save, and build closed part bodies.

Wires the Segment tab to the session: it loads argmax / built-in / saved
segmentations, paints vertices onto parts with a sphere brush or a rectangle
marquee, saves the result, and builds closed part topologies whose per-part and
total masses are reported at the loaded body's density.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import viser
from better_robot.spatial import Inertia

from better_human.core.mass import build_part_topology, meshes_from_topology, segment_vertices

from . import persistence, picking
from .layers.base import SELECTION_COLOR
from .layers.parts import PartsLayer
from .state import StudioSession, compact_segmentation

_OVERLAY = "/segment/selection"
_ARGMAX_ALL = "argmax (all native joints)"


class SegmentEditor:
    """Segment-tab controller for authoring and building part segmentations."""

    def __init__(
        self,
        server: viser.ViserServer,
        session: StudioSession,
        paths: persistence.StudioPaths,
        set_coloring: Callable[[bool], None],
        parts: PartsLayer,
        status: Callable[[str], None],
        set_edit_overlay: Callable[[bool], None],
    ) -> None:
        self._server = server
        self._session = session
        self._paths = paths
        self._set_coloring = set_coloring
        self._parts = parts
        self._status = status
        self._set_edit_overlay = set_edit_overlay
        self._editing = False
        self._overlay: object | None = None
        self._click = None
        self._rect = None
        self.on_edit_start: Callable[[], None] = lambda: None

    def build(self) -> None:
        """Populate the Segment tab GUI for the active session."""
        gui = self._server.gui
        self._source = gui.add_dropdown("Segmentation", self._source_options(), initial_value="argmax")
        self._source.on_update(lambda _event: self._load_source())
        self._edit = gui.add_checkbox("Edit mode", initial_value=False)
        self._edit.on_update(lambda _event: self._toggle_edit())
        self._active = gui.add_dropdown("Active part", self._session.part_names)
        self._active.on_update(lambda _event: self._refresh_overlay())
        self._new_name = gui.add_text("New part", initial_value="")
        gui.add_button("Add part").on_click(lambda _event: self._add_part())
        self._brush = gui.add_slider("Brush radius", min=0.01, max=0.3, step=0.005, initial_value=0.05)
        self._tool = gui.add_dropdown("Tool", ("brush", "rect"), initial_value="brush")
        self._action = gui.add_dropdown("Action", ("assign", "unassign"), initial_value="assign")
        self._save_name = gui.add_text("Save as", initial_value="custom")
        gui.add_button("Save segmentation").on_click(lambda _event: self._save())
        gui.add_button("Build bodies").on_click(lambda _event: self._build_bodies())
        self._mass = gui.add_markdown("")

    # -- segmentation sources ------------------------------------------------

    def _source_options(self) -> tuple[str, ...]:
        options = ["argmax", _ARGMAX_ALL]
        options.extend(self._builtin_files())
        options.extend(persistence.list_saved(self._paths.segmentation_dir(self._session.model_key)))
        return tuple(dict.fromkeys(options))

    def _builtin_files(self) -> list[str]:
        family = self._session.model_key.split("_")[0]
        return {"smpl": ["smpl_vert_segmentation.json"], "smplx": ["smplx_vert_segmentation.json"]}.get(family, [])

    def _load_source(self) -> None:
        choice = self._source.value
        try:
            if choice == "argmax":
                self._session.reset_segmentation()
            elif choice == _ARGMAX_ALL:
                self._install_argmax_all()
            elif choice.endswith("_vert_segmentation.json"):
                parts = persistence.load_builtin_segmentation(choice, self._session.body.structure.num_vertices)
                self._install_named(parts)
            else:
                path = self._paths.segmentation_dir(self._session.model_key) / f"{choice}.json"
                parts = persistence.load_segmentation(path, self._session.body.structure.num_vertices)
                self._install_named(parts)
        except (ValueError, OSError) as error:
            self._status(f"segmentation load failed: {error}")
            return
        self._set_coloring(True)
        self._sync_active()
        self._refresh_overlay()

    def _install_named(self, parts: dict[str, list[int]]) -> None:
        names = tuple(parts)
        part_ids = torch.full((self._session.body.structure.num_vertices,), -1, dtype=torch.long)
        for index, ids in enumerate(parts.values()):
            part_ids[torch.tensor(ids, dtype=torch.long)] = index
        self._session.set_segmentation(names, part_ids)

    def _install_argmax_all(self) -> None:
        """Segment every vertex to its strongest-weighted native joint.

        Unlike the default authored/argmax segmentation, this exposes all native
        joints that win at least one vertex (for MHR this is far more than the
        ~19 authored groups); part names are the winning joint names.
        """
        body = self._session.body
        winners = segment_vertices(body.values.skinning_weights, body.structure.skinning_indices)
        present = sorted({int(joint) for joint in winners.unique().tolist()})
        remap = {joint: index for index, joint in enumerate(present)}
        part_ids = torch.tensor([remap[int(joint)] for joint in winners.tolist()], dtype=torch.long)
        names = tuple(body.structure.joint_names[joint] for joint in present)
        self._session.set_segmentation(names, part_ids)

    # -- editing -------------------------------------------------------------

    def _toggle_edit(self) -> None:
        if self._edit.value:
            self.on_edit_start()
            self._editing = True
            self._set_coloring(True)
            self._set_edit_overlay(True)
            self._register()
            self._refresh_overlay()
            self._status("segmentation edit on: click to paint the active part")
        else:
            self.exit_edit()

    def exit_edit(self) -> None:
        """Leave edit mode, dropping pointer callbacks and the overlay."""
        self._editing = False
        self._edit.value = False
        self._unregister()
        self._clear_overlay()
        self._set_edit_overlay(False)

    def _register(self) -> None:
        self._server.scene.remove_click_callback("all")
        self._server.scene.remove_rect_select_callback("all")
        self._click = self._server.scene.on_click()(self._on_click)
        self._rect = self._server.scene.on_rect_select()(self._on_rect)

    def _unregister(self) -> None:
        if self._click is not None:
            self._server.scene.remove_click_callback(self._click)
            self._click = None
        if self._rect is not None:
            self._server.scene.remove_rect_select_callback(self._rect)
            self._rect = None

    def _on_click(self, event: viser.SceneClickEvent) -> None:
        if not self._editing or self._tool.value != "brush":
            return
        origin = torch.tensor(event.ray_origin, dtype=torch.float32)
        direction = torch.tensor(event.ray_direction, dtype=torch.float32)
        hit = picking.ray_hit(origin, direction, self._session.vertices, self._session.body.structure.faces)
        if hit is None:
            return
        selected = picking.vertices_in_sphere(self._session.vertices, hit[0], self._brush.value)
        self._apply(selected)

    def _on_rect(self, event: viser.SceneRectSelectEvent) -> None:
        if not self._editing or self._tool.value != "rect":
            return
        camera = event.client.camera
        frustum = picking.CameraFrustum(
            position=torch.tensor(camera.position, dtype=torch.float32),
            wxyz=torch.tensor(camera.wxyz, dtype=torch.float32),
            fov=float(camera.fov),
            aspect=float(camera.aspect),
        )
        selected = picking.vertices_in_rect(
            self._session.vertices,
            frustum,
            torch.tensor(event.screen_min, dtype=torch.float32),
            torch.tensor(event.screen_max, dtype=torch.float32),
        )
        self._apply(selected)

    def _apply(self, selected: torch.Tensor) -> None:
        if selected.numel() == 0:
            return
        part_ids = self._session.part_ids.clone()
        value = self._active_index() if self._action.value == "assign" else -1
        part_ids[selected] = value
        self._session.set_segmentation(self._session.part_names, part_ids)
        self._refresh_overlay()

    # -- parts and mass ------------------------------------------------------

    def _add_part(self) -> None:
        name = self._new_name.value.strip()
        if not name or name in self._session.part_names:
            self._status("part name must be new and non-empty")
            return
        names = (*self._session.part_names, name)
        self._session.set_segmentation(names, self._session.part_ids)
        self._sync_active()
        self._active.value = name
        self._refresh_overlay()

    def _build_bodies(self) -> None:
        session = self._session
        part_ids, names = compact_segmentation(session.body.structure.faces, session.part_ids, session.part_names)
        try:
            topology = build_part_topology(session.body.structure.faces, part_ids, len(names))
        except ValueError as error:
            self._status(f"build bodies failed: {error}")
            return
        self._parts.set_topology(topology, part_ids)
        density = float(session.body.values.density.float().mean())
        meshes = meshes_from_topology(session.body.values.v_shaped, topology)
        rows = [
            f"_density: uniform {density:.0f} kg/m3 (mean of baked per-part density)_",
            "",
            "| part | mass (kg) |",
            "| --- | --- |",
        ]
        total = 0.0
        for name, (vertices, faces) in zip(names, meshes, strict=True):
            mass = float(Inertia.from_mesh(vertices, faces, density=density).data[..., 0])
            total += mass
            rows.append(f"| {name} | {mass:.3f} |")
        rows.append(f"| **total** | **{total:.3f}** |")
        self._mass.content = "\n".join(rows)
        self._status(f"built {len(names)} bodies at {density:.0f} kg/m3, total {total:.2f} kg")

    # -- helpers -------------------------------------------------------------

    def refresh(self) -> None:
        """Repopulate source and part dropdowns after a body reload."""
        self._source.options = self._source_options()
        self._source.value = "argmax"
        self._sync_active()
        self._mass.content = ""
        self.exit_edit()

    def _sync_active(self) -> None:
        self._active.options = self._session.part_names or ("",)
        if self._active.value not in self._session.part_names and self._session.part_names:
            self._active.value = self._session.part_names[0]

    def _active_index(self) -> int:
        try:
            return self._session.part_names.index(self._active.value)
        except ValueError:
            return 0

    def _refresh_overlay(self) -> None:
        self._clear_overlay()
        if not self._editing:
            return
        mask = self._session.part_ids == self._active_index()
        points = self._session.vertices[mask]
        if points.shape[0] == 0:
            return
        self._overlay = self._server.scene.add_point_cloud(
            _OVERLAY,
            points=points.detach().cpu().numpy(),
            colors=SELECTION_COLOR,
            point_size=0.006,
        )

    def _clear_overlay(self) -> None:
        if self._overlay is not None:
            self._overlay.remove()
            self._overlay = None

    def _save(self) -> None:
        name = self._save_name.value.strip()
        if not name:
            self._status("provide a name to save the segmentation")
            return
        parts: dict[str, list[int]] = {}
        for index, part_name in enumerate(self._session.part_names):
            ids = torch.nonzero(self._session.part_ids == index, as_tuple=False).flatten().tolist()
            if ids:
                parts[part_name] = ids
        path = self._paths.segmentation_dir(self._session.model_key) / f"{name}.json"
        persistence.save_segmentation(path, parts)
        self._source.options = self._source_options()
        self._status(f"saved segmentation to {path}")


__all__ = ["SegmentEditor"]
