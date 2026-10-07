"""Segment tab: paint vertices into named parts, save/load segmentations, build part bodies."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from better_robot.viewer.tools import vertices_in_sphere

from better_human.core.mass import body_inertias_from_shape, build_part_topology
from better_human.viz.studio import persistence

if TYPE_CHECKING:
    from better_human.viz.studio.app import HumanStudio

_BUILTIN = {"smpl": "smpl_vert_segmentation.json", "smplx": "smplx_vert_segmentation.json"}
_UNASSIGNED = "(unassigned)"


class SegmentEditor:
    """Owns the Segment tab; paints through :meth:`paint` (sphere brush) and :meth:`assign` (marquee)."""

    def __init__(self, studio: HumanStudio) -> None:
        self.studio = studio
        self.painting = False
        self.part = _UNASSIGNED

    @property
    def session(self):  # noqa: ANN201
        return self.studio.session

    def build(self, gui) -> None:  # noqa: ANN001
        """Add the Segment tab controls to ``gui`` (a viser tab or the GUI root)."""
        layers = self.studio.layers
        self.color_mode = gui.add_dropdown("Colour", options=("skin", "segmentation", "parts"), initial_value=layers.mesh.mode)
        self.color_mode.on_update(self.studio._locked(lambda _event: layers.set_mode(self.color_mode.value)))
        self.part_dropdown = gui.add_dropdown("Part", options=self._options(), initial_value=_UNASSIGNED)
        self.part_dropdown.on_update(self.studio._locked(lambda _event: setattr(self, "part", self.part_dropdown.value)))
        self.new_part = gui.add_text("New part", initial_value="")
        gui.add_button("Add part").on_click(self.studio._locked(lambda _event: self.add_part(self.new_part.value)))
        self.paint_toggle = gui.add_checkbox("Paint (click mesh)", initial_value=False)
        self.paint_toggle.on_update(self.studio._locked(lambda _event: self.set_painting(bool(self.paint_toggle.value))))
        self.brush = gui.add_slider("Brush radius", min=0.005, max=0.2, step=0.005, initial_value=0.03)
        gui.add_button("Marquee select").on_click(self.studio._locked(lambda _event: self.studio.start_marquee()))
        self.saved = gui.add_dropdown("Saved", options=self._saved_options(), initial_value=self._saved_options()[0])
        self.save_name = gui.add_text("Save as", initial_value="custom")
        gui.add_button("Save segmentation").on_click(self.studio._locked(lambda _event: self.save(self.save_name.value)))
        gui.add_button("Load segmentation").on_click(self.studio._locked(lambda _event: self.load(self.saved.value)))
        gui.add_button("Load built-in").on_click(self.studio._locked(lambda _event: self.load_builtin()))
        gui.add_button("Reset to skinning").on_click(self.studio._locked(lambda _event: self.session.reset_segmentation()))
        gui.add_button("Build bodies").on_click(self.studio._locked(lambda _event: self.build_bodies()))
        self.mass_table = gui.add_markdown("")

    def _options(self) -> tuple[str, ...]:
        return (_UNASSIGNED, *self.session.part_names)

    def _directory(self):  # noqa: ANN202
        return self.studio.paths.segmentation_dir(self.session.model_key)

    def _saved_options(self) -> tuple[str, ...]:
        return tuple(persistence.list_saved(self._directory())) or ("(none)",)

    def refresh(self) -> None:
        """Re-sync dropdowns after a body or segmentation change."""
        if hasattr(self, "part_dropdown"):
            self.part_dropdown.options = self._options()
            self.saved.options = self._saved_options()

    def set_painting(self, enabled: bool) -> None:
        """Enter or leave sphere-brush painting; it is exclusive with the other click modes."""
        self.painting = enabled
        if enabled:
            self.studio.set_mode("segment")
        elif self.studio.mode == "segment":
            self.studio.set_mode("pose")

    def add_part(self, name: str) -> None:
        """Append an empty part and make it the paint target."""
        name = name.strip()
        if not name or name in self.session.part_names:
            self.studio.set_status(f"part name {name!r} is empty or taken")
            return
        self.session.set_segmentation((*self.session.part_names, name), self.session.part_ids)
        self.part = name
        self.refresh()

    def paint(self, center: np.ndarray) -> None:
        """Assign every vertex within the brush radius of ``center (3,)`` to the active part."""
        ids = vertices_in_sphere(self.session.vertices.detach().numpy(), center, float(self.brush.value))
        self.assign(ids)

    def assign(self, vertex_ids: np.ndarray) -> None:
        """Assign ``vertex_ids (n,)`` to the active part (``-1`` for unassigned)."""
        label = -1 if self.part == _UNASSIGNED else self.session.part_names.index(self.part)
        part_ids = self.session.part_ids.clone()
        part_ids[torch.as_tensor(np.asarray(vertex_ids), dtype=torch.long)] = label
        self.session.set_segmentation(self.session.part_names, part_ids)
        self.studio.layers.set_edit_vertices(np.flatnonzero(part_ids.numpy() == label) if label >= 0 else [])

    def _as_parts(self) -> dict[str, list[int]]:
        ids = self.session.part_ids
        parts = {name: torch.nonzero(ids == k).flatten().tolist() for k, name in enumerate(self.session.part_names)}
        return {name: vertices for name, vertices in parts.items() if vertices}

    def _apply(self, parts: dict[str, list[int]]) -> None:
        part_ids = torch.full((self.session.vertices.shape[0],), -1, dtype=torch.long)
        for k, vertices in enumerate(parts.values()):
            part_ids[torch.as_tensor(vertices, dtype=torch.long)] = k
        self.session.set_segmentation(tuple(parts), part_ids)
        self.refresh()

    def save(self, name: str) -> None:
        """Write the current segmentation to ``<data>/segmentations/<model_key>/<name>.json``."""
        path = self._directory() / f"{name}.json"
        persistence.save_segmentation(path, self._as_parts())
        self.refresh()
        self.studio.set_status(f"saved segmentation {path}")

    def load(self, name: str) -> None:
        """Load a saved segmentation by name."""
        path = persistence.list_saved(self._directory()).get(name)
        if path is None:
            self.studio.set_status(f"no saved segmentation {name!r}")
            return
        self._apply(persistence.load_segmentation(path, self.session.vertices.shape[0]))

    def load_builtin(self) -> None:
        """Load the shipped vertex segmentation of SMPL / SMPL-X."""
        filename = _BUILTIN.get(self.studio.family)
        if filename is None:
            self.studio.set_status(f"no built-in segmentation for {self.studio.family!r}")
            return
        self._apply(persistence.load_builtin_segmentation(filename, self.session.vertices.shape[0]))

    def build_bodies(self) -> list[tuple[str, float]]:
        """Close each part mesh, integrate its mass at the session density and fill the mass table.

        Masses use the rest shape (``identity.v_shaped``, identity part poses), as ``Body.bake`` does; returns ``[(part, kg), ...]`` with a trailing ``("total", kg)``.
        """
        session, num_parts = self.session, len(self.session.part_names)
        body = session.body
        authored = body.topology is not None and num_parts == body.spec.parts.count
        if authored and torch.equal(torch.as_tensor(session.part_ids).to(body.spec.parts.ids), body.spec.parts.ids):
            topology = body.topology  # the body's own (possibly authored, closed) parts: same mass as ``Body.bake``
        else:
            topology = build_part_topology(body.faces, session.part_ids, num_parts)
        poses = torch.zeros(num_parts, 7)
        poses[:, 6] = 1.0
        density = session.body.density
        if density.numel() != num_parts:
            density = density.mean().expand(num_parts)
        masses = body_inertias_from_shape(session.identity.v_shaped.detach(), poses, topology, density)[1:, 0].tolist()
        rows = [*zip(session.part_names, masses, strict=True), ("total", float(sum(masses)))]
        table = "\n".join(f"| {name} | {mass:.2f} |" for name, mass in rows)
        self.mass_table.content = f"| part | kg |\n|---|---|\n{table}"
        self.studio.set_status(f"built {num_parts} part bodies, total {rows[-1][1]:.2f} kg")
        return rows
