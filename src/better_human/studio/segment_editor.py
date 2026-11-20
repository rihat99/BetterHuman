"""Segment tab: paint vertices into named parts, save/load segmentations, build part bodies."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, final

import numpy as np
import torch
from better_robot.viewer.tools import vertices_in_sphere

from better_human.mass import PartTopology, part_inertias
from better_human.studio import persistence

if TYPE_CHECKING:
    from pathlib import Path

    from better_human.studio.app import HumanStudio
    from better_human.studio.state import StudioSession

_UNASSIGNED = "(unassigned)"


@final
class SegmentEditor:
    """Owns the Segment tab; paints through :meth:`paint` (sphere brush) and :meth:`assign` (marquee)."""

    def __init__(self, studio: HumanStudio) -> None:
        self.studio = studio
        self.painting = False
        self.part = _UNASSIGNED

    @property
    def session(self) -> StudioSession:
        return self.studio.session

    def build(self, sidebar: Any) -> None:
        """Add the Segment section to the viewer's ``Sidebar``."""
        locked, gui = self.studio.locked, sidebar.gui
        self.part_dropdown = sidebar.dropdown(
            "Segment", "Part", options=self._options(), value=_UNASSIGNED, on_change=locked(self._set_part)
        )
        with sidebar.section("Segment"):
            self.new_part = gui.add_text("New part", initial_value="")
        sidebar.button("Segment", "Add part", on_click=locked(lambda _=None: self.add_part(self.new_part.value)))
        self.paint_toggle = sidebar.checkbox(
            "Segment", "Paint (click mesh)", value=False, on_change=locked(lambda value: self.set_painting(bool(value)))
        )
        with sidebar.section("Segment"):
            self.brush = gui.add_slider("Brush radius", min=0.005, max=0.2, step=0.005, initial_value=0.03)
        sidebar.button("Segment", "Marquee select", on_click=locked(lambda _=None: self.studio.start_marquee()))
        with sidebar.section("Segment"):
            self.saved = gui.add_dropdown("Saved", options=self._saved_options())
            self.save_name = gui.add_text("Save as", initial_value="custom")
        for label, action in (
            ("Save segmentation", lambda: self.save(self.save_name.value)),
            ("Load segmentation", lambda: self.load(self.saved.value)),
            ("Load built-in", self.load_builtin),
            ("Reset to skinning", self.session.reset_segmentation),
            ("Build bodies", self.build_bodies),
        ):
            sidebar.button("Segment", label, on_click=locked(lambda _=None, action=action: action()))
        with sidebar.section("Segment"):
            self.mass_table = gui.add_markdown("")

    def _set_part(self, part: str) -> None:
        self.part = part

    def _options(self) -> tuple[str, ...]:
        return (_UNASSIGNED, *self.session.part_names)

    def _directory(self) -> Path:
        return self.studio.paths.segmentation_dir(self.session.model_key)

    def _saved_options(self) -> tuple[str, ...]:
        return tuple(persistence.list_saved(self._directory())) or ("(none)",)

    def refresh(self) -> None:
        """Re-sync dropdowns after a body or segmentation change."""
        self.part_dropdown.options = self._options()
        self.saved.options = self._saved_options()

    def set_painting(self, enabled: bool) -> None:  # noqa: FBT001 - the switch is the main operand
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
        ids = vertices_in_sphere(self.session.vertices.detach().cpu().numpy(), center, radius=float(self.brush.value))
        self.assign(ids)

    def assign(self, vertex_ids: np.ndarray) -> None:
        """Assign ``vertex_ids (n,)`` to the active part (``-1`` for unassigned)."""
        label = -1 if self.part == _UNASSIGNED else self.session.part_names.index(self.part)
        part_ids = self.session.part_ids.clone()
        part_ids[torch.as_tensor(np.asarray(vertex_ids), dtype=torch.long)] = label
        self.session.set_segmentation(self.session.part_names, part_ids)
        self.studio.layers.set_edit_vertices(np.flatnonzero(part_ids.cpu().numpy() == label) if label >= 0 else [])

    def _as_parts(self) -> dict[str, list[int]]:
        ids = self.session.part_ids
        parts = {name: torch.nonzero(ids == k).flatten().tolist() for k, name in enumerate(self.session.part_names)}
        return {name: vertices for name, vertices in parts.items() if vertices}

    def _apply(self, parts: dict[str, list[int]], *, source: str) -> None:
        persistence.check_vertex_ids(
            (vertex for ids in parts.values() for vertex in ids), self.session.vertices.shape[0], source=source
        )
        part_ids = torch.full((self.session.vertices.shape[0],), -1, dtype=torch.long)
        for k, vertices in enumerate(parts.values()):
            part_ids[torch.as_tensor(vertices, dtype=torch.long)] = k
        self.session.set_segmentation(tuple(parts), part_ids)
        self.refresh()

    def save(self, name: str) -> None:
        """Write the current segmentation to ``<data>/segmentations/<model_key>/<name>.json``; ``name`` is one file
        name."""
        path = persistence.saved_path(self._directory(), name)
        persistence.write_segmentation(path, self._as_parts())
        self.refresh()
        self.studio.set_status(f"saved segmentation {path}")

    def load(self, name: str) -> None:
        """Load a saved segmentation by name."""
        path = persistence.list_saved(self._directory()).get(name)
        if path is None:
            self.studio.set_status(f"no saved segmentation {name!r}")
            return
        self._apply(persistence.read_segmentation(path), source=str(path))

    def load_builtin(self) -> None:
        """Load the shipped vertex segmentation of SMPL / SMPL-X."""
        family = self.studio.family
        self._apply(persistence.builtin_segmentation(family), source=f"built-in segmentation {family!r}")

    def build_bodies(self) -> list[tuple[str, float]]:
        """Close each part mesh, integrate its mass at the session density and fill the mass table.

        Masses use the rest shape (``identity.rest_vertices``, identity part poses), as ``HumanModel.bake`` does; returns ``[(part, kg), ...]`` with a trailing ``("total", kg)``.
        """
        session, num_parts = self.session, len(self.session.part_names)
        body = session.body
        authored = body.part_topology is not None and num_parts == body.spec.parts.count
        if authored and torch.equal(torch.as_tensor(session.part_ids).to(body.spec.parts.ids), body.spec.parts.ids):
            topology = (
                body.part_topology
            )  # the body's own (possibly authored, closed) parts: same mass as ``HumanModel.bake``
        else:
            topology = PartTopology.from_part_ids(body.spec.faces, session.part_ids, num_parts=num_parts)
        poses = torch.zeros(num_parts, 7)
        poses[:, 6] = 1.0
        density = session.body.density
        if density.numel() != num_parts:
            density = density.mean().expand(num_parts)
        masses = part_inertias(
            session.identity.rest_vertices.detach(), poses, topology=topology, density=density, check_mesh=True
        )[:, 0].tolist()
        rows = [*zip(session.part_names, masses, strict=True), ("total", float(sum(masses)))]
        table = "\n".join(f"| {name} | {mass:.2f} |" for name, mass in rows)
        self.mass_table.content = f"| part | kg |\n|---|---|\n{table}"
        self.studio.set_status(f"built {num_parts} part bodies, total {rows[-1][1]:.2f} kg")
        return rows
