"""Frames tab: add contact frames on mesh vertices, rename, re-parent, delete, save and export."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from better_human.viz.studio import persistence
from better_human.viz.studio.state import FrameEntry

if TYPE_CHECKING:
    from better_human.viz.studio.app import HumanStudio

_DEFAULT_JOINT = "(skinning default)"


class FramesEditor:
    """Owns the Frames tab; :meth:`add_at` places a frame on the vertex nearest a clicked point."""

    def __init__(self, studio: HumanStudio) -> None:
        self.studio = studio
        self.adding = False
        self.active: str | None = None
        self.snippet = ""

    @property
    def session(self):  # noqa: ANN201
        return self.studio.session

    def build(self, gui) -> None:  # noqa: ANN001
        """Add the Frames tab controls to ``gui``."""
        self.add_toggle = gui.add_checkbox("Add frames (click mesh)", initial_value=False)
        self.add_toggle.on_update(self.studio._locked(lambda _event: self.set_adding(bool(self.add_toggle.value))))
        self.frame_dropdown = gui.add_dropdown("Frame", options=self._names(), initial_value=self._names()[0])
        self.frame_dropdown.on_update(self.studio._locked(lambda _event: self.select_frame(self.frame_dropdown.value)))
        self.rename_text = gui.add_text("Name", initial_value="")
        gui.add_button("Rename").on_click(self.studio._locked(lambda _event: self.rename(self.active, self.rename_text.value)))
        joints = (_DEFAULT_JOINT, *self.session.body.joint_names)
        self.joint_dropdown = gui.add_dropdown("Parent joint", options=joints, initial_value=_DEFAULT_JOINT)
        self.joint_dropdown.on_update(self.studio._locked(lambda _event: self.set_joint(self.active, self.joint_dropdown.value)))
        gui.add_button("Delete").on_click(self.studio._locked(lambda _event: self.delete(self.active)))
        self.saved = gui.add_dropdown("Saved", options=self._saved_options(), initial_value=self._saved_options()[0])
        gui.add_button("Load frames").on_click(self.studio._locked(lambda _event: self.load(self.saved.value)))
        self.save_name = gui.add_text("Save as", initial_value="frames")
        gui.add_button("Save frames").on_click(self.studio._locked(lambda _event: self.save(self.save_name.value)))
        gui.add_button("Export as config").on_click(self.studio._locked(lambda _event: self.export()))
        self.export_box = gui.add_markdown("")

    def _saved_options(self) -> tuple[str, ...]:
        return tuple(persistence.list_saved(self.studio.paths.frame_dir(self.session.model_key))) or ("(none)",)

    def _names(self) -> tuple[str, ...]:
        return tuple(self.session.frame_set) or ("(none)",)

    def refresh(self) -> None:
        """Re-sync the frame dropdown with the session frame set."""
        if hasattr(self, "frame_dropdown"):
            self.frame_dropdown.options = self._names()

    def set_adding(self, enabled: bool) -> None:
        """Enter or leave add-frame mode; it is exclusive with the other click modes."""
        self.adding = enabled
        if enabled:
            self.studio.set_mode("frames")
        elif self.studio.mode == "frames":
            self.studio.set_mode("pose")

    def add_at(self, point: np.ndarray) -> str:
        """Add a frame on the vertex nearest ``point (3,)``; returns its generated name."""
        vertices = self.session.vertices.detach().numpy()
        vertex = int(np.argmin(((vertices - point) ** 2).sum(-1)))
        index = len(self.session.frame_set)
        while f"frame_{index}" in self.session.frame_set:
            index += 1
        name = f"frame_{index}"
        self.session.set_frame_set({**self.session.frame_set, name: FrameEntry(vertex)})
        self.select_frame(name)
        return name

    def select_frame(self, name: str) -> None:
        """Make ``name`` the frame the rename / parent / delete controls act on."""
        if name not in self.session.frame_set:
            return
        self.active = name
        self.refresh()
        entry = self.session.frame_set[name]
        self.frame_dropdown.value = name
        self.rename_text.value = name
        self.joint_dropdown.value = (
            _DEFAULT_JOINT if entry.joint is None else self.session.body.joint_names[entry.joint]
        )
        self.studio.layers.set_edit_vertices([entry.vertex])
        self.studio.set_status(f"frame {name!r}: vertex {entry.vertex}, joint {self.session.resolved_joint(entry)}")

    def rename(self, old: str | None, new: str) -> None:
        """Rename a frame, keeping its position in the authored order."""
        new = new.strip()
        if old is None or not new or new in self.session.frame_set:
            self.studio.set_status(f"cannot rename {old!r} to {new!r}")
            return
        self.session.set_frame_set({(new if key == old else key): v for key, v in self.session.frame_set.items()})
        self.select_frame(new)

    def set_joint(self, name: str | None, joint_name: str) -> None:
        """Override the parent joint of ``name``; the default entry restores the skinning argmax."""
        if name is None:
            return
        joint = None if joint_name == _DEFAULT_JOINT else self.session.body.joint_names.index(joint_name)
        self.session.set_frame_joint(name, joint)

    def delete(self, name: str | None) -> None:
        """Remove a frame."""
        if name is None or name not in self.session.frame_set:
            return
        self.session.set_frame_set({key: v for key, v in self.session.frame_set.items() if key != name})
        self.active = None
        self.studio.layers.set_edit_vertices([])
        self.refresh()

    def _as_io(self) -> dict[str, tuple[int, str | None]]:
        names = self.session.body.joint_names
        return {
            key: (entry.vertex, None if entry.joint is None else names[entry.joint])
            for key, entry in self.session.frame_set.items()
        }

    def save(self, name: str) -> None:
        """Write the frame set to ``<data>/frames/<model_key>/<name>.json``."""
        path = self.studio.paths.frame_dir(self.session.model_key) / f"{name}.json"
        persistence.save_frame_set(path, self._as_io())
        self.saved.options = self._saved_options()
        self.saved.value = name
        self.studio.set_status(f"saved frames {path}")

    def load(self, name: str) -> None:
        """Load a saved frame set by name."""
        path = self.studio.paths.frame_dir(self.session.model_key) / f"{name}.json"
        if not path.exists():
            self.studio.set_status(f"no saved frame set {name!r}")
            return
        raw = persistence.load_frame_set(path, self.session.vertices.shape[0])
        names = self.session.body.joint_names
        self.session.set_frame_set(
            {
                key: FrameEntry(vertex, None if joint is None else names.index(joint))
                for key, (vertex, joint) in raw.items()
            }
        )
        self.refresh()

    def export(self) -> str:
        """Render the frame set as a ``contact_frames={...}`` argument for ``bh.load`` and show it."""
        self.snippet = persistence.frame_config_snippet(self._as_io())
        self.export_box.content = f"```python\n{self.snippet}\n```"
        return self.snippet
