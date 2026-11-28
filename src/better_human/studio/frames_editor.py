"""Frames tab: add contact frames on mesh vertices, rename, re-parent, delete, save and export."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, final

import numpy as np

from better_robot.exceptions import UnknownNameError

from better_human.io import read_frame_set, write_frame_set
from better_human.studio import persistence
from better_human.studio.state import FrameEntry, contact_frames

if TYPE_CHECKING:
    from better_human.studio.app import HumanStudio
    from better_human.studio.state import StudioSession

_DEFAULT_JOINT = "(skinning default)"


@final
class FramesEditor:
    """Owns the Frames tab; :meth:`add_at` places a frame on the vertex nearest a clicked point.

    ``locked`` wraps each control's callback as the studio runs it; ``on_status`` shows a message on the studio's
    status line.
    """

    def __init__(
        self,
        studio: HumanStudio,
        *,
        locked: Callable[[Callable[[Any], None]], Callable[..., None]],
        on_status: Callable[[str], None],
    ) -> None:
        self.studio = studio
        self.locked, self.on_status = locked, on_status
        self.adding = False
        self.active: str | None = None
        self.snippet = ""

    @property
    def session(self) -> StudioSession:
        return self.studio.session

    def build(self, sidebar: Any) -> None:
        """Add the Frames section to the viewer's ``Sidebar``."""
        locked, gui = self.locked, sidebar.gui
        self.add_toggle = sidebar.checkbox(
            "Frames",
            "Add frames (click mesh)",
            value=False,
            on_change=locked(lambda value: self.set_adding(bool(value))),
        )
        self.frame_dropdown = sidebar.dropdown(
            "Frames", "Frame", options=self._names(), value=self._names()[0], on_change=locked(self.select_frame)
        )
        with sidebar.section("Frames"):
            self.rename_text = gui.add_text("Name", initial_value="")
        sidebar.button(
            "Frames", "Rename", on_click=locked(lambda _=None: self.rename(self.active, self.rename_text.value))
        )
        self.joint_dropdown = sidebar.dropdown(
            "Frames", "Parent joint", options=self._joints(), value=_DEFAULT_JOINT,
            on_change=locked(lambda joint: self.set_joint(self.active, joint)),
        )  # fmt: skip
        sidebar.button("Frames", "Delete", on_click=locked(lambda _=None: self.delete(self.active)))
        with sidebar.section("Frames"):
            self.saved = gui.add_dropdown("Saved", options=self._saved_options())
        sidebar.button("Frames", "Load frames", on_click=locked(lambda _=None: self.load(self.saved.value)))
        with sidebar.section("Frames"):
            self.save_name = gui.add_text("Save as", initial_value="frames")
        sidebar.button("Frames", "Save frames", on_click=locked(lambda _=None: self.save(self.save_name.value)))
        sidebar.button("Frames", "Export as config", on_click=locked(lambda _=None: self.export()))
        with sidebar.section("Frames"):
            self.export_box = gui.add_markdown("")

    def _joints(self) -> tuple[str, ...]:
        return (_DEFAULT_JOINT, *self.session.body.spec.joint_names)

    def _saved_options(self) -> tuple[str, ...]:
        return tuple(persistence.list_saved(self.studio.paths.frame_dir(self.session.model_key))) or ("(none)",)

    def _names(self) -> tuple[str, ...]:
        return tuple(self.session.frame_set) or ("(none)",)

    def refresh(self) -> None:
        """Re-sync the frame dropdown with the session frame set."""
        self.frame_dropdown.options = self._names()

    def set_adding(self, enabled: bool) -> None:  # noqa: FBT001 - the switch is the main operand
        """Enter or leave add-frame mode; it is exclusive with the other click modes."""
        self.adding = enabled
        if enabled:
            self.studio.set_mode("frames")
        elif self.studio.mode == "frames":
            self.studio.set_mode("pose")

    def add_at(self, point: np.ndarray) -> str:
        """Add a frame on the vertex nearest ``point (3,)``; returns its generated name."""
        vertices = self.session.vertices.detach().cpu().numpy()
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
        sidebar = self.studio.sidebar
        sidebar.set_value("Frames", "Frame", value=name)
        self.rename_text.value = name
        joint = _DEFAULT_JOINT if entry.joint is None else self.session.body.spec.joint_names[entry.joint]
        sidebar.set_value("Frames", "Parent joint", value=joint)
        self.studio.layers.set_edit_vertices([entry.vertex])
        self.on_status(f"frame {name!r}: vertex {entry.vertex}, joint {joint}")

    def rename(self, old: str | None, new: str) -> None:
        """Rename a frame, keeping its position in the authored order."""
        new = new.strip()
        if old is None or not new or new in self.session.frame_set:
            self.on_status(f"cannot rename {old!r} to {new!r}")
            return
        self.session.set_frame_set({(new if key == old else key): v for key, v in self.session.frame_set.items()})
        self.select_frame(new)

    def set_joint(self, name: str | None, joint_name: str) -> None:
        """Override the parent joint of ``name``; the default entry restores the skinning argmax."""
        if name is None:
            return
        joint = None if joint_name == _DEFAULT_JOINT else self.session.body.spec.joint_names.index(joint_name)
        self.session.set_frame_joint(name, joint)

    def delete(self, name: str | None) -> None:
        """Remove a frame."""
        if name is None or name not in self.session.frame_set:
            return
        self.session.set_frame_set({key: v for key, v in self.session.frame_set.items() if key != name})
        self.active = None
        self.studio.layers.set_edit_vertices([])
        self.refresh()

    def save(self, name: str) -> None:
        """Write the frame set to ``<data>/frames/<model_key>/<name>.json``; ``name`` is one file name."""
        path = persistence.saved_path(self.studio.paths.frame_dir(self.session.model_key), name)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_frame_set(path, contact_frames(self.session.frame_set, self.session.body.spec.joint_names))
        self.saved.options = self._saved_options()
        self.saved.value = name
        self.on_status(f"saved frames {path}")

    def load(self, name: str) -> None:
        """Load a saved frame set by name."""
        path = persistence.saved_path(self.studio.paths.frame_dir(self.session.model_key), name)
        if not path.exists():
            self.on_status(f"no saved frame set {name!r}")
            return
        raw = read_frame_set(path)
        persistence.check_vertex_ids(
            (vertex for vertex, _ in raw.values()), self.session.vertices.shape[0], source=str(path)
        )
        names = self.session.body.spec.joint_names
        unknown = [joint for _, joint in raw.values() if joint is not None and joint not in names]
        if unknown:
            raise UnknownNameError(f"parent joint in {path}", unknown[0], valid=names)
        self.session.set_frame_set(
            {
                key: FrameEntry(vertex, None if joint is None else names.index(joint))
                for key, (vertex, joint) in raw.items()
            }
        )
        self.refresh()

    def export(self) -> str:
        """Render the frame set as a ``contact_frames={...}`` argument for ``bh.load`` and show it."""
        self.snippet = persistence.frame_config_snippet(
            contact_frames(self.session.frame_set, self.session.body.spec.joint_names)
        )
        self.export_box.content = f"```python\n{self.snippet}\n```"
        return self.snippet
