"""Frames editor: click to add named frames, then save or export as config.

Wires the Frames tab to the session: it loads config / saved frame sets, adds a
frame at the vertex nearest a click, sets each frame's parent joint (argmax
default or a per-frame override), renames or deletes frames, saves the set, and
exports the paste-ready ``frame_names`` / ``frame_vertex_ids`` snippet.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import viser

from . import persistence, picking
from .state import FrameEntry, StudioSession

_ARGMAX_PREFIX = "argmax ("


class FramesEditor:
    """Frames-tab controller for authoring named vertex frames."""

    def __init__(
        self,
        server: viser.ViserServer,
        session: StudioSession,
        paths: persistence.StudioPaths,
        status: Callable[[str], None],
        default_frames: Callable[[], dict[str, FrameEntry]],
        set_edit_overlay: Callable[[bool], None],
        set_active_frame: Callable[[str | None], None],
    ) -> None:
        self._server = server
        self._session = session
        self._paths = paths
        self._status = status
        self._default_frames = default_frames
        self._set_edit_overlay = set_edit_overlay
        self._set_active_frame = set_active_frame
        self._editing = False
        self._highlighted: str | None = None
        self._suppress_joint = False
        self._click = None
        self.on_edit_start: Callable[[], None] = lambda: None

    def build(self) -> None:
        """Populate the Frames tab GUI for the active session."""
        gui = self._server.gui
        self._source = gui.add_dropdown("Frame set", self._source_options(), initial_value="config")
        self._source.on_update(lambda _event: self._load_source())
        self._edit = gui.add_checkbox("Edit mode", initial_value=False)
        self._edit.on_update(lambda _event: self._toggle_edit())
        self._selected = gui.add_dropdown("Frame", self._frame_names())
        self._selected.on_update(lambda _event: self._on_selected_change())
        gui.add_button("Deselect frame").on_click(lambda _event: self.clear_active())
        self._parent_joint = gui.add_dropdown("Parent joint", ("(no frame selected)",))
        self._parent_joint.on_update(lambda _event: self._on_parent_joint_change())
        self._rename = gui.add_text("Rename to", initial_value="")
        gui.add_button("Rename").on_click(lambda _event: self._do_rename())
        gui.add_button("Delete").on_click(lambda _event: self._do_delete())
        self._save_name = gui.add_text("Save as", initial_value="frames")
        gui.add_button("Save frames").on_click(lambda _event: self._save())
        gui.add_button("Export as config").on_click(lambda _event: self._export())
        self._config = gui.add_markdown("")
        self._sync_parent_joint()

    # -- sources -------------------------------------------------------------

    def _source_options(self) -> tuple[str, ...]:
        options = ["config"]
        options.extend(persistence.list_saved(self._paths.frame_dir(self._session.model_key)))
        return tuple(dict.fromkeys(options))

    def _load_source(self) -> None:
        choice = self._source.value
        try:
            if choice == "config":
                self._session.set_frame_set(self._default_frames())
            else:
                path = self._paths.frame_dir(self._session.model_key) / f"{choice}.json"
                raw = persistence.load_frame_set(path, self._session.body.structure.num_vertices)
                self._session.set_frame_set(self._resolve_loaded(raw))
        except (ValueError, OSError) as error:
            self._status(f"frame load failed: {error}")
            return
        self._sync_selected()

    def _resolve_loaded(self, raw: dict[str, tuple[int, str | None]]) -> dict[str, FrameEntry]:
        """Resolve loaded parent-joint names to native indices for this body.

        An unknown joint name falls back to the argmax default (``None``) and is
        reported in the status bar.
        """
        index = {name: position for position, name in enumerate(self._session.body.structure.joint_names)}
        frames: dict[str, FrameEntry] = {}
        unknown: list[str] = []
        for name, (vertex, joint_name) in raw.items():
            joint = None
            if joint_name is not None:
                if joint_name in index:
                    joint = index[joint_name]
                else:
                    unknown.append(joint_name)
            frames[name] = FrameEntry(vertex, joint)
        if unknown:
            self._status(f"unknown parent joint(s) {', '.join(sorted(set(unknown)))}: using argmax")
        return frames

    # -- editing -------------------------------------------------------------

    def _toggle_edit(self) -> None:
        if self._edit.value:
            self.on_edit_start()
            self._editing = True
            self._set_edit_overlay(True)
            self._server.scene.remove_click_callback("all")
            self._click = self._server.scene.on_click()(self._on_click)
            self._status("frame edit on: click the mesh to add a frame")
        else:
            self.exit_edit()

    def exit_edit(self) -> None:
        """Leave edit mode and drop the click callback."""
        self._editing = False
        self._edit.value = False
        if self._click is not None:
            self._server.scene.remove_click_callback(self._click)
            self._click = None
        self._set_edit_overlay(False)

    def _on_click(self, event: viser.SceneClickEvent) -> None:
        if not self._editing:
            return
        origin = torch.tensor(event.ray_origin, dtype=torch.float32)
        direction = torch.tensor(event.ray_direction, dtype=torch.float32)
        hit = picking.ray_hit(origin, direction, self._session.vertices, self._session.body.structure.faces)
        if hit is None:
            return
        vertex_id = int(((self._session.vertices - hit[0]) ** 2).sum(-1).argmin())
        frames = dict(self._session.frame_set)
        name = self._unique_name(frames)
        frames[name] = FrameEntry(vertex_id, None)
        self._session.set_frame_set(frames)
        self._selected.options = self._frame_names()
        self._selected.value = name
        self._set_highlight(name)
        self._status(f"added {name} at vertex {vertex_id}")

    # -- selection (click sphere or dropdown) --------------------------------

    def select_frame(self, name: str) -> None:
        """Activate a frame from a viewport sphere click; a repeat click clears."""
        if name not in self._session.frame_set:
            return
        if name == self._highlighted:
            self.clear_active()
            return
        self._selected.value = name
        self._set_highlight(name)

    def clear_active(self) -> None:
        """Clear the active (highlighted) frame; the dropdown value is kept."""
        self._set_highlight(None)

    def _set_highlight(self, name: str | None) -> None:
        self._highlighted = name
        self._set_active_frame(name)
        self._sync_parent_joint()

    def _on_selected_change(self) -> None:
        self._set_highlight(self._active_name())

    # -- parent joint --------------------------------------------------------

    def _sync_parent_joint(self) -> None:
        """Populate the Parent-joint dropdown for the active frame."""
        name = self._highlighted
        joint_names = tuple(self._session.body.structure.joint_names)
        self._suppress_joint = True
        try:
            if name is None or name not in self._session.frame_set:
                self._parent_joint.options = ("(no frame selected)",)
                self._parent_joint.value = "(no frame selected)"
                self._parent_joint.disabled = True
                return
            entry = self._session.frame_set[name]
            argmax_index = int(self._session.vertex_native_joints[entry.vertex])
            argmax_label = f"{_ARGMAX_PREFIX}{joint_names[argmax_index]})"
            self._parent_joint.options = (argmax_label, *joint_names)
            self._parent_joint.value = argmax_label if entry.joint is None else joint_names[entry.joint]
            self._parent_joint.disabled = False
        finally:
            self._suppress_joint = False

    def _on_parent_joint_change(self) -> None:
        if self._suppress_joint:
            return
        name = self._highlighted
        if name is None or name not in self._session.frame_set:
            return
        value = self._parent_joint.value
        if value.startswith(_ARGMAX_PREFIX):
            joint: int | None = None
        else:
            joint_names = list(self._session.body.structure.joint_names)
            joint = joint_names.index(value) if value in joint_names else None
        self._session.set_frame_joint(name, joint)

    def _unique_name(self, frames: dict[str, int]) -> str:
        index = len(frames)
        while f"frame_{index}" in frames:
            index += 1
        return f"frame_{index}"

    # -- rename / delete -----------------------------------------------------

    def _do_rename(self) -> None:
        old = self._selected.value
        new = self._rename.value.strip()
        if not old or not new or new in self._session.frame_set:
            self._status("rename needs a selected frame and an unused new name")
            return
        frames = {new if name == old else name: entry for name, entry in self._session.frame_set.items()}
        self._session.set_frame_set(frames)
        self._sync_selected()

    def _do_delete(self) -> None:
        name = self._selected.value
        if name not in self._session.frame_set:
            return
        frames = {key: entry for key, entry in self._session.frame_set.items() if key != name}
        self._session.set_frame_set(frames)
        self._sync_selected()

    # -- persistence ---------------------------------------------------------

    def _save(self) -> None:
        name = self._save_name.value.strip()
        if not name:
            self._status("provide a name to save frames")
            return
        joint_names = self._session.body.structure.joint_names
        records = {
            frame: (entry.vertex, None if entry.joint is None else joint_names[entry.joint])
            for frame, entry in self._session.frame_set.items()
        }
        path = self._paths.frame_dir(self._session.model_key) / f"{name}.json"
        persistence.save_frame_set(path, records)
        self._source.options = self._source_options()
        self._status(f"saved frames to {path}")

    def _export(self) -> None:
        vertices = {name: entry.vertex for name, entry in self._session.frame_set.items()}
        snippet = persistence.frame_config_snippet(vertices)
        self._config.content = f"```python\n{snippet}\n```"

    # -- helpers -------------------------------------------------------------

    def refresh(self) -> None:
        """Repopulate dropdowns after a body reload."""
        self._source.options = self._source_options()
        self._source.value = "config"
        self._sync_selected()
        self._config.content = ""
        self.exit_edit()

    def _frame_names(self) -> tuple[str, ...]:
        return tuple(self._session.frame_set) or ("",)

    def _active_name(self) -> str | None:
        value = self._selected.value
        return value if value in self._session.frame_set else None

    def _sync_selected(self) -> None:
        self._selected.options = self._frame_names()
        if self._selected.value not in self._session.frame_set and self._session.frame_set:
            self._selected.value = next(iter(self._session.frame_set))
        self._set_highlight(self._active_name())


__all__ = ["FramesEditor"]
