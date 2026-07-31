"""Frames layer: green spheres at each named frame's anchor vertex.

Each frame is a vertex id, so the spheres follow pose and shape by reading the
posed vertex position.  The active frame (just added, or selected in the Frames
tab) is drawn orange, the rest green.  When both this layer and the skeleton are
visible, a thin connector cylinder links each frame to its associated native
joint (the frame vertex's strongest-weighted skin joint).
"""

from __future__ import annotations

from collections.abc import Callable

import viser

from ..state import StudioSession
from .base import CONNECTOR_COLOR, FRAME_ACTIVE_COLOR, FRAME_COLOR, bone_transform

_ROOT = "/frames"


class FramesLayer:
    """Green (or orange, if active) spheres marking each named frame vertex."""

    name = "frames"

    def __init__(self, on_select: Callable[[str], None] | None = None, radius: float = 0.012) -> None:
        self._on_select = on_select or (lambda _name: None)
        self._radius = radius
        self._visible = False
        self._skeleton_visible = True
        self._dirty = False
        self._active: str | None = None
        self._server: viser.ViserServer | None = None
        self._session: StudioSession | None = None
        self._nodes: list[tuple[object, int, str]] = []
        self._connectors: list[tuple[object, int, int]] = []

    def build(self, server: viser.ViserServer, session: StudioSession) -> None:
        """Bind the server and session, then create the frame spheres."""
        self._server = server
        self._session = session
        self._dirty = False
        self._rebuild()

    def update(self, session: StudioSession, kind: str) -> None:
        """Reposition on pose or shape; rebuild when the frame set changes."""
        self._session = session
        if kind not in ("frames", "pose", "shape"):
            return
        if not self._visible:
            self._dirty = True
            return
        if kind == "frames":
            self._rebuild()
        else:
            self._place()

    def set_visible(self, visible: bool) -> None:
        """Show or hide every frame sphere and connector, refreshing if behind."""
        self._visible = visible
        for handle, _, _ in self._nodes:
            handle.visible = visible
        self._apply_connector_visibility()
        if visible and self._dirty:
            self._dirty = False
            self._rebuild()

    def set_skeleton_visible(self, visible: bool) -> None:
        """Track skeleton visibility; connectors show only when both are on."""
        self._skeleton_visible = visible
        self._apply_connector_visibility()

    def set_active(self, name: str | None) -> None:
        """Mark one frame as active (orange); repaint if it changed."""
        if name == self._active:
            return
        self._active = name
        for handle, _, frame_name in self._nodes:
            handle.color = FRAME_ACTIVE_COLOR if frame_name == self._active else FRAME_COLOR

    def set_radius(self, radius: float) -> None:
        """Set the sphere radius (and proportional connector radius)."""
        self._radius = radius
        for handle, _, _ in self._nodes:
            handle.radius = radius
        for handle, _, _ in self._connectors:
            handle.radius = self._connector_radius

    def clear(self) -> None:
        """Remove every frame sphere and connector from the scene."""
        self._clear()

    @property
    def _connector_radius(self) -> float:
        return 0.3 * self._radius

    def _clear(self) -> None:
        for handle, _, _ in self._nodes:
            handle.remove()
        self._nodes = []
        self._clear_connectors()

    def _clear_connectors(self) -> None:
        for handle, _, _ in self._connectors:
            handle.remove()
        self._connectors = []

    def _rebuild(self) -> None:
        if self._server is None or self._session is None:
            return
        self._clear()
        vertices = self._session.vertices
        for name, vertex_id in self._session.frame_set.items():
            color = FRAME_ACTIVE_COLOR if name == self._active else FRAME_COLOR
            handle = self._server.scene.add_icosphere(
                f"{_ROOT}/{name}",
                radius=self._radius,
                color=color,
                position=vertices[vertex_id].detach().cpu().numpy(),
                visible=self._visible,
            )
            handle.on_click(self._make_click(name))
            self._nodes.append((handle, vertex_id, name))
        self._rebuild_connectors()

    def _make_click(self, name: str) -> Callable[[object], None]:
        def _handler(_event: object) -> None:
            self._on_select(name)

        return _handler

    def _rebuild_connectors(self) -> None:
        self._clear_connectors()
        session = self._session
        vertices = session.vertices
        poses = session.native_poses
        joints = session.vertex_native_joints
        connectors_visible = self._visible and self._skeleton_visible
        for index, (_handle, vertex_id, _name) in enumerate(self._nodes):
            native_joint = int(joints[vertex_id])
            position, orientation, height = bone_transform(vertices[vertex_id], poses[native_joint, :3])
            if height <= 0.0:
                continue
            handle = self._server.scene.add_cylinder(
                f"{_ROOT}/connector/{index}",
                radius=self._connector_radius,
                height=height,
                color=CONNECTOR_COLOR,
                position=position,
                wxyz=orientation,
                visible=connectors_visible,
            )
            self._connectors.append((handle, vertex_id, native_joint))

    def _place(self) -> None:
        vertices = self._session.vertices
        poses = self._session.native_poses
        for handle, vertex_id, _name in self._nodes:
            handle.position = vertices[vertex_id].detach().cpu().numpy()
        for handle, vertex_id, native_joint in self._connectors:
            position, orientation, _ = bone_transform(vertices[vertex_id], poses[native_joint, :3])
            handle.position = position
            handle.wxyz = orientation

    def _apply_connector_visibility(self) -> None:
        connectors_visible = self._visible and self._skeleton_visible
        for handle, _, _ in self._connectors:
            handle.visible = connectors_visible


__all__ = ["FramesLayer"]
