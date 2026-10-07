"""Joint posing gizmo: a BetterRobot :class:`PoseGizmo` at the selected joint's world pose.

Non-root joints get a rotation-only handle; the free-flyer root gets a 6-DoF handle. Joints with no
pose coordinate (fixed MHR joints) get no handle; the status callback reports it instead.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from better_robot.viewer.tools import PoseGizmo

from .posing import JointHandle
from .state import ChangeKind, StudioSession


class GizmoController:
    """Owns at most one :class:`PoseGizmo`, bound to the selected native joint of ``session``."""

    def __init__(
        self,
        viewer: Any,
        session: StudioSession,
        *,
        on_status: Callable[[str], None] | None = None,
        scale: float = 0.15,
    ) -> None:
        self.viewer = viewer
        self.session = session
        self.on_status = on_status
        self.scale = scale
        self.joint: int | None = None
        self.gizmo: PoseGizmo | None = None
        session.subscribe(self._on_change)

    def handle_for(self, joint: int) -> JointHandle | None:
        """Return the pose handle of native joint ``joint``, or ``None`` if it has no degree of freedom."""
        return next((handle for handle in self.session.handles if handle.native_index == joint), None)

    def select(self, joint: int | None) -> None:
        """Bind the gizmo to native joint ``joint`` (``None`` clears it)."""
        self.clear()
        self.joint = joint
        if joint is None:
            return
        handle = self.handle_for(joint)
        name = self.session.body.joint_names[joint]
        if handle is None:
            self._status(f"joint {name!r} has no degree of freedom")
            return
        options: dict[str, Any] = {}
        if handle.kind != "free_flyer":
            options = {"disable_sliders": True, "disable_axes": True}
        self.gizmo = PoseGizmo(
            self.viewer,
            "joint_gizmo",
            self._world_pose(joint),
            on_change=lambda pose: self._drag_locked(handle, pose),
            scale=self.scale,
            **options,
        )
        self._status(f"posing {name!r} ({handle.kind})")

    def clear(self) -> None:
        """Remove the gizmo, if any."""
        if self.gizmo is not None:
            self.gizmo.remove()
        self.gizmo = None
        self.joint = None

    def _world_pose(self, joint: int) -> np.ndarray:
        return self.session.joint_poses[joint].detach().cpu().numpy().astype(np.float32)

    def _drag_locked(self, handle: JointHandle, pose: np.ndarray) -> None:
        with self.session.lock:  # drags arrive on BR's worker thread
            self._on_drag(handle, pose)

    def _on_drag(self, handle: JointHandle, pose: np.ndarray) -> None:
        world_old = self.session.joint_poses[handle.native_index]
        world_new = torch.as_tensor(pose, dtype=world_old.dtype, device=world_old.device)
        if handle.kind == "free_flyer":
            self.session.move_root(handle, world_old, world_new)
        else:
            self.session.rotate_joint(handle, world_old, world_new)

    def _on_change(self, kind: ChangeKind) -> None:
        if kind == "body":
            self.clear()
        elif kind in ("pose", "shape") and self.gizmo is not None and self.joint is not None:
            self.gizmo.pose = self._world_pose(self.joint)

    def _status(self, message: str) -> None:
        if self.on_status is not None:
            self.on_status(message)


__all__ = ["GizmoController"]
