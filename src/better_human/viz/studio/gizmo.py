"""Single-gizmo joint posing with latest-wins event coalescing.

viser dispatches transform-control ``on_update`` callbacks fire-and-forget on a
thread pool, so a fast drag queues dozens of full FK/LBS recomputes behind the
session lock and can starve the UI.  :class:`GizmoController` decouples the two:
the (async, in-order) viser callback only records the newest gizmo pose into a
single slot and wakes a dedicated worker thread; the worker processes the latest
state and silently drops stale intermediates.

Dropping intermediates is lossless for full-DOF joints because every edit is
applied as a delta against the *last processed* gizmo pose, and both kinds of
delta compose exactly: a world-frame right-multiplied SO(3) delta
(``rotate_joint``) and a translation delta (``move_root``) are associative, so
jumping straight from the last processed pose to the newest pose equals replaying
every skipped step (spherical, free-flyer, and full three-channel joints).  For
under-actuated MHR channel joints (missing Euler channels) the per-step
projection onto the reachable axes is nonlinear, so the final angle is mildly
drag-speed dependent (bounded, measured within ~1.5 deg); the end-of-drag
re-seat snaps the handle back onto the achieved pose so it never visibly drifts.

The gizmo is never re-seated during a drag (that fought the user and was the
"super buggy" root behaviour).  It is re-seated only when the pose changes from a
non-gizmo source (shape/reset/load, via :meth:`reseat_external`) or once a drag
settles (``phase == "end"``), where the handle snaps onto the *achieved* joint
pose so a channels-clamp projection can never leave it drifting.
"""

from __future__ import annotations

import threading
from collections.abc import Callable

import numpy as np
import torch
import viser

from .layers.base import wxyz
from .posing import JointHandle
from .state import StudioSession

#: Slightly larger than a bare joint so the handles poke out of the body mesh.
DEFAULT_GIZMO_SCALE = 0.25


def _event_world(target: viser.TransformControlsHandle) -> torch.Tensor:
    """Read a transform control's pose as a scalar-last ``(7,)`` world pose."""
    position = np.asarray(target.position, dtype=np.float64)
    w, x, y, z = np.asarray(target.wxyz, dtype=np.float64).tolist()
    return torch.tensor([position[0], position[1], position[2], x, y, z, w], dtype=torch.float32)


class GizmoController:
    """Own the single posing gizmo and coalesce its drag events onto ``q``."""

    def __init__(
        self,
        server: viser.ViserServer,
        session: StudioSession,
        status: Callable[[str], None] | None = None,
    ) -> None:
        self._server = server
        self._session = session
        self._status = status
        self._handle: JointHandle | None = None
        self._native_index: int | None = None
        self._gizmo: viser.TransformControlsHandle | None = None
        self._last_processed: torch.Tensor | None = None
        # Bumped on every attach/detach so in-flight worker work for a stale
        # selection is discarded instead of writing to the wrong joint.
        self._generation = 0
        self._applying = False
        self._pending: tuple[torch.Tensor, str] | None = None
        self._pending_gen = -1
        self._processing = False
        self._stop = False
        self._cond = threading.Condition()
        self._worker = threading.Thread(target=self._run, name="gizmo-coalesce", daemon=True)
        self._worker.start()

    # -- read-only state -----------------------------------------------------

    @property
    def gizmo(self) -> viser.TransformControlsHandle | None:
        """The live transform-control handle, or ``None`` when detached."""
        return self._gizmo

    @property
    def native_index(self) -> int | None:
        """Native joint index the gizmo is attached to, or ``None``."""
        return self._native_index

    @property
    def applying(self) -> bool:
        """``True`` while the worker is writing a gizmo edit into the session."""
        return self._applying

    # -- attach / detach -----------------------------------------------------

    def attach(self, handle: JointHandle, native_pose: torch.Tensor, *, scale: float = DEFAULT_GIZMO_SCALE) -> None:
        """Attach the gizmo to a joint at its world ``native_pose`` ``(7,)``."""
        self.detach()
        with self._cond:
            self._generation += 1
            self._handle = handle
            self._native_index = handle.native_index
            self._last_processed = native_pose.clone()
        is_root = handle.kind == "free_flyer"
        gizmo = self._server.scene.add_transform_controls(
            "/gizmo",
            scale=scale,
            position=native_pose[:3].detach().cpu().numpy(),
            wxyz=wxyz(native_pose[3:7]),
            disable_axes=not is_root,
            disable_sliders=not is_root,
        )
        gizmo.on_update(self._on_gizmo)
        with self._cond:
            self._gizmo = gizmo

    def detach(self) -> None:
        """Remove the gizmo and forget the current selection."""
        with self._cond:
            self._generation += 1
            self._handle = None
            self._native_index = None
            self._pending = None
            self._last_processed = None
            gizmo = self._gizmo
            self._gizmo = None
        if gizmo is not None:
            gizmo.remove()

    def reseat_external(self) -> None:
        """Re-seat the gizmo after a non-gizmo pose or shape change."""
        with self._cond:
            gizmo = self._gizmo
            native_index = self._native_index
            gen = self._generation
        if gizmo is None or native_index is None:
            return
        pose = self._session.native_poses[native_index]
        gizmo.position = pose[:3].detach().cpu().numpy()
        gizmo.wxyz = wxyz(pose[3:7])
        with self._cond:
            if gen == self._generation:
                self._last_processed = pose.clone()

    def shutdown(self) -> None:
        """Stop the worker thread; call once when tearing down the studio."""
        with self._cond:
            self._stop = True
            self._cond.notify()
        self._worker.join(timeout=1.0)

    # -- event intake --------------------------------------------------------

    async def _on_gizmo(self, event: viser.TransformControlsEvent) -> None:
        # Runs in-order on viser's event loop; only records + wakes the worker,
        # never blocks on the recompute.
        self.submit_pose(_event_world(event.target), event.phase)

    def submit_pose(self, world: torch.Tensor, phase: str) -> None:
        """Record the newest gizmo pose into the slot and wake the worker."""
        with self._cond:
            if self._handle is None:
                return
            self._pending = (world, phase)
            self._pending_gen = self._generation
            self._cond.notify()

    def wait_idle(self, timeout: float = 2.0) -> bool:
        """Block until the slot is drained and no edit is in flight (tests)."""
        with self._cond:
            deadline = threading.TIMEOUT_MAX if timeout is None else timeout
            return self._cond.wait_for(
                lambda: self._pending is None and not self._processing,
                timeout=deadline,
            )

    # -- worker --------------------------------------------------------------

    def _run(self) -> None:
        while True:
            with self._cond:
                while self._pending is None and not self._stop:
                    self._cond.wait()
                if self._stop:
                    return
                world, phase = self._pending
                gen = self._pending_gen
                self._pending = None
                self._processing = True
            try:
                self._apply(world, phase, gen)
            except Exception as error:  # noqa: BLE001 - never let a bad edit kill the worker
                # A single degenerate edit must not brick posing for the whole
                # session; drop it, surface a line, and keep serving.
                self._applying = False
                self._report(f"gizmo edit failed: {error}")
            finally:
                with self._cond:
                    self._processing = False
                    self._cond.notify_all()

    def _report(self, message: str) -> None:
        if self._status is not None:
            self._status(message)

    def _apply(self, world: torch.Tensor, phase: str, gen: int) -> None:
        with self._cond:
            if gen != self._generation or self._handle is None or self._last_processed is None:
                return
            handle = self._handle
            last = self._last_processed
        self._applying = True
        try:
            if handle.kind == "free_flyer":
                self._session.move_root(handle, last, world)
            else:
                self._session.rotate_joint(handle, last, world)
        finally:
            self._applying = False
        with self._cond:
            if gen != self._generation:
                return
            self._last_processed = world
        if phase == "end":
            self._settle(gen)

    def _settle(self, gen: int) -> None:
        with self._cond:
            if gen != self._generation:
                return
            gizmo = self._gizmo
            native_index = self._native_index
        if gizmo is None or native_index is None:
            return
        pose = self._session.native_poses[native_index]
        gizmo.position = pose[:3].detach().cpu().numpy()
        gizmo.wxyz = wxyz(pose[3:7])
        with self._cond:
            if gen == self._generation:
                self._last_processed = pose.clone()


__all__ = ["DEFAULT_GIZMO_SCALE", "GizmoController"]
