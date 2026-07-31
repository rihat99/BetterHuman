"""Skeleton layer: clickable red joint spheres and white-gray bone cylinders.

Each joint sphere is scaled by its local bone length so tiny finger and eye
joints render small while limb joints stay large; the Layers "Joint radius"
slider is a global multiplier on those per-joint radii.  Joints with zero total
skin weight that are also non-poseable (MHR's world root and cosmetic
teeth/tongue/eye/null joints) are not drawn.  Bones connect each native joint to
its parent and are skipped when either endpoint is hidden.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import viser

from ..state import StudioSession
from .base import BONE_COLOR, JOINT_COLOR, SELECTED_JOINT_COLOR, bone_transform


_ROOT = "/skeleton"

#: Finger and toe joints get a single small radius (bones vary in length, but the
#: owner wants them uniformly small, not gradually decreasing).
_FINGER_RADIUS_FACTOR = 0.30
_FINGER_KEYWORDS = ("thumb", "index", "middle", "ring", "pinky", "toe")


def _is_finger_or_toe(name: str) -> bool:
    """Return whether a native joint name denotes a finger or toe joint."""
    lowered = name.lower()
    return any(keyword in lowered for keyword in _FINGER_KEYWORDS)


def _joint_radii(positions: torch.Tensor, parents: torch.Tensor, base: float) -> list[float]:
    """Return per-joint sphere radii from local bone lengths.

    ``radius_j = clamp(0.4 * d_j, 0.25 * base, 1.0 * base)`` where ``d_j`` is the
    distance to the parent joint, falling back to the smallest distance to a
    child (for the root) and finally to ``base``.  Finger/toe joints are handled
    separately by :meth:`SkeletonLayer._recompute_radii`.
    """
    num_joints = positions.shape[0]
    child_min = [float("inf")] * num_joints
    for child in range(num_joints):
        parent = int(parents[child])
        if 0 <= parent < num_joints:
            distance = float((positions[child] - positions[parent]).norm())
            child_min[parent] = min(child_min[parent], distance)
    radii: list[float] = []
    for joint in range(num_joints):
        parent = int(parents[joint])
        if 0 <= parent < num_joints:
            distance = float((positions[joint] - positions[parent]).norm())
        elif child_min[joint] != float("inf"):
            distance = child_min[joint]
        else:
            distance = base
        radii.append(min(max(0.4 * distance, 0.25 * base), 1.0 * base))
    return radii


def _hidden_joints(session: StudioSession) -> set[int]:
    """Return native joints with zero skin weight that are also non-poseable.

    This hides MHR's disconnected world root and its weightless cosmetic joints
    (teeth, tongue, eyes, nulls) while keeping every poseable joint clickable,
    even the weightless structural ones (hips/arms whose skin rides on twist
    procs) that the user still needs to select and rotate.
    """
    weights = session.body.values.skinning_weight_matrix
    zero_weight = torch.nonzero(weights.sum(0) < 1e-9).flatten().tolist()
    poseable = {handle.native_index for handle in session.handles}
    return {int(joint) for joint in zero_weight if int(joint) not in poseable}


class SkeletonLayer:
    """Red joint spheres plus white-gray bone cylinders for the native skeleton."""

    name = "skeleton"

    def __init__(self, on_select: Callable[[int], None], base_radius: float = 0.015) -> None:
        self._on_select = on_select
        self._base_radius = base_radius
        self._multiplier = 1.0
        self._visible = True
        self._server: viser.ViserServer | None = None
        self._session: StudioSession | None = None
        self._dirty = False
        self._selected: int | None = None
        self._radii: list[float] = []
        self._hidden: set[int] = set()
        self._spheres: list[tuple[object, int]] = []
        self._bones: list[tuple[object, int, int]] = []

    @property
    def base_radius(self) -> float:
        """The unscaled base radius the per-joint clamp is expressed against."""
        return self._base_radius

    def set_base_radius(self, base_radius: float) -> None:
        """Set the base radius (e.g. after a model switch) and resize if built."""
        self._base_radius = base_radius
        if self._spheres:
            self._recompute_radii()
            self._resize_spheres()

    def build(self, server: viser.ViserServer, session: StudioSession) -> None:
        """Create one sphere per visible joint and one cylinder per visible bone."""
        self._server = server
        self._session = session
        self._dirty = False
        self._selected = None
        self._hidden = _hidden_joints(session)
        self._recompute_radii()
        self._add_spheres()
        self._rebuild_bones()

    def set_selected(self, native_index: int | None) -> None:
        """Highlight one joint sphere gold; restore the rest to red."""
        self._selected = native_index
        for sphere, index in self._spheres:
            sphere.color = SELECTED_JOINT_COLOR if index == native_index else JOINT_COLOR

    def update(self, session: StudioSession, kind: str) -> None:
        """Reposition on pose; recompute radii and rebuild bones on shape."""
        self._session = session
        if kind not in ("pose", "shape"):
            return
        if not self._visible:
            self._dirty = True
            return
        if kind == "shape":
            self._recompute_radii()
            self._resize_spheres()
        self._place_spheres()
        if kind == "shape":
            self._rebuild_bones()
        else:
            self._place_bones()

    def set_visible(self, visible: bool) -> None:
        """Show or hide the whole skeleton, refreshing if it fell behind."""
        self._visible = visible
        for sphere, _ in self._spheres:
            sphere.visible = visible
        for handle, _, _ in self._bones:
            handle.visible = visible
        if visible and self._dirty:
            self._dirty = False
            self._recompute_radii()
            self._resize_spheres()
            self._place_spheres()
            self._rebuild_bones()

    def clear(self) -> None:
        """Remove all joint spheres and bone cylinders from the scene."""
        for sphere, _ in self._spheres:
            sphere.remove()
        self._spheres = []
        self._clear_bones()

    def set_radius_multiplier(self, multiplier: float) -> None:
        """Scale every per-joint radius by a global multiplier."""
        self._multiplier = multiplier
        self._resize_spheres()

    def _recompute_radii(self) -> None:
        session = self._session
        positions = session.native_poses[:, :3]
        radii = _joint_radii(positions, session.body.structure.parents, self._base_radius)
        finger_radius = _FINGER_RADIUS_FACTOR * self._base_radius
        for index, name in enumerate(session.body.structure.joint_names):
            if _is_finger_or_toe(name):
                radii[index] = finger_radius
        self._radii = radii

    def _sphere_radius(self, native_index: int) -> float:
        return self._multiplier * self._radii[native_index]

    def _bone_radius(self, parent: int, child: int) -> float:
        return 0.4 * self._multiplier * min(self._radii[parent], self._radii[child])

    def _add_spheres(self) -> None:
        session = self._session
        poses = session.native_poses
        self._spheres = []
        for native_index in range(poses.shape[0]):
            if native_index in self._hidden:
                continue
            color = SELECTED_JOINT_COLOR if native_index == self._selected else JOINT_COLOR
            handle = self._server.scene.add_icosphere(
                f"{_ROOT}/joint/{native_index}",
                radius=self._sphere_radius(native_index),
                color=color,
                position=poses[native_index, :3].detach().cpu().numpy(),
                visible=self._visible,
            )
            handle.on_click(self._make_click(native_index))
            self._spheres.append((handle, native_index))

    def _make_click(self, native_index: int) -> Callable[[object], None]:
        def _handler(_event: object) -> None:
            self._on_select(native_index)

        return _handler

    def _resize_spheres(self) -> None:
        for sphere, native_index in self._spheres:
            sphere.radius = self._sphere_radius(native_index)
        for handle, parent, child in self._bones:
            handle.radius = self._bone_radius(parent, child)

    def _place_spheres(self) -> None:
        poses = self._session.native_poses
        for sphere, native_index in self._spheres:
            sphere.position = poses[native_index, :3].detach().cpu().numpy()

    def _clear_bones(self) -> None:
        for handle, _, _ in self._bones:
            handle.remove()
        self._bones = []

    def _rebuild_bones(self) -> None:
        self._clear_bones()
        session = self._session
        poses = session.native_poses
        parents = session.body.structure.parents
        for child, parent_raw in enumerate(parents):
            parent = int(parent_raw)
            if parent < 0 or parent >= poses.shape[0]:
                continue
            if child in self._hidden or parent in self._hidden:
                continue
            position, orientation, height = bone_transform(poses[parent, :3], poses[child, :3])
            if height <= 0.0:
                continue
            handle = self._server.scene.add_cylinder(
                f"{_ROOT}/bone/{child}",
                radius=self._bone_radius(parent, child),
                height=height,
                color=BONE_COLOR,
                position=position,
                wxyz=orientation,
                visible=self._visible,
            )
            self._bones.append((handle, parent, child))

    def _place_bones(self) -> None:
        poses = self._session.native_poses
        for handle, parent, child in self._bones:
            position, orientation, _ = bone_transform(poses[parent, :3], poses[child, :3])
            handle.position = position
            handle.wxyz = orientation


__all__ = ["SkeletonLayer"]
