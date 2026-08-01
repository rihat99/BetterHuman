"""Mutable session hub shared by every HumanStudio layer and editor.

:class:`StudioSession` owns the current body, its configuration ``q``, the
active segmentation and frame set, and the derived FK caches (world joint
poses, posed vertices).  It recomputes those caches whenever pose or shape
changes and notifies subscribers with a change ``kind`` so layers can update
incrementally.  It never imports viser: it is the pure-state boundary between
the viser front end (``app.py``) and the deformation core.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import NamedTuple

import torch
from better_robot.lie import so3

from better_human.core import ParametricBody
from better_human.core.mass import _majority_face_parts, segment_vertices

from .posing import JointHandle, apply_local_rotation, joint_handles

#: Change kinds broadcast to subscribers.
ChangeKind = str  # "pose" | "shape" | "body" | "segmentation" | "frames"


class FrameEntry(NamedTuple):
    """A named frame: an anchor vertex and an optional parent-joint override.

    ``joint`` is a native joint index that overrides the default parent (the
    frame vertex's strongest-weighted skinning joint); ``None`` selects that
    argmax default.  Resolve it with :meth:`StudioSession.resolved_joint`.
    """

    vertex: int
    joint: int | None = None


def _vertex_native_joints(body: ParametricBody) -> torch.Tensor:
    """Return the strongest-weighted native skin joint per vertex ``(V,)``."""
    return segment_vertices(body.values.skinning_weights, body.structure.skinning_indices)


def _part_representatives(part_ids: torch.Tensor, num_parts: int, vertex_native_joints: torch.Tensor) -> list[int]:
    """Return the dominant native joint index for each part label."""
    representatives: list[int] = []
    for part in range(num_parts):
        members = vertex_native_joints[part_ids == part]
        representatives.append(0 if members.numel() == 0 else int(members.mode().values))
    return representatives


def compact_segmentation(
    faces: torch.Tensor,
    part_ids: torch.Tensor,
    part_names: tuple[str, ...],
) -> tuple[torch.Tensor, tuple[str, ...]]:
    """Drop parts that own no majority face so a closed topology can be built.

    :func:`better_human.core.mass.build_part_topology` requires every part index
    to win the majority of at least one face. A user may add a part before
    painting it, and remapping dropped parts to ``-1`` can flip a tiebreaker
    face's majority, so this iterates to a fixed point: it keeps only the parts
    that are the majority of some face under the current labeling, compacts their
    indices to ``0..K-1`` (everything else ``-1``), and repeats until stable.

    :returns: ``(compacted part_ids (V,), kept part names)`` aligned by index.
    """
    ids = part_ids.clone()
    names = list(part_names)
    while True:
        num_parts = len(names)
        face_parts = _majority_face_parts(faces, ids)
        present = sorted({int(part) for part in face_parts.unique().tolist() if 0 <= part < num_parts})
        if len(present) == num_parts:
            return ids, tuple(names)
        compacted = torch.full_like(ids, -1)
        for new_index, original in enumerate(present):
            compacted[ids == original] = new_index
        ids = compacted
        names = [names[original] for original in present]


class StudioSession:
    """The single mutable hub the studio front end reads and mutates."""

    def __init__(self, body: ParametricBody, model_key: str, frame_set: dict[str, int] | None = None) -> None:
        # Reentrant so a mutating method may call another (set_shape -> set_q)
        # while a single mutate -> recompute -> notify sequence stays atomic
        # against viser's fire-and-forget callback threads.
        self._lock = threading.RLock()
        self._subscribers: list[Callable[[ChangeKind], None]] = []
        self.load_body(body, model_key, frame_set, notify=False)

    # -- lifecycle -----------------------------------------------------------

    def load_body(
        self,
        body: ParametricBody,
        model_key: str,
        frame_set: dict[str, FrameEntry] | None = None,
        *,
        notify: bool = True,
    ) -> None:
        """Swap in a freshly loaded body and reset pose, shape, and selections."""
        with self._lock:
            self.body = body
            self.model_key = model_key
            self.handles = joint_handles(body)
            self.q = body.robot.q_neutral.clone()
            self.shape = {name: value.clone() for name, value in body.values.shape_parameters.items()}
            self.vertex_native_joints = _vertex_native_joints(body)
            self._reset_default_segmentation()
            self.frame_set = dict(frame_set) if frame_set else {}
            self._recompute()
            if notify:
                self._notify("body")

    def _reset_default_segmentation(self) -> None:
        """Install the argmax segmentation with joint-derived part names."""
        part_ids = self.body.structure.part_ids.clone()
        num_parts = int(part_ids.max()) + 1
        representatives = _part_representatives(part_ids, num_parts, self.vertex_native_joints)
        joint_names = self.body.structure.joint_names
        self.part_ids = part_ids
        self.part_names = tuple(joint_names[rep] for rep in representatives)

    def _recompute(self) -> None:
        """Refresh the FK data, posed vertices, and native world joint poses."""
        self.data = self.body.fk(self.q)
        self.vertices = self.body.vertices_from_data(self.data)
        self.native_poses = self.body.native_joint_poses(self.data)

    # -- pose ----------------------------------------------------------------

    def set_q(self, q: torch.Tensor) -> None:
        """Set the configuration, recompute caches, and notify a pose change."""
        with self._lock:
            self.q = q.to(dtype=self.q.dtype, device=self.q.device)
            self._recompute()
            self._notify("pose")

    def rotate_joint(self, handle: JointHandle, world_old: torch.Tensor, world_new: torch.Tensor) -> None:
        """Apply a gizmo world-rotation edit to one joint's local ``q`` slots."""
        with self._lock:
            self.set_q(apply_local_rotation(self.q, handle, world_old, world_new))

    def move_root(self, handle: JointHandle, world_old: torch.Tensor, world_new: torch.Tensor) -> None:
        """Move the free-flyer root by a world delta and set its world rotation.

        The free-flyer origin can carry a fixed offset (MHR's pelvis height), so
        the world position is nudged by the gizmo delta rather than overwritten,
        while the world rotation equals the free-flyer quaternion for both SMPL
        and MHR.
        """
        if handle.kind != "free_flyer":
            raise ValueError(f"move_root requires a free_flyer handle, got {handle.kind!r}")
        with self._lock:
            q = self.q.clone()
            idx = handle.idx_q
            q[idx : idx + 3] = q[idx : idx + 3] + (world_new[:3] - world_old[:3])
            q[idx + 3 : idx + 7] = so3.normalize(world_new[3:7])
            self.set_q(q)

    def reset_pose(self) -> None:
        """Return the configuration to the robot neutral pose."""
        self.set_q(self.body.robot.q_neutral.clone())

    # -- shape ---------------------------------------------------------------

    def set_shape(self, **groups: torch.Tensor) -> None:
        """Bake named shape groups into the body and notify a shape change."""
        with self._lock:
            self.shape.update({name: value for name, value in groups.items()})
            self.body = self.body.with_shape(**groups)
            self.vertex_native_joints = _vertex_native_joints(self.body)
            self._recompute()
            self._notify("shape")

    def shape_tensor(self, group: str, values: list[float]) -> torch.Tensor:
        """Build a validated tensor for a named shape group from slider values.

        :param group: shape-group name; its declared width bounds ``values``.
        :param values: one coefficient per group slider.
        :returns: a ``(width,)`` tensor matching the body dtype and device.
        """
        width = self.body.structure.parameter_groups[group]
        if len(values) != width:
            raise ValueError(f"shape group {group!r} expects {width} values, got {len(values)}")
        exemplar = self.body.values.v_template
        return torch.tensor(values, dtype=exemplar.dtype, device=exemplar.device)

    @property
    def total_mass(self) -> float | None:
        """Return the loaded body's total mass in kilograms, if mass is baked."""
        inertias = self.body.values.body_inertias
        if inertias is None:
            return None
        return float(inertias[..., 0].sum())

    # -- segmentation and frames --------------------------------------------

    def set_segmentation(self, part_names: tuple[str, ...], part_ids: torch.Tensor) -> None:
        """Install a new active segmentation and notify a segmentation change."""
        with self._lock:
            self.part_names = part_names
            self.part_ids = part_ids.to(dtype=torch.long, device=self.q.device)
            self._notify("segmentation")

    def reset_segmentation(self) -> None:
        """Restore the argmax segmentation and notify a segmentation change."""
        with self._lock:
            self._reset_default_segmentation()
            self._notify("segmentation")

    def set_frame_set(self, frame_set: dict[str, FrameEntry]) -> None:
        """Install a new active frame set and notify a frames change."""
        with self._lock:
            self.frame_set = dict(frame_set)
            self._notify("frames")

    def set_frame_joint(self, name: str, joint: int | None) -> None:
        """Override (``None`` clears) one frame's parent native joint index."""
        with self._lock:
            entry = self.frame_set.get(name)
            if entry is None:
                return
            self.frame_set[name] = FrameEntry(entry.vertex, joint)
            self._notify("frames")

    def resolved_joint(self, entry: FrameEntry) -> int:
        """Return a frame's parent joint: its override, else the argmax joint."""
        if entry.joint is not None:
            return entry.joint
        return int(self.vertex_native_joints[entry.vertex])

    # -- subscription --------------------------------------------------------

    def subscribe(self, callback: Callable[[ChangeKind], None]) -> None:
        """Register a callback invoked with the change kind after every update."""
        self._subscribers.append(callback)

    def _notify(self, kind: ChangeKind) -> None:
        for callback in self._subscribers:
            callback(kind)


__all__ = ["ChangeKind", "FrameEntry", "StudioSession"]
