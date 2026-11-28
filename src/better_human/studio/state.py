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
from collections.abc import Callable, Mapping, Sequence
from typing import NamedTuple, final

import torch
from better_robot.exceptions import ArgumentError

import better_human as bh
from better_human.mass import PartTopology, segment_vertices

from .posing import JointHandle, apply_local_rotation, joint_handles

#: Change kinds broadcast to subscribers.
ChangeKind = str  # "pose" | "shape" | "body" | "segmentation" | "frames"


@final
class FrameEntry(NamedTuple):
    """A named frame: an anchor vertex and an optional parent-joint override.

    ``joint`` is a native joint index that overrides the default parent (the
    one ``HumanModel.from_spec(contact_frames=...)`` picks); ``None`` selects that default.
    """

    vertex: int
    joint: int | None = None


def contact_frames(
    frame_set: Mapping[str, FrameEntry], joint_names: Sequence[str]
) -> dict[str, tuple[int, str | None]]:
    """``frame_set`` as ``HumanModel.from_spec(contact_frames=...)`` takes it: name -> ``(vertex, joint name | None)``."""
    return {
        name: (entry.vertex, None if entry.joint is None else joint_names[entry.joint])
        for name, entry in frame_set.items()
    }


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

    :meth:`better_human.mass.PartTopology.from_part_ids` requires every part index
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
        face_parts = PartTopology.face_parts(faces, ids)
        present = sorted({int(part) for part in face_parts.unique().tolist() if 0 <= part < num_parts})
        if len(present) == num_parts:
            return ids, tuple(names)
        compacted = torch.full_like(ids, -1)
        for new_index, original in enumerate(present):
            compacted[ids == original] = new_index
        ids = compacted
        names = [names[original] for original in present]


@final
class StudioSession:
    """The single mutable hub the studio front end reads and mutates."""

    def __init__(self, body: bh.HumanModel, model_key: str, frame_set: dict[str, FrameEntry] | None = None) -> None:
        # The one studio lock: every GUI callback, the gizmo worker and each mutation run under it (re-entrant: nested calls).
        self.lock = threading.RLock()
        self._subscribers: list[Callable[[ChangeKind], None]] = []
        self.load_body(body, model_key, frame_set, notify=False)

    # -- lifecycle -----------------------------------------------------------

    def load_body(
        self,
        body: bh.HumanModel,
        model_key: str,
        frame_set: dict[str, FrameEntry] | None = None,
        *,
        notify: bool = True,
    ) -> None:
        """Swap in a freshly loaded body and reset pose, shape, and selections."""
        with self.lock:
            self.body = body
            self.model_key = model_key
            self.shape = {
                name: torch.zeros(width, dtype=body.spec.dtype, device=body.spec.device)
                for name, width in body.spec.identity_groups.items()
            }
            identity = body.bake(**self.shape)
            self.handles = joint_handles(identity)
            # The strongest-weighted native skin joint of each vertex, (V,).
            self.vertex_native_joints = segment_vertices(body.spec.skin_weights, body.spec.skin_indices)
            self._reset_default_segmentation()
            self.frame_set = dict(frame_set) if frame_set else {}
            self._recompute(identity, identity.model.q_neutral.clone())
            if notify:
                self._notify("body")

    def _reset_default_segmentation(self) -> None:
        """Install the argmax segmentation with joint-derived part names."""
        part_ids = self.body.spec.parts.ids.clone().to(torch.long)
        num_parts = int(part_ids.max()) + 1
        representatives = _part_representatives(part_ids, num_parts, self.vertex_native_joints)
        joint_names = self.body.spec.joint_names
        self.part_ids = part_ids
        self.part_names = tuple(joint_names[rep] for rep in representatives)

    def _recompute(self, identity: bh.Identity, q: torch.Tensor) -> None:
        """Install ``identity`` posed at ``q (nq,)`` with its state, vertices ``(V, 3)``, native world joint poses
        ``(J, 7)`` and frame poses ``(F, 7)``.

        Everything is computed before anything is assigned, so an edit the body refuses leaves the session unchanged.
        """
        state = identity.forward(q)
        vertices, joint_poses = state.vertices, state.joint_poses
        frame_poses = state.contact_frame_poses(contact_frames(self.frame_set, self.body.spec.joint_names))
        self.identity: bh.Identity = identity
        self.q = q
        self.state: bh.HumanState = state
        self.vertices, self.joint_poses, self.frame_poses = vertices, joint_poses, frame_poses

    # -- pose ----------------------------------------------------------------

    def set_q(self, q: torch.Tensor) -> None:
        """Set the configuration, recompute caches, and notify a pose change."""
        with self.lock:
            self._recompute(self.identity, q.to(dtype=self.q.dtype, device=self.q.device))
            self._notify("pose")

    def rotate_joint(self, handle: JointHandle, world_old: torch.Tensor, world_new: torch.Tensor) -> None:
        """Apply a gizmo world-rotation edit to one joint's local ``q`` slots."""
        with self.lock:
            self.set_q(apply_local_rotation(self.q, handle, world_old, world_new))

    def move_root(self, handle: JointHandle, world_old: torch.Tensor, world_new: torch.Tensor) -> None:
        """Move the free-flyer root by the gizmo's world delta in position and rotation.

        The reported root frame can sit at a fixed offset from the free flyer (MHR's pelvis height, SOMA's T-pose
        Hips orientation), so ``q`` is nudged by the delta rather than overwritten: the translation delta is added to
        the position, and :func:`apply_local_rotation` turns the quaternion.
        """
        if handle.kind != "free_flyer":
            raise ArgumentError(f"move_root requires a free_flyer handle, got {handle.kind!r}")
        with self.lock:
            q = apply_local_rotation(self.q, handle, world_old, world_new)
            q[handle.idx_q : handle.idx_q + 3] = q[handle.idx_q : handle.idx_q + 3] + (world_new[:3] - world_old[:3])
            self.set_q(q)

    def reset_pose(self) -> None:
        """Return the configuration to the model neutral pose."""
        self.set_q(self.identity.model.q_neutral.clone())

    # -- shape ---------------------------------------------------------------

    def set_shape(self, **groups: torch.Tensor) -> None:
        """Re-bake the identity with named identity groups ``(n,)`` and notify a shape change."""
        with self.lock:
            shape = {**self.shape, **groups}
            self._recompute(self.body.bake(**shape), self.q)
            self.shape = shape
            self._notify("shape")

    def shape_tensor(self, group: str, values: list[float]) -> torch.Tensor:
        """Build a validated tensor for a named shape group from slider values.

        :param group: shape-group name; its declared width bounds ``values``.
        :param values: one coefficient per group slider.
        :returns: a ``(width,)`` tensor matching the body dtype and device.
        """
        width = self.body.spec.identity_groups[group]
        if len(values) != width:
            raise ArgumentError(f"shape group {group!r} expects {width} values, got {len(values)}")
        return torch.tensor(values, dtype=self.body.spec.dtype, device=self.body.spec.device)

    @property
    def total_mass(self) -> float | None:
        """Return the baked identity's total mass in kilograms, or ``None`` when loaded without mass."""
        if self.body.part_topology is None:
            return None
        return float(self.identity.model.body_inertias[..., 0].sum())

    # -- segmentation and frames --------------------------------------------

    def set_segmentation(self, part_names: tuple[str, ...], part_ids: torch.Tensor) -> None:
        """Install a new active segmentation and notify a segmentation change."""
        with self.lock:
            self.part_names = part_names
            self.part_ids = part_ids.to(dtype=torch.long, device=self.q.device)
            self._notify("segmentation")

    def reset_segmentation(self) -> None:
        """Restore the argmax segmentation and notify a segmentation change."""
        with self.lock:
            self._reset_default_segmentation()
            self._notify("segmentation")

    def set_frame_set(self, frame_set: dict[str, FrameEntry]) -> None:
        """Install a new active frame set and notify a frames change.

        A frame set the body refuses (MHR and SOMA take none) raises before anything changes.
        """
        with self.lock:
            poses = self.state.contact_frame_poses(contact_frames(frame_set, self.body.spec.joint_names))
            self.frame_set = dict(frame_set)
            self.frame_poses = poses
            self._notify("frames")

    def set_frame_joint(self, name: str, joint: int | None) -> None:
        """Override (``None`` clears) one frame's parent native joint index."""
        with self.lock:
            entry = self.frame_set.get(name)
            if entry is None:
                return
            self.set_frame_set({**self.frame_set, name: FrameEntry(entry.vertex, joint)})

    # -- subscription --------------------------------------------------------

    def subscribe(self, callback: Callable[[ChangeKind], None]) -> None:
        """Register a callback invoked with the change kind after every update."""
        self._subscribers.append(callback)

    def _notify(self, kind: ChangeKind) -> None:
        for callback in self._subscribers:
            callback(kind)
