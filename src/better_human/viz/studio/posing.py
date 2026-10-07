"""Map transform-gizmo edits onto a BetterRobot configuration ``q``.

Pure torch plus :mod:`better_robot.lie.so3`; no viser import.  One
:class:`JointHandle` is built per poseable native skin joint, and
:func:`apply_local_rotation` turns a joint's world-pose edit into a safe local
``q`` update.

The MHR channel convention was verified empirically against
``HumanState.joint_poses``: each native rotation joint is a chain of revolute
hosts added in archive channel order ``z, y, x``
(``bodies/mhr/tree.py:88-98``), so its local rotation composes
``Rz(z) @ Ry(y) @ Rx(x)`` -- exactly the ``so3.from_euler`` /
``so3.to_euler`` extrinsic-XYZ convention.  Mimic joints carry no public
coordinate (``robot.nqs == 0`` and ``robot.mimic_source != joint_id``) and are
skipped; they follow their source host automatically through FK.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from better_robot.lie import so3

from better_human.core.body import Identity

_REVOLUTE_AXIS = {
    "revolute_rx": "x",
    "revolute_ry": "y",
    "revolute_rz": "z",
}
_EULER_INDEX = {"x": 0, "y": 1, "z": 2}


@dataclass(frozen=True)
class JointHandle:
    """One editable native joint mapped to its BetterRobot ``q`` slots."""

    #: Index into ``structure.joint_names``.
    native_index: int
    #: Native joint name.
    name: str
    #: Joint model family: ``"free_flyer"``, ``"spherical"`` or ``"channels"``.
    kind: str
    #: Start of the ``q`` slot for free-flyer/spherical; ``-1`` for channels.
    idx_q: int
    #: ``(axis, idx_q)`` per revolute host in application order (outermost
    #: ``z`` first, then ``y``, then ``x``); empty unless ``kind == "channels"``.
    channels: tuple[tuple[str, int], ...]


def joint_handles(identity: Identity) -> tuple[JointHandle, ...]:
    """Return one handle per poseable native skin joint of ``identity.model``.

    SMPL/SMPLX map native joint ``j`` to BetterRobot joint ``j + 1``: the root
    is a free-flyer, every other joint a spherical joint.  MHR walks the tree
    upward from ``body.native_joint_ids[j]`` to the parent native joint,
    collecting revolute host joints and skipping fixed and mimic joints; native
    joint 0 (the world anchor) and joints without any host are not returned.
    """
    body = identity.body
    robot = identity.model
    joint_names = body.joint_names
    native_leaves = [int(value) for value in body.native_joint_ids.tolist()]

    if native_leaves == list(range(1, len(joint_names) + 1)):
        handles = [JointHandle(0, joint_names[0], "free_flyer", robot.idx_qs[1], ())]
        for native_index in range(1, len(joint_names)):
            branch = robot.idx_qs[native_index + 1]
            handles.append(JointHandle(native_index, joint_names[native_index], "spherical", branch, ()))
        return tuple(handles)

    native_set = set(native_leaves)
    kind_codes = [joint.kind for joint in robot.joint_models]
    handles = []
    for native_index in range(1, len(native_leaves)):
        leaf = native_leaves[native_index]
        if native_index == 1:
            handles.append(JointHandle(native_index, joint_names[native_index], "free_flyer", robot.idx_qs[leaf], ()))
            continue
        hosts: list[tuple[str, int]] = []
        current = leaf
        while True:
            if robot.nqs[current] > 0 and kind_codes[current] in _REVOLUTE_AXIS:
                hosts.append((_REVOLUTE_AXIS[kind_codes[current]], robot.idx_qs[current]))
            parent = robot.parents[current]
            if parent in native_set or parent < 0:
                break
            current = parent
        if not hosts:
            continue
        # Walking up yields leaf-first (x, y, z); reverse to z, y, x order.
        channels = tuple(reversed(hosts))
        handles.append(JointHandle(native_index, joint_names[native_index], "channels", -1, channels))
    return tuple(handles)


def _local_delta(world_old: torch.Tensor, world_new: torch.Tensor) -> torch.Tensor:
    """Return the ``so3`` tangent taking ``world_old`` to ``world_new`` locally."""
    old_quat = so3.normalize(world_old[3:7])
    new_quat = so3.normalize(world_new[3:7])
    return so3.log(so3.compose(so3.inverse(old_quat), new_quat))


def apply_local_rotation(
    q: torch.Tensor,
    handle: JointHandle,
    world_old: torch.Tensor,
    world_new: torch.Tensor,
) -> torch.Tensor:
    """Return ``q`` updated so the joint's world rotation becomes ``world_new``.

    :param q: Current configuration ``(nq,)``.
    :param world_old: The joint's current world pose ``(7,)`` (scalar-last).
    :param world_new: The desired world pose ``(7,)`` from the gizmo.

    Free-flyer edits touch the rotation only.  Spherical edits right-multiply
    the local quaternion by the world-frame delta.  Channel edits recompose the
    current euler rotation with the delta and re-decompose it, dropping any
    missing channel (a best-effort projection onto the reachable axes) and
    splitting a shared axis equally between its hosts.
    """
    q = q.clone()
    if handle.kind == "free_flyer":
        q[handle.idx_q + 3 : handle.idx_q + 7] = so3.normalize(world_new[3:7])
        return q

    delta = _local_delta(world_old, world_new)
    if handle.kind == "spherical":
        idx = handle.idx_q
        local = so3.normalize(q[idx : idx + 4])
        q[idx : idx + 4] = so3.normalize(so3.compose(local, so3.exp(delta)))
        return q

    if handle.kind == "channels":
        euler_old = q.new_zeros(3)
        counts = {"x": 0, "y": 0, "z": 0}
        for axis, idx_q in handle.channels:
            euler_old[_EULER_INDEX[axis]] = euler_old[_EULER_INDEX[axis]] + q[idx_q]
            counts[axis] += 1
        rotation_new = so3.compose(so3.from_euler(euler_old), so3.exp(delta))
        euler_new = so3.to_euler(rotation_new)
        for axis, idx_q in handle.channels:
            q[idx_q] = euler_new[_EULER_INDEX[axis]] / counts[axis]
        return q

    raise ValueError(f"unknown joint handle kind {handle.kind!r}")


__all__ = ["JointHandle", "apply_local_rotation", "joint_handles"]
