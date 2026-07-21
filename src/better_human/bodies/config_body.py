"""Authored static metadata shared by every SMPL-family parametric body."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class BodyConfig:
    """Per-model static metadata consumed when a parametric body loads.

    Holds only authored data that a concrete body reads at construction:
    the native joint order, the optional reduction index groups, the landmark
    keypoint tables, the native axis conventions, and the segmentation/density
    asset filenames resolved from the models directory. Parents are not stored
    here; they come from the weight file's ``kintree_table``.
    """

    #: Native LBS joint names in weight-file order.
    joint_names: tuple[str, ...]
    #: OpenPose-style landmark keypoint names, aligned with ``frame_vertex_ids``.
    frame_names: tuple[str, ...]
    #: Vertex indices of each landmark keypoint, aligned with ``frame_names``.
    frame_vertex_ids: tuple[int, ...]
    #: Native length unit after loader normalization.
    unit: str
    #: Native world up axis.
    up_axis: str
    #: Native world forward axis.
    forward_axis: str
    #: Pose representation accepted by the classic converter.
    native_pose_format: str
    #: Per-part vertex-segmentation asset filename (order 03 resolves it).
    segmentation_vertex_file: str
    #: Named per-part density-table asset filename (order 03 resolves it).
    density_table: str
    #: Authored closed-3D segmentation asset filename, or ``None`` when absent.
    segmentation_closed_file: str | None = None
    #: Native indices of the body subtree (empty when it has no reduction groups).
    body_joints: tuple[int, ...] = field(default=())
    #: Native indices of the face subtree (jaw and eyes).
    face_joints: tuple[int, ...] = field(default=())
    #: Native indices of the left-hand subtree.
    left_hand_joints: tuple[int, ...] = field(default=())
    #: Native indices of the right-hand subtree.
    right_hand_joints: tuple[int, ...] = field(default=())


__all__ = ["BodyConfig"]
