"""Studio scene layers on the BetterRobot viewer: segmentation submeshes, rigid parts, and the group owner."""

from __future__ import annotations

import colorsys
from typing import Any

import numpy as np
import torch
from better_robot.viewer import DARK, Layer, PointsLayer, SkeletonLayer, Theme

from better_human.core.mass import _majority_face_parts, build_part_topology, meshes_from_topology
from better_human.viz.human_layers import HumanView

from .state import ChangeKind, StudioSession, compact_segmentation

GROUP = "human"
_UNASSIGNED = (0.55, 0.55, 0.55)
_HIGHLIGHT = np.array([255, 200, 40], dtype=np.uint8)


def part_colors(num_parts: int) -> np.ndarray:
    """Return ``(P, 3)`` well-spread RGB colours in ``[0, 1]``."""
    return np.array([colorsys.hsv_to_rgb((0.61803 * p) % 1.0, 0.55, 0.9) for p in range(num_parts)], np.float32)


def _quat_matrix(quat: np.ndarray) -> np.ndarray:
    """Rotation matrix ``(3, 3)`` of a scalar-last unit quaternion ``(4,)``."""
    x, y, z, w = quat
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


class SegmentationLayer(Layer):
    """One coloured submesh per part plus one for unassigned faces; :meth:`rebuild` after a segmentation edit.

    ``vertices (1, V, 3)`` is rewritten in place on pose changes; faces are split by majority part label.
    """

    def __init__(self, name: str, session: StudioSession, *, opacity: float = 1.0) -> None:
        super().__init__(name, 1)
        self.session = session
        self.opacity = opacity
        self.vertices = session.vertices.detach().cpu().numpy().astype(np.float32)[None]
        self.faces = session.body.faces.cpu().numpy().astype(np.uint32)
        self._scene: Any = None
        self._root = ""
        self._meshes: list[Any] = []

    def face_groups(self) -> list[tuple[np.ndarray, tuple[float, float, float]]]:
        """Return ``(faces (F_p, 3), rgb)`` per part, then the unassigned faces; empty groups are omitted."""
        face_parts = _majority_face_parts(self.session.body.faces, self.session.part_ids).cpu().numpy()
        colors = part_colors(len(self.session.part_names))
        groups = [(self.faces[face_parts == p], tuple(colors[p])) for p in range(len(self.session.part_names))]
        groups.append((self.faces[(face_parts < 0) | (face_parts >= len(colors))], _UNASSIGNED))
        return [(faces, color) for faces, color in groups if len(faces)]

    def build(self, scene: Any, root: str) -> None:
        self._scene, self._root = scene, root
        self._meshes = [
            self._track(
                scene.add_mesh_simple(
                    f"{root}/part_{index}",
                    self.vertices[0],
                    faces,
                    color=tuple(int(255 * c) for c in color),
                    opacity=None if self.opacity >= 1.0 else self.opacity,
                )
            )
            for index, (faces, color) in enumerate(self.face_groups())
        ]

    def rebuild(self) -> None:
        """Recreate the submeshes from the session's current segmentation."""
        self.dispose()
        self.build(self._scene, self._root)

    def set_step(self, step: int) -> None:
        for mesh in self._meshes:
            mesh.vertices = self.vertices[step]


class PartsLayer(Layer):
    """Closed part meshes posed rigidly by each part's dominant native joint, pushed apart by ``explode``.

    Parts come from :func:`build_part_topology` on the compacted segmentation (``closed = True``); when no
    closed topology can be built for the current labels they fall back to open submeshes (``closed = False``).
    """

    def __init__(self, name: str, session: StudioSession, *, explode: float = 0.0) -> None:
        super().__init__(name, 1)
        self.session = session
        self.explode = explode
        self.closed = False
        self._scene: Any = None
        self._root = ""
        self._handles_by_part: list[tuple[int, Any, np.ndarray]] = []

    def part_meshes(self, rest_vertices: torch.Tensor) -> list[tuple[int, np.ndarray, np.ndarray, np.ndarray]]:
        """Return ``(part, mesh vertex ids (n0,), vertices (n, 3), faces (f, 3))`` per part of ``rest_vertices``.

        ``part`` indexes ``session.part_names``; closed meshes append cap vertices after the ``n0`` mesh vertices.
        """
        session = self.session
        faces = session.body.faces
        ids, names = compact_segmentation(faces, session.part_ids, session.part_names)
        kept = [session.part_names.index(name) for name in names]
        try:
            topology = build_part_topology(faces, ids, len(names))
            meshes = meshes_from_topology(rest_vertices, topology)
        except ValueError:
            self.closed = False
        else:
            self.closed = True
            return [
                (kept[p], used.cpu().numpy(), vertices.detach().cpu().numpy(), part_faces.cpu().numpy())
                for p, (used, (vertices, part_faces)) in enumerate(zip(topology.vertex_indices, meshes, strict=True))
            ]
        face_parts = _majority_face_parts(faces, session.part_ids).cpu().numpy()
        faces_np, rest_np = faces.cpu().numpy(), rest_vertices.detach().cpu().numpy()
        meshes_open = []
        for part in range(len(session.part_names)):
            part_faces = faces_np[face_parts == part]
            if len(part_faces):
                used, local_faces = np.unique(part_faces, return_inverse=True)
                meshes_open.append((part, used, rest_np[used], local_faces.reshape(-1, 3)))
        return meshes_open

    def build(self, scene: Any, root: str) -> None:
        self._scene, self._root = scene, root
        session = self.session
        rest = session.identity.forward(session.identity.q_neutral)
        rest_poses = rest.joint_poses.detach().cpu().numpy()
        joints = session.vertex_native_joints.cpu().numpy()
        colors = part_colors(len(session.part_names))
        self._handles_by_part = []
        for part, used, vertices, faces in self.part_meshes(rest.vertices().detach().cpu()):
            joint = int(np.bincount(joints[used]).argmax())
            local = (vertices - rest_poses[joint, :3]) @ _quat_matrix(rest_poses[joint, 3:])
            handle = scene.add_mesh_simple(
                f"{root}/part_{part}",
                local.astype(np.float32),
                faces.astype(np.uint32),
                color=tuple(int(255 * c) for c in colors[part]),
            )
            self._handles_by_part.append((joint, self._track(handle), local.mean(axis=0)))

    def rebuild(self) -> None:
        """Recreate the part meshes from the session's current segmentation and shape."""
        self.dispose()
        self.build(self._scene, self._root)

    def set_step(self, step: int) -> None:
        poses = self.session.joint_poses.detach().cpu().numpy()
        center = poses[:, :3].mean(axis=0)
        for joint, handle, local_center in self._handles_by_part:
            pose = poses[joint]
            world_center = _quat_matrix(pose[3:]) @ local_center + pose[:3]
            handle.position = pose[:3] + self.explode * (world_center - center)
            handle.wxyz = np.array([pose[6], pose[3], pose[4], pose[5]])


class StudioLayers:
    """Owns the studio layers under viewer group ``"human"`` and keeps them in sync with ``session``.

    Layers: the :class:`HumanView` mesh (and frames when the body has contact frames), a skeleton with
    per-joint colours (selection highlight), :class:`SegmentationLayer`, :class:`PartsLayer` and the
    edit-overlay :class:`PointsLayer`.
    """

    def __init__(self, viewer: Any, session: StudioSession, *, theme: Theme = DARK) -> None:
        self.viewer = viewer
        self.session = session
        self.theme = theme
        self.opacity = 1.0
        self.radius = 0.015
        self.explode = 0.0
        self.highlight: int | None = None
        self.edit_vertices = np.zeros(0, dtype=np.int64)
        self.visible: dict[str, bool] = {"mesh": True, "skeleton": True, "frames": True}
        self._add()
        session.subscribe(self._on_change)

    def _add(self) -> None:
        session = self.session
        frames = session.body.contact_frames is not None
        self.view = HumanView(session.state, skeleton=False, frames=frames, theme=self.theme, opacity=self.opacity)
        num_joints = len(session.body.joint_names)
        self._base_colors = np.broadcast_to(np.asarray(self.theme.joint, np.float32), (1, num_joints, 3)).copy()
        self.skeleton = SkeletonLayer(
            "skeleton",
            session.state.joints.detach().cpu().numpy()[None],
            session.body.parents,
            radius=self.radius,
            colors=self._base_colors,
            bone_color=self.theme.link,
        )
        self.segmentation = SegmentationLayer("segmentation", session, opacity=self.opacity)
        self.parts = PartsLayer("parts", session, explode=self.explode)
        ids = self.edit_vertices if len(self.edit_vertices) else np.zeros(1, dtype=np.int64)
        self.edit = PointsLayer(
            "edit", session.vertices.detach().cpu().numpy()[ids][None], colors=(1.0, 0.3, 0.2), point_size=0.012
        )
        self.layers: dict[str, Layer] = {layer.name: layer for layer in self.view}
        self.layers.update(skeleton=self.skeleton, segmentation=self.segmentation, parts=self.parts, edit=self.edit)
        for name, layer in self.layers.items():
            self.viewer.add(layer, GROUP)
            self.viewer.set_visible(layer, self.visible.get(name, name == "edit" and bool(len(self.edit_vertices))))
        self.set_highlight(self.highlight, show=False)
        self.viewer.show()

    def rebuild(self) -> None:
        """Remove the ``"human"`` group and add every layer again from the session."""
        self.viewer.remove_group(GROUP)
        self._add()

    def set_visible(self, name: str, visible: bool) -> None:
        """Show or hide one layer by name (``mesh``, ``skeleton``, ``frames``, ``segmentation``, ``parts``)."""
        self.visible[name] = visible
        if name in self.layers:
            self.viewer.set_visible(self.layers[name], visible)

    def set_highlight(self, joint: int | None, *, show: bool = True) -> None:
        """Colour native joint ``joint`` with the highlight colour (``None`` clears it)."""
        self.highlight = joint
        colors = self.skeleton.colors  # (1, J, 3) uint8, read by SkeletonLayer.set_step
        colors[...] = np.round(self._base_colors * 255).astype(np.uint8)
        if joint is not None:
            colors[0, joint] = _HIGHLIGHT
        if show:
            self.viewer.show()

    def set_opacity(self, opacity: float) -> None:
        """Set the mesh and segmentation opacity in ``[0, 1]``."""
        self.opacity = opacity
        self.rebuild()

    def set_radius(self, radius: float) -> None:
        """Set the skeleton joint radius in metres."""
        self.radius = radius
        self.rebuild()

    def set_explode(self, explode: float) -> None:
        """Set the parts explode factor (0 = assembled)."""
        self.explode = self.parts.explode = explode
        self.viewer.show()

    def set_edit_vertices(self, vertex_ids: np.ndarray) -> None:
        """Mark vertices ``(N,)`` in the edit overlay (empty hides it)."""
        self.edit_vertices = np.asarray(vertex_ids, dtype=np.int64)
        self.rebuild()

    def _on_change(self, kind: ChangeKind) -> None:
        if kind in ("pose", "shape"):
            session = self.session
            self.view.update(session.state)
            if kind == "shape":
                self.parts.rebuild()
            self.skeleton.joints[...] = session.state.joints.detach().cpu().numpy()[None]
            self.segmentation.vertices[...] = session.vertices.detach().cpu().numpy()[None]
            if len(self.edit_vertices):
                self.edit.points[...] = session.vertices.detach().cpu().numpy()[self.edit_vertices][None]
            self.viewer.show()
        elif kind == "segmentation":
            self.segmentation.rebuild()
            self.parts.rebuild()
            self.viewer.show()
        elif kind == "body":
            self.highlight = None
            self.edit_vertices = np.zeros(0, dtype=np.int64)
            self.rebuild()


__all__ = ["GROUP", "PartsLayer", "SegmentationLayer", "StudioLayers", "part_colors"]
