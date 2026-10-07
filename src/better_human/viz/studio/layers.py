"""Studio scene layers on the BetterRobot viewer: segmentation submeshes, rigid parts, and the group owner."""

from __future__ import annotations

import colorsys
from typing import Any, Literal

import numpy as np
import torch
from better_robot.viewer import LIGHT, FramesLayer, Layer, PointsLayer, SkeletonLayer, Theme

from better_human.core.mass import _majority_face_parts, build_part_topology, meshes_from_topology
from better_human.viz.human_layers import mesh_layer

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


def frame_poses(session: StudioSession) -> np.ndarray:
    """Authored frames as ``(1, F, 7)`` poses: vertex position, parent-joint orientation (``session.frame_set`` order)."""
    entries = list(session.frame_set.values())
    poses = np.zeros((1, len(entries), 7), np.float32)
    if entries:
        joint_poses = session.state.joint_poses.detach().cpu().numpy().reshape(-1, 7)
        poses[0, :, :3] = session.vertices.detach().cpu().numpy()[[entry.vertex for entry in entries]]
        poses[0, :, 3:] = joint_poses[[session.resolved_joint(entry) for entry in entries], 3:]
    return poses


class SegmentationLayer(Layer):
    """One coloured submesh per part plus one for unassigned faces; :meth:`rebuild` after a segmentation edit.

    ``vertices (1, V, 3)`` is rewritten in place on pose changes; faces are split by majority part label.
    """

    stale = False

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

    def set_visible(self, visible: bool) -> None:
        # A hidden layer skips rebuilds (StudioLayers marks it stale); catch up when it is shown.
        if visible and self.stale:
            self.stale = False
            self.rebuild()
        super().set_visible(visible)

    def set_step(self, step: int) -> None:
        for mesh in self._meshes:
            mesh.vertices = self.vertices[step]


class PartsLayer(Layer):
    """Closed part meshes posed rigidly by each part's dominant native joint, pushed apart by ``explode``.

    Parts come from :func:`build_part_topology` on the compacted segmentation (``closed = True``); when no
    closed topology can be built for the current labels they fall back to open submeshes (``closed = False``).
    """

    stale = False

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

    def set_visible(self, visible: bool) -> None:
        # A hidden layer skips rebuilds (StudioLayers marks it stale); catch up when it is shown.
        if visible and self.stale:
            self.stale = False
            self.rebuild()
        super().set_visible(visible)

    def set_step(self, step: int) -> None:
        poses = self.session.joint_poses.detach().cpu().numpy()
        center = poses[:, :3].mean(axis=0)
        for joint, handle, local_center in self._handles_by_part:
            pose = poses[joint]
            world_center = _quat_matrix(pose[3:]) @ local_center + pose[:3]
            handle.position = pose[:3] + self.explode * (world_center - center)
            handle.wxyz = np.array([pose[6], pose[3], pose[4], pose[5]])


MeshMode = Literal["skin", "segmentation", "parts"]


class BodyMeshLayer(Layer):
    """The one body mesh, drawn in colour ``mode``: ``skin``, ``segmentation`` (a submesh per part) or ``parts``.

    A viser mesh carries a single colour, so a mode switch rebuilds this layer's nodes (``viewer.rebuild``)
    instead of adding a second layer. The :class:`SegmentationLayer` and :class:`PartsLayer` it delegates to
    are drawing providers only and are never added to the viewer themselves; ``parts`` keeps ``explode``.
    """

    def __init__(
        self, name: str, session: StudioSession, *, color: Any, opacity: float = 1.0, explode: float = 0.0, mode: MeshMode = "skin"
    ) -> None:
        super().__init__(name, 1)
        self.mode = mode
        self.skin = mesh_layer(session.state, name=name, color=color, opacity=opacity)
        self.segmentation = SegmentationLayer(name, session, opacity=opacity)
        self.parts = PartsLayer(name, session, explode=explode)

    @property
    def active(self) -> Layer:
        """The provider drawing the current mode."""
        return {"skin": self.skin, "segmentation": self.segmentation, "parts": self.parts}[self.mode]

    def build(self, scene: Any, root: str) -> None:
        self.active.stale = False
        self.active.build(scene, root)

    def set_step(self, step: int) -> None:
        self.active.set_step(step)

    def world_points(self, step: int) -> np.ndarray | None:
        return self.active.world_points(step)

    def set_visible(self, visible: bool) -> None:
        super().set_visible(visible)
        self.active.set_visible(visible)

    def dispose(self) -> None:
        # All providers: a mode switch changes ``mode`` before the viewer disposes the old nodes.
        for provider in (self.skin, self.segmentation, self.parts):
            provider.dispose()

    def set_opacity(self, opacity: float) -> None:
        """Set skin and segmentation opacity on the live nodes (parts stay opaque)."""
        self.skin.opacity = self.segmentation.opacity = opacity
        if self.mode != "parts":
            for handle in self.active._handles:
                handle.opacity = None if opacity >= 1.0 else opacity


class StudioLayers:
    """Owns the studio layers under viewer group ``"human"`` and keeps them in sync with ``session``.

    Layers: the :class:`HumanView` mesh (and frames when the body has contact frames), a skeleton with
    per-joint colours (selection highlight), :class:`SegmentationLayer`, :class:`PartsLayer` and the
    edit-overlay :class:`PointsLayer`.
    """

    def __init__(self, viewer: Any, session: StudioSession, *, theme: Theme = LIGHT) -> None:
        self.viewer = viewer
        self.session = session
        self.theme = theme
        self.opacity = 1.0
        self.radius = 0.015
        self.explode = 0.0
        self.highlight: int | None = None
        self.edit_vertices = np.zeros(0, dtype=np.int64)
        self.layers: dict[str, Layer] = {}
        for name, (cls, args, kwargs) in self._specs().items():
            layer = self.layers[name] = cls(name, *args, **kwargs)
            self.viewer.add(layer, GROUP)
            self.viewer.set_visible(layer, name in ("mesh", "skeleton", "frames"))
        self.set_highlight(None, show=False)
        self.viewer.show()
        session.subscribe(self._on_change)

    mesh = property(lambda self: self.layers["mesh"])
    frames = property(lambda self: self.layers["frames"])
    skeleton = property(lambda self: self.layers["skeleton"])
    edit = property(lambda self: self.layers["edit"])

    @property
    def visible(self) -> dict[str, bool]:
        """Visibility of every layer by name; the viewer's Layers checkboxes own it."""
        return {name: layer.visible for name, layer in self.layers.items()}

    def _specs(self) -> dict[str, tuple[type, tuple, dict]]:
        # Constructor arguments of every layer from the session; a body switch re-runs them on the same objects.
        session = self.session
        num_joints = len(session.body.joint_names)
        self._base_colors = np.broadcast_to(np.asarray(self.theme.joint, np.float32), (1, num_joints, 3)).copy()
        mode = self.layers["mesh"].mode if "mesh" in self.layers else "skin"
        ids = self.edit_vertices if len(self.edit_vertices) else np.zeros(1, dtype=np.int64)
        joints = session.state.joints.detach().cpu().numpy()[None]
        return {
            "mesh": (BodyMeshLayer, (session,), dict(color=self.theme.human, opacity=self.opacity, explode=self.explode, mode=mode)),
            # The authored frame set replaces HumanView's model-frames layer (it starts from body.contact_frames).
            "frames": (FramesLayer, (frame_poses(session),), dict(axes_length=0.04)),
            "skeleton": (
                SkeletonLayer,
                (joints, session.body.parents),
                dict(radius=self.radius, colors=self._base_colors, bone_color=self.theme.link),
            ),
            "edit": (
                PointsLayer,
                (session.vertices.detach().cpu().numpy()[ids][None],),
                dict(colors=(1.0, 0.3, 0.2), point_size=0.012),
            ),
        }

    def _renew(self, name: str) -> None:
        # Same layer object, new data: the viewer keeps its entry and Layers checkbox (no remove_group).
        layer = self.layers[name]
        cls, args, kwargs = self._specs()[name]
        visible = layer.visible
        layer.dispose()
        cls.__init__(layer, name, *args, **kwargs)
        layer._visible = visible
        self.viewer.rebuild(layer)

    def set_mode(self, mode: MeshMode) -> None:
        """Draw the body mesh in colour ``mode`` (``skin``, ``segmentation``, ``parts``) on the same layer."""
        self.mesh.mode = mode
        self.viewer.rebuild(self.mesh)

    def set_visible(self, name: str, visible: bool) -> None:
        """Show or hide one layer by name (``mesh``, ``skeleton``, ``frames``, ``segmentation``, ``parts``)."""
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
        self.mesh.set_opacity(opacity)

    def set_radius(self, radius: float) -> None:
        """Set the skeleton joint radius in metres."""
        self.radius = radius
        self.skeleton.radius = np.broadcast_to(np.float32(radius), self.skeleton.radius.shape)
        self.skeleton.bone_radius = radius * 0.6
        self.viewer.show()

    def set_explode(self, explode: float) -> None:
        """Set the parts explode factor (0 = assembled)."""
        self.explode = self.mesh.parts.explode = explode
        self.viewer.show()

    def set_edit_vertices(self, vertex_ids: np.ndarray) -> None:
        """Mark vertices ``(N,)`` in the edit overlay (empty hides it)."""
        self.edit_vertices = np.asarray(vertex_ids, dtype=np.int64)
        self._renew("edit")
        self.viewer.set_visible(self.edit, bool(len(self.edit_vertices)))

    def _refresh(self, layer: SegmentationLayer | PartsLayer) -> None:
        # Through the viewer so the mesh keeps its visibility; hidden or inactive providers rebuild when drawn.
        if self.mesh.active is layer and self.mesh.visible:
            self.viewer.rebuild(self.mesh)
        else:
            layer.stale = True

    def refresh_frames(self) -> None:
        """Redraw the authored frames from ``session.frame_set``; a changed count rebuilds the layer."""
        poses = frame_poses(self.session)
        if poses.shape == self.frames.poses.shape:
            self.frames.poses[...] = poses
            self.viewer.show()
        else:
            self.frames.poses = poses
            self.viewer.rebuild(self.frames)

    def _on_change(self, kind: ChangeKind) -> None:
        if kind == "frames":
            self.refresh_frames()
        elif kind in ("pose", "shape"):
            session = self.session
            if kind == "shape":
                self._refresh(self.mesh.parts)
            self.skeleton.joints[...] = session.state.joints.detach().cpu().numpy()[None]
            self.mesh.skin.vertices[...] = self.mesh.segmentation.vertices[...] = session.vertices.detach().cpu().numpy()[None]
            if len(self.edit_vertices):
                self.edit.points[...] = session.vertices.detach().cpu().numpy()[self.edit_vertices][None]
            self.frames.poses[...] = frame_poses(session)
            self.viewer.show()
        elif kind == "segmentation":
            self._refresh(self.mesh.segmentation)
            self._refresh(self.mesh.parts)
            self.viewer.show()
        elif kind == "body":
            self.highlight = None
            self.edit_vertices = np.zeros(0, dtype=np.int64)
            for name in self.layers:
                self._renew(name)
            self.set_highlight(None)


__all__ = ["GROUP", "BodyMeshLayer", "MeshMode", "PartsLayer", "frame_poses", "SegmentationLayer", "StudioLayers", "part_colors"]
