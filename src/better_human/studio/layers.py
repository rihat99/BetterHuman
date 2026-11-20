"""The studio's scene on the BetterRobot viewer: the body in one of three colour modes, the native skeleton with the
selected joint, the authored frames, the edit overlay and a ground under the feet.

Every layer draws exact vertices (no skinned mesh), so a body swap replaces plain meshes only. Each group of layers
has its own tag, so the viewer's Layers section shows and hides it.
"""

from __future__ import annotations

import colorsys
from typing import Any, Literal, final

import numpy as np
import torch
from better_robot.exceptions import ArgumentError
from better_robot.lie import se3
from better_robot.viewer import (
    FramesLayer,
    GroundLayer,
    Layer,
    MeshLayer,
    MeshSpec,
    PointsLayer,
    RigidMeshLayer,
    SpheresLayer,
    Theme,
)

from better_human.mass import PartTopology
from better_human.viz import HUMAN_THEME, mesh_layer, skeleton_layers

from .state import ChangeKind, StudioSession, compact_segmentation

MeshMode = Literal["skin", "segmentation", "parts"]
#: The tag of every layer: the viewer's Layers section has one checkbox per tag but the ground's.
TAGS = {
    "mesh": "body",
    "joints": "skeleton",
    "bones": "skeleton",
    "frames": "frames",
    "selected": "selection",
    "edit": "selection",
    "ground": "ground",
}
_UNASSIGNED = (0.55, 0.55, 0.55)
_HIGHLIGHT = (1.0, 0.78, 0.16)
_HIGHLIGHT_SCALE = 1.6  # the selected joint's sphere over the joint radius, so it covers the joint's own
_IDENTITY = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], np.float32)


def part_colors(num_parts: int) -> np.ndarray:
    """Return ``(P, 3)`` well-spread RGB colours in ``[0, 1]``."""
    return np.array([colorsys.hsv_to_rgb((0.61803 * p) % 1.0, 0.55, 0.9) for p in range(num_parts)], np.float32)


def frame_poses(session: StudioSession) -> np.ndarray:
    """Authored frames as ``(1, F, 7)`` poses, as the model's contact frames (``session.frame_set`` order)."""
    return session.frame_poses.detach().cpu().numpy().astype(np.float32).reshape(1, -1, 7)


def vertex_colors(session: StudioSession) -> np.ndarray:
    """``(V, 3)``: each vertex in its part's colour, grey when it has none."""
    colors = np.concatenate([part_colors(len(session.part_names)), np.float32([_UNASSIGNED])])
    ids = session.part_ids.cpu().numpy()
    return colors[np.where((ids >= 0) & (ids < len(session.part_names)), ids, -1)]


def part_meshes(
    session: StudioSession, rest_vertices: torch.Tensor
) -> tuple[list[tuple[int, np.ndarray, np.ndarray, np.ndarray]], bool]:
    """``(meshes, closed)``: ``(part, mesh vertex ids (n0,), vertices (n, 3), faces (f, 3))`` per part of
    ``rest_vertices``, and whether they are closed.

    ``part`` indexes ``session.part_names``; closed meshes (``PartTopology.from_part_ids`` on the compacted
    segmentation) append cap vertices after the ``n0`` mesh vertices. Labels that close into no part meshes (a part
    without faces, an unwalkable seam) still draw, as open submeshes; Build bodies reports the reason.
    """
    faces = session.body.spec.faces
    ids, names = compact_segmentation(faces, session.part_ids, session.part_names)
    kept = [session.part_names.index(name) for name in names]
    try:
        topology = PartTopology.from_part_ids(faces, ids, num_parts=len(names))
        meshes = topology.meshes(rest_vertices)
    except ArgumentError:
        face_parts = PartTopology.face_parts(faces, session.part_ids).cpu().numpy()
        faces_np, rest_np = faces.cpu().numpy(), rest_vertices.detach().cpu().numpy()
        open_meshes = []
        for part in range(len(session.part_names)):
            part_faces = faces_np[face_parts == part]
            if len(part_faces):
                used, local_faces = np.unique(part_faces, return_inverse=True)
                open_meshes.append((part, used, rest_np[used], local_faces.reshape(-1, 3)))
        return open_meshes, False
    return [
        (kept[p], used.cpu().numpy(), vertices.detach().cpu().numpy(), part_faces.cpu().numpy())
        for p, (used, (vertices, part_faces)) in enumerate(zip(topology.vertex_ids, meshes, strict=True))
    ], True


@final
class StudioLayers:
    """The studio's layers, kept in sync with ``session``: ``mesh`` (the body in colour ``mode``), ``joints``,
    ``bones`` and ``selected`` (the native skeleton and the selected joint), ``frames``, ``edit`` (the vertices an
    editor marks) and ``ground``, each under its tag of ``TAGS``.

    ``mode`` ``"skin"`` draws the exact mesh in the theme's body colour, ``"segmentation"`` the exact mesh with each
    vertex in its part's colour (opaque), ``"parts"`` closed part meshes riding each part's dominant native joint,
    pushed apart from the joints' centre by ``explode``. A change of mode, segmentation, frame count or body replaces
    a layer; a pose change writes the arrays in place. The layers bring their own ground: build ``viewer`` with
    ``ground=False``.
    """

    def __init__(self, viewer: Any, session: StudioSession, *, theme: Theme = HUMAN_THEME) -> None:
        self.viewer = viewer
        self.session = session
        self.theme = theme
        self.mode: MeshMode = "skin"
        self.opacity = 1.0
        self.radius = 0.015
        self.explode = 0.0
        self.closed = False  # whether the parts mode draws closed meshes (part_meshes)
        self.highlight: int | None = None
        self.edit_vertices = np.zeros(0, dtype=np.int64)
        self._parts: tuple[list[int], np.ndarray] = ([], np.zeros((0, 3), np.float32))  # (joint, local centre) per part
        self.layers: dict[str, Layer] = self._layers()
        for name, layer in self.layers.items():
            viewer.add(layer, tags=TAGS[name])
        session.subscribe(self._on_change)

    mesh = property(lambda self: self.layers["mesh"])
    joints = property(lambda self: self.layers["joints"])
    bones = property(lambda self: self.layers["bones"])
    selected = property(lambda self: self.layers["selected"])
    frames = property(lambda self: self.layers["frames"])
    edit = property(lambda self: self.layers["edit"])

    def _layers(self) -> dict[str, Layer]:
        """Every layer, built from the session."""
        joints, bones = skeleton_layers(self.session.state, radius=self.radius, theme=self.theme)
        return {
            "mesh": self._mesh(),
            "joints": joints,
            "bones": bones,
            "selected": SpheresLayer(
                "selected", self._selected_center(), radii=self._selected_radius(), colors=_HIGHLIGHT, opacity=1.0
            ),
            "frames": FramesLayer("frames", frame_poses(self.session), axes_length=0.04),
            "edit": PointsLayer("edit", [self._edit_points()], colors=(1.0, 0.3, 0.2), point_size=0.012),
            "ground": GroundLayer(height=self._floor()),
        }

    def _mesh(self) -> MeshLayer | RigidMeshLayer:
        state = self.session.state
        if self.mode == "skin":
            return mesh_layer(state, color=self.theme.hues[0], opacity=self.opacity)
        if self.mode == "segmentation":
            return mesh_layer(state, colors=vertex_colors(self.session))
        identity = self.session.identity
        rest = identity.forward(identity.model.q_neutral)
        rest_poses = rest.joint_poses.detach().cpu()
        vertex_joints = self.session.vertex_native_joints.cpu().numpy()
        colors = part_colors(len(self.session.part_names))
        meshes, self.closed = part_meshes(self.session, rest.vertices.detach().cpu())
        specs, joints, centres = [], [], []
        for part, used, vertices, faces in meshes:
            joint = int(np.bincount(vertex_joints[used]).argmax())
            local = se3.act(se3.inverse(rest_poses[joint]), torch.from_numpy(vertices).to(rest_poses)).numpy()
            specs.append(
                MeshSpec(
                    local.astype(np.float32), faces.astype(np.uint32), body=len(specs), origin=_IDENTITY,
                    color=tuple(colors[part].tolist()),
                )
            )  # fmt: skip
            joints.append(joint)
            centres.append(local.mean(axis=0))
        self._parts = joints, np.asarray(centres, np.float32).reshape(-1, 3)
        return RigidMeshLayer("mesh", specs, body_poses=self._part_poses(), opacity=self.opacity)

    def _part_poses(self) -> np.ndarray:
        """``(1, P, 7)``: each part's joint pose, moved ``explode`` times its centre's offset from the joints'."""
        joints, centres = self._parts
        poses = self.session.joint_poses.detach().cpu()
        world = poses[joints].clone()
        offsets = se3.act(world, torch.from_numpy(centres).to(world)) - poses[:, :3].mean(dim=0)
        world[:, :3] += self.explode * offsets
        return world.numpy().astype(np.float32)[None]

    def _floor(self) -> float:
        """The lowest point of the body at ``q_neutral``, where the ground goes: the families stand there."""
        identity = self.session.identity
        return float(identity.forward(identity.model.q_neutral).vertices[..., 1].min())

    def _selected_center(self) -> np.ndarray:
        joint = 0 if self.highlight is None else self.highlight
        return self.session.joint_poses[joint, :3].detach().cpu().numpy().reshape(1, 1, 3)

    def _selected_radius(self) -> np.ndarray:
        return np.float32([0.0 if self.highlight is None else _HIGHLIGHT_SCALE * self.radius])  # 0 hides it

    def _edit_points(self) -> np.ndarray:
        return self.session.vertices.detach().cpu().numpy()[self.edit_vertices]

    def _replace(self, name: str, layer: Layer) -> None:
        # The new layer takes the old one's tag, place and visibility.
        self.viewer.replace(self.layers[name], layer)
        self.layers[name] = layer

    def set_mode(self, mode: MeshMode) -> None:
        """Draw the body in colour ``mode`` (``skin``, ``segmentation``, ``parts``)."""
        if mode not in ("skin", "segmentation", "parts"):
            raise ArgumentError(f"mode must be 'skin', 'segmentation' or 'parts', got {mode!r}")
        self.mode = mode
        self._replace("mesh", self._mesh())

    def set_highlight(self, joint: int | None) -> None:
        """Mark native joint ``joint`` with the highlight sphere (``None`` hides it)."""
        self.highlight = joint
        self.selected.centers[...] = self._selected_center()
        self.selected.radii[...] = self._selected_radius()
        self.selected.redraw()

    def set_opacity(self, opacity: float) -> None:
        """Set the body opacity in ``(0, 1]``; the segmentation mode stays opaque."""
        self.opacity = opacity
        self.mesh.set_opacity(opacity)

    def set_radius(self, radius: float) -> None:
        """Set the skeleton joint radius in metres."""
        self.radius = radius
        self.joints.radii[...] = radius
        self.selected.radii[...] = self._selected_radius()
        self.viewer.show()

    def set_explode(self, explode: float) -> None:
        """Set the parts explode factor (0 = assembled)."""
        self.explode = explode
        if self.mode == "parts":
            self.mesh.body_poses[...] = self._part_poses()
            self.mesh.redraw()

    def set_edit_vertices(self, vertex_ids: np.ndarray) -> None:
        """Mark vertices ``(N,)`` in the edit overlay (empty hides it)."""
        self.edit_vertices = np.asarray(vertex_ids, dtype=np.int64)
        self._replace("edit", PointsLayer("edit", [self._edit_points()], colors=(1.0, 0.3, 0.2), point_size=0.012))

    def refresh_frames(self) -> None:
        """Redraw the authored frames from ``session.frame_set``; a changed count replaces the layer."""
        poses = frame_poses(self.session)
        if poses.shape == self.frames.poses.shape:
            self.frames.poses[...] = poses
            self.frames.redraw()
        else:
            self._replace("frames", FramesLayer("frames", poses, axes_length=0.04))

    def _on_change(self, kind: ChangeKind) -> None:
        if kind == "frames":
            self.refresh_frames()
        elif kind == "body":
            self.highlight = None
            self.edit_vertices = np.zeros(0, dtype=np.int64)
            for name, layer in self._layers().items():
                self._replace(name, layer)
        elif kind == "segmentation" and self.mode != "skin":
            self._replace("mesh", self._mesh())
        elif kind in ("pose", "shape"):
            self._pose(shape=kind == "shape")

    def _pose(self, *, shape: bool) -> None:
        """Write the session's pose into every layer; a new shape also rebuilds the parts and moves the ground."""
        session = self.session
        if isinstance(self.mesh, RigidMeshLayer):
            if shape:
                self._replace("mesh", self._mesh())
            else:
                self.mesh.body_poses[...] = self._part_poses()
        else:
            self.mesh.vertices[...] = session.vertices.detach().cpu().numpy()
        joints, bones = skeleton_layers(session.state, radius=self.radius, theme=self.theme)
        self.joints.centers[...] = joints.centers
        self.bones.points[...] = bones.points
        self.selected.centers[...] = self._selected_center()
        if len(self.edit_vertices):
            self.edit.points[0, : len(self.edit_vertices)] = self._edit_points()
        self.frames.poses[...] = frame_poses(session)
        if shape:
            self._replace("ground", GroundLayer(height=self._floor()))
        self.viewer.show()
