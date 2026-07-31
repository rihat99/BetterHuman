"""Skinned-surface layer: solid, segmentation-colored, or GPU-skinned mesh.

Plain mode uploads one CPU-deformed mesh (solid light blue or one submesh per
segmentation part) and rewrites ``.vertices`` in place on pose/shape.  Fast mode
uploads a viser skinned mesh once (rest ``v_shaped``, native bind poses, native
skin weights) and only writes per-bone transforms per pose -- no LBS, no vertex
upload -- at the cost of the pose correctives (skipped by design) and
segmentation coloring (a skinned mesh is single-colored).  Updates are skipped
entirely while the layer is hidden.
"""

from __future__ import annotations

import numpy as np
import torch
import viser
from better_robot.lie import se3

from better_human.core.mass import _majority_face_parts

from ..state import StudioSession
from .base import MESH_COLOR, UNASSIGNED_COLOR, part_palette, to_numpy

_ROOT = "/mesh"
#: Thin dark color for the wireframe overlay drawn on top of the solid mesh.
WIREFRAME_COLOR: tuple[int, int, int] = (60, 60, 70)


def fast_skinning_available(body: object) -> bool:
    """Return whether client-side LBS can reproduce this body's skinning.

    viser applies plain linear blend skinning, so a body is representable when
    it carries no per-joint similarity scale (or a scale that is the identity);
    a non-trivial scale (some MHR levels of detail) cannot be expressed and the
    fast path is disabled for it.
    """
    scales = body.values.joint_scales
    return scales is None or bool(torch.allclose(scales, torch.ones_like(scales)))


def _top4_normalized_weights(weights: torch.Tensor) -> np.ndarray:
    """Return skin weights truncated to the top 4 per vertex and renormalized.

    viser keeps only each vertex's four largest weights and does *not*
    renormalize (``_scene_api.py:1693-1721``); vertices with more than four
    influences (MHR) would otherwise sum to < 1 and pull the surface toward the
    origin.  Pre-truncating and renormalizing here keeps the shape correct.
    SMPL/SMPL-X (<= 4 influences) are unchanged.
    """
    top = torch.topk(weights, k=min(4, weights.shape[-1]), dim=-1)
    truncated = torch.zeros_like(weights)
    truncated.scatter_(-1, top.indices, top.values)
    truncated = truncated / truncated.sum(-1, keepdim=True).clamp(min=1e-8)
    return to_numpy(truncated)


def _bind_bone_poses(body: object) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(bone_positions (J,3), bone_wxyzs (J,4))`` for the rest bind pose.

    The client computes each bone's inverse-bind from these initial transforms
    and applies ``world_j @ bind_j^-1`` per frame, matching
    :func:`better_human.core.pipeline.skinning_transforms` exactly.
    """
    values = body.values
    num_joints = len(body.structure.joint_names)
    if values.inverse_bind_poses is None:
        positions = to_numpy(values.rest_joints)
        wxyzs = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (num_joints, 1))
        return positions, wxyzs
    bind = se3.inverse(values.inverse_bind_poses)
    positions = to_numpy(bind[:, :3])
    quaternion = to_numpy(bind[:, 3:7])  # scalar-last (x, y, z, w)
    wxyzs = np.column_stack((quaternion[:, 3], quaternion[:, 0], quaternion[:, 1], quaternion[:, 2]))
    return positions, wxyzs.astype(np.float32)


class MeshLayer:
    """One skinned surface rendered solid, segmentation-colored, or GPU-skinned."""

    name = "mesh"

    def __init__(self) -> None:
        self._server: viser.ViserServer | None = None
        self._session: StudioSession | None = None
        self._mode = "solid"
        self._fast = False
        self._wireframe = False
        self._opacity = 0.9
        self._visible = True
        self._dirty = False
        #: Per-node ``(handle, global vertex indices)`` for in-place updates.
        self._nodes: list[tuple[object, np.ndarray]] = []
        self._skinned: object | None = None
        #: Optional wireframe mesh drawn on top of the solid surface.
        self._wire_overlay: object | None = None

    def build(self, server: viser.ViserServer, session: StudioSession) -> None:
        """Bind the server and session, then create the surface nodes."""
        self._server = server
        self._session = session
        self._rebuild()

    def update(self, session: StudioSession, kind: str) -> None:
        """Rewrite vertices/bones in place, or rebuild on a segmentation change."""
        self._session = session
        if not self._visible:
            self._dirty = True
            return
        if self._skinned is not None:
            if kind == "shape":
                self._rebuild()
            elif kind == "pose":
                self._place_bones()
            return
        if kind == "segmentation" and self._mode == "segmentation":
            self._rebuild()
        elif kind in ("pose", "shape"):
            vertices = to_numpy(session.vertices)
            for handle, indices in self._nodes:
                handle.vertices = vertices[indices]
            if self._wire_overlay is not None:
                self._wire_overlay.vertices = vertices

    def set_visible(self, visible: bool) -> None:
        """Show or hide every surface node, refreshing if it fell behind."""
        self._visible = visible
        for handle, _ in self._nodes:
            handle.visible = visible
        if self._skinned is not None:
            self._skinned.visible = visible
        if self._wire_overlay is not None:
            self._wire_overlay.visible = visible
        if visible and self._dirty:
            self._dirty = False
            self._rebuild()

    def clear(self) -> None:
        """Remove every surface node from the scene."""
        self._clear()

    def set_opacity(self, opacity: float) -> None:
        """Set the surface opacity on every node."""
        self._opacity = opacity
        for handle, _ in self._nodes:
            handle.opacity = opacity
        if self._skinned is not None:
            self._skinned.opacity = opacity

    def set_wireframe(self, wireframe: bool) -> None:
        """Toggle a wireframe overlay drawn on top of the solid mesh.

        The base surface stays solid; a separate thin dark wireframe node is
        overlaid so both are visible at once.  Unavailable in fast-skinning mode
        (the caller greys out the checkbox), so nothing is drawn while ``_fast``.
        """
        self._wireframe = wireframe
        self._refresh_wire_overlay()

    def set_mode(self, mode: str) -> None:
        """Switch between ``"solid"`` and ``"segmentation"`` coloring."""
        if mode not in ("solid", "segmentation"):
            raise ValueError(f"unknown mesh mode {mode!r}")
        self._mode = mode
        self._rebuild()

    def set_fast(self, fast: bool) -> None:
        """Enable or disable the GPU-skinned fast path and rebuild the mesh."""
        self._fast = fast
        self._rebuild()

    @property
    def fast(self) -> bool:
        """Whether the GPU-skinned fast path is currently active."""
        return self._fast

    def _clear(self) -> None:
        for handle, _ in self._nodes:
            handle.remove()
        self._nodes = []
        if self._skinned is not None:
            self._skinned.remove()
            self._skinned = None
        if self._wire_overlay is not None:
            self._wire_overlay.remove()
            self._wire_overlay = None

    def _rebuild(self) -> None:
        if self._server is None or self._session is None:
            return
        self._clear()
        if self._fast and fast_skinning_available(self._session.body):
            self._add_skinned()
        elif self._mode == "solid":
            self._add_solid()
        else:
            self._add_segmentation()
        self._refresh_wire_overlay()

    def _refresh_wire_overlay(self) -> None:
        if self._wire_overlay is not None:
            self._wire_overlay.remove()
            self._wire_overlay = None
        if not self._wireframe or self._fast or self._server is None or self._session is None:
            return
        session = self._session
        self._wire_overlay = self._server.scene.add_mesh_simple(
            f"{_ROOT}/wire",
            vertices=to_numpy(session.vertices),
            faces=session.body.structure.faces.detach().cpu().numpy(),
            color=WIREFRAME_COLOR,
            wireframe=True,
            opacity=1.0,
            side="double",
            visible=self._visible,
        )

    def _add_skinned(self) -> None:
        session = self._session
        body = session.body
        positions, wxyzs = _bind_bone_poses(body)
        self._skinned = self._server.scene.add_mesh_skinned(
            f"{_ROOT}/skinned",
            vertices=to_numpy(body.values.v_shaped),
            faces=body.structure.faces.detach().cpu().numpy(),
            bone_wxyzs=wxyzs,
            bone_positions=positions,
            skin_weights=_top4_normalized_weights(body.values.skinning_weight_matrix),
            color=MESH_COLOR,
            wireframe=False,
            opacity=self._opacity,
            side="double",
            visible=self._visible,
        )
        self._place_bones()

    def _place_bones(self) -> None:
        if self._skinned is None or self._session is None:
            return
        poses = self._session.native_poses
        for index, bone in enumerate(self._skinned.bones):
            pose = poses[index]
            bone.position = pose[:3].detach().cpu().numpy()
            quaternion = pose[3:7].detach().cpu().tolist()
            bone.wxyz = np.array([quaternion[3], quaternion[0], quaternion[1], quaternion[2]], dtype=np.float64)

    def _add_solid(self) -> None:
        session = self._session
        indices = np.arange(session.body.structure.num_vertices)
        handle = self._server.scene.add_mesh_simple(
            f"{_ROOT}/solid",
            vertices=to_numpy(session.vertices),
            faces=session.body.structure.faces.detach().cpu().numpy(),
            color=MESH_COLOR,
            wireframe=False,
            opacity=self._opacity,
            side="double",
            visible=self._visible,
        )
        self._nodes = [(handle, indices)]

    def _add_segmentation(self) -> None:
        session = self._session
        faces = session.body.structure.faces
        vertices = to_numpy(session.vertices)
        face_parts = _majority_face_parts(faces, session.part_ids).detach().cpu().numpy()
        faces_np = faces.detach().cpu().numpy()
        num_parts = len(session.part_names)
        palette = part_palette(num_parts)
        self._nodes = []
        for part in range(num_parts):
            self._add_submesh(f"{_ROOT}/seg/{part}", faces_np[face_parts == part], vertices, tuple(palette[part]))
        unassigned = faces_np[(face_parts < 0) | (face_parts >= num_parts)]
        self._add_submesh(f"{_ROOT}/seg/unassigned", unassigned, vertices, UNASSIGNED_COLOR)

    def _add_submesh(self, name: str, part_faces: np.ndarray, vertices: np.ndarray, color: tuple) -> None:
        if part_faces.shape[0] == 0:
            return
        used = np.unique(part_faces)
        remap = np.full(vertices.shape[0], -1, dtype=np.int64)
        remap[used] = np.arange(used.size)
        handle = self._server.scene.add_mesh_simple(
            name,
            vertices=vertices[used],
            faces=remap[part_faces],
            color=tuple(int(component) for component in color),
            wireframe=False,
            opacity=self._opacity,
            side="double",
            visible=self._visible,
        )
        self._nodes.append((handle, used))


__all__ = ["MeshLayer", "fast_skinning_available"]
