"""Parts layer: closed per-part bodies posed rigidly and optionally exploded.

Each part's closed mesh is uploaded once in the shaped rest frame and moved
rigidly by its associated native joint's skinning transform, so the parts
follow the pose without a per-frame re-upload. An explode slider spreads the
parts radially from the body centroid. Hidden by default.
"""

from __future__ import annotations

import torch
import viser

from better_human.core.mass import PartMeshTopology, meshes_from_topology
from better_human.core.pipeline import skinning_transforms, skinning_transforms_from_inverse

from ..state import StudioSession, _part_representatives
from .base import part_palette, to_numpy, wxyz

_ROOT = "/parts"


def _skinning_transforms(session: StudioSession) -> torch.Tensor:
    """Return per native joint the world skinning transform ``(J, 7)``."""
    body = session.body
    if body.values.inverse_bind_poses is not None:
        return skinning_transforms_from_inverse(session.native_poses, body.values.inverse_bind_poses)
    return skinning_transforms(session.native_poses, body.values.rest_joints)


class PartsLayer:
    """Closed per-part meshes, rigidly posed by their native joint transforms."""

    name = "parts"

    def __init__(self) -> None:
        self._server: viser.ViserServer | None = None
        self._session: StudioSession | None = None
        self._visible = False
        self._explode = 0.0
        self._topology: PartMeshTopology | None = None
        self._part_ids: torch.Tensor | None = None
        self._representatives: list[int] = []
        self._rest_centroids: list[torch.Tensor] = []
        self._body_centroid = torch.zeros(3)
        self._nodes: list[object] = []

    def build(self, server: viser.ViserServer, session: StudioSession) -> None:
        """Bind the server and session; nodes appear once a topology is set."""
        self._server = server
        self._session = session

    def set_topology(self, topology: PartMeshTopology, part_ids: torch.Tensor) -> None:
        """Install a closed-part topology and its (compacted) vertex labels.

        ``part_ids`` labels each vertex with the topology part it belongs to
        (``0..num_parts-1``, ``-1`` for none); it drives per-part joint
        representatives and must match the topology's part count exactly.
        """
        self._topology = topology
        self._part_ids = part_ids
        if self._session is not None:
            self._rebuild()

    def update(self, session: StudioSession, kind: str) -> None:
        """Re-place on pose; re-materialize rest meshes on shape."""
        self._session = session
        if self._topology is None:
            return
        if kind == "shape":
            self._rebuild()
        elif kind == "pose":
            self._place()

    def set_visible(self, visible: bool) -> None:
        """Show or hide every part mesh."""
        self._visible = visible
        for handle in self._nodes:
            handle.visible = visible

    def set_explode(self, factor: float) -> None:
        """Set the radial explode factor and re-place the parts."""
        self._explode = factor
        self._place()

    def clear(self) -> None:
        """Remove every part mesh and drop the current topology."""
        self._clear()
        self._topology = None
        self._part_ids = None

    def _clear(self) -> None:
        for handle in self._nodes:
            handle.remove()
        self._nodes = []

    def _rebuild(self) -> None:
        if self._server is None or self._session is None or self._topology is None or self._part_ids is None:
            return
        self._clear()
        session = self._session
        num_parts = self._topology.num_parts
        self._representatives = _part_representatives(self._part_ids, num_parts, session.vertex_native_joints)
        meshes = meshes_from_topology(session.body.values.v_shaped, self._topology)
        palette = part_palette(num_parts)
        self._rest_centroids = [vertices.mean(dim=0).detach().cpu() for vertices, _ in meshes]
        self._body_centroid = session.body.values.v_shaped.mean(dim=0).detach().cpu()
        self._nodes = []
        for part, (vertices, faces) in enumerate(meshes):
            handle = self._server.scene.add_mesh_simple(
                f"{_ROOT}/{part}",
                vertices=to_numpy(vertices),
                faces=faces.detach().cpu().numpy(),
                color=tuple(int(component) for component in palette[part]),
                side="double",
                visible=self._visible,
            )
            self._nodes.append(handle)
        self._place()

    def _place(self) -> None:
        if not self._nodes or self._session is None:
            return
        transforms = _skinning_transforms(self._session)
        for part, handle in enumerate(self._nodes):
            pose = transforms[self._representatives[part]]
            offset = self._explode * (self._rest_centroids[part] - self._body_centroid)
            handle.position = (pose[:3].detach().cpu() + offset).numpy()
            handle.wxyz = wxyz(pose[3:7])


__all__ = ["PartsLayer"]
