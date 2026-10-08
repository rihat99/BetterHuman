"""Human layers for the BetterRobot viewer: a :class:`HumanState` clip as mesh, skeleton, frames and forces."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch
from better_robot.viewer import (
    ArrowsLayer,
    FramesLayer,
    Layer,
    MeshLayer,
    SkeletonLayer,
    SkinnedMeshLayer,
)
from better_robot.viewer import LIGHT, Theme

from ..core.body import HumanState, Identity

RGB = tuple[float, float, float]

_SKIN: RGB = LIGHT.human


def _frames(value: torch.Tensor, event_ndim: int) -> np.ndarray:
    """``(..., *event) -> (T, *event)`` float32 numpy; a clip is ``(T, ...)``, a single pose becomes ``T = 1``."""
    array = value.detach().cpu().float().numpy()
    if array.ndim == event_ndim:
        return array[None]
    if array.ndim != event_ndim + 1:
        raise ValueError(f"expected ({'T, ' if event_ndim else ''}...) with {event_ndim} event axes; got {array.shape}")
    return np.ascontiguousarray(array)


def mesh_layer(state: HumanState, *, name: str = "mesh", color: RGB = _SKIN, opacity: float = 1.0) -> MeshLayer:
    """Exact posed mesh per frame (correctives included): ``(T, V, 3)`` vertices written each frame."""
    return MeshLayer(name, _frames(state.vertices, 2), state.body.faces.cpu().numpy(), color=color, opacity=opacity)


def skinned_layer(
    identity: Identity, state: HumanState, *, name: str = "mesh", color: RGB = _SKIN, opacity: float = 1.0
) -> SkinnedMeshLayer:
    """Client-side skinning: bind mesh once, bone poses per frame. Cheap on the wire; no pose correctives.

    Not for authored rigs with joint scales (MHR): use :func:`mesh_layer`.
    """
    if identity.joint_scales is not None:
        raise ValueError("skinned_layer needs a rig without joint scales; use mesh_layer")
    if identity.rest_joints.ndim != 2:
        raise ValueError("skinned_layer needs a single identity (no identity batch)")
    rest = identity.rest_joints
    quaternion = torch.zeros_like(rest[:, :1]).repeat(1, 4)
    quaternion[:, 3] = 1.0
    bind = torch.cat((rest, quaternion), dim=-1)
    return SkinnedMeshLayer(
        name,
        identity.v_shaped.detach().cpu().numpy(),
        identity.body.faces.cpu().numpy(),
        identity.body.skin_matrix.detach().cpu().numpy(),
        bind.detach().cpu().numpy(),
        _frames(state.joint_poses, 2),
        color=color,
        opacity=opacity,
    )


def skeleton_layer(
    state: HumanState, *, name: str = "skeleton", theme: Theme = LIGHT, radius: float = 0.015
) -> SkeletonLayer:
    """Joints and bones of the native skeleton."""
    return SkeletonLayer(
        name, _frames(state.joints, 2), state.body.parents, radius=radius, colors=theme.joint, bone_color=theme.link
    )


def frames_layer(state: HumanState, *, name: str = "frames", axes_length: float = 0.05) -> FramesLayer:
    """Every model frame (contact frames included) as axes."""
    return FramesLayer(name, _frames(state.frame_poses, 2), axes_length=axes_length)


def forces_layer(
    state: HumanState,
    frame_ids: torch.Tensor,
    forces: torch.Tensor,
    *,
    name: str = "forces",
    scale: float = 1e-3,
    theme: Theme = LIGHT,
) -> ArrowsLayer:
    """World ``forces (T, C, 3)`` applied at model ``frame_ids (C,)``, drawn as arrows (``scale`` metres per newton)."""
    origins = state.kinematics.frame_translation[..., frame_ids, :]
    return ArrowsLayer(name, _frames(origins, 2), _frames(forces, 2), scale=scale, colors=theme.force)


def human_layers(
    state: HumanState,
    *,
    mesh: str | None = "exact",
    skeleton: bool = True,
    frames: bool = False,
    forces: Mapping[str, torch.Tensor] | None = None,
    theme: Theme = LIGHT,
    color: RGB = _SKIN,
    opacity: float = 1.0,
) -> list[Layer]:
    """Layers for a posed clip: ``mesh`` is ``"exact"`` (per-frame vertices), ``"skinned"`` (client LBS) or ``None``.

    ``forces`` maps a model frame name to world forces ``(T, 3)``. Add the result with
    ``viewer.add(layer, group="human")`` for each layer.
    """
    layers: list[Layer] = []
    if mesh == "exact":
        layers.append(mesh_layer(state, color=color, opacity=opacity))
    elif mesh == "skinned":
        layers.append(skinned_layer(state.identity, state, color=color, opacity=opacity))
    elif mesh is not None:
        raise ValueError("mesh must be 'exact', 'skinned' or None")
    if skeleton:
        layers.append(skeleton_layer(state, theme=theme))
    if frames:
        layers.append(frames_layer(state))
    if forces:
        model = state.model
        ids = torch.tensor([model.frame_id(name) for name in forces], dtype=torch.long)
        stacked = torch.stack([_as_clip(value) for value in forces.values()], dim=-2)
        layers.append(forces_layer(state, ids, stacked, theme=theme))
    return layers


class HumanView:
    """Live human: layers with ``T = 1`` arrays that :meth:`update` rewrites in place (then call ``viewer.show()``).

    Same options as :func:`human_layers` except ``forces``; iterate it to add the layers.
    """

    def __init__(
        self,
        state: HumanState,
        *,
        mesh: str | None = "exact",
        skeleton: bool = True,
        frames: bool = False,
        theme: Theme = LIGHT,
        color: RGB = _SKIN,
        opacity: float = 1.0,
    ) -> None:
        self.layers = human_layers(
            state, mesh=mesh, skeleton=skeleton, frames=frames, theme=theme, color=color, opacity=opacity
        )
        if any(layer.nsteps != 1 for layer in self.layers):
            raise ValueError("HumanView takes a single pose, not a clip")

    def __iter__(self):
        return iter(self.layers)

    def update(self, state: HumanState) -> None:
        for layer in self.layers:
            if isinstance(layer, MeshLayer):
                layer.vertices[...] = _frames(state.vertices, 2)
            elif isinstance(layer, SkinnedMeshLayer):
                layer.bone_poses[...] = _frames(state.joint_poses, 2)
            elif isinstance(layer, SkeletonLayer):
                layer.joints[...] = _frames(state.joints, 2)
            elif isinstance(layer, FramesLayer):
                layer.poses[...] = _frames(state.frame_poses, 2)


def _as_clip(value: torch.Tensor) -> torch.Tensor:
    return value if value.ndim == 2 else value[None]


__all__ = ["HumanView", "forces_layer", "frames_layer", "human_layers", "mesh_layer", "skeleton_layer", "skinned_layer"]
