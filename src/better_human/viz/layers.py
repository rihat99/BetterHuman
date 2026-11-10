"""Human layers for the BetterRobot viewer: a :class:`HumanState` clip as mesh, skeleton, frames and forces.

Every function takes a state ``(T, ...)`` (a clip: the batch axis is the viewer's timeline) or a single pose, and
names frames by name. ``HUMAN_THEME`` is BetterRobot's ``LIGHT`` with ``+y`` up, the axis the human families stand on.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import replace
from typing import Final, Literal, final

import numpy as np
import torch
from better_robot.exceptions import ArgumentError, ShapeError
from better_robot.lie import se3, so3
from better_robot.viewer import (
    LIGHT,
    ArrowsLayer,
    FramesLayer,
    Layer,
    MeshLayer,
    SkeletonLayer,
    SkinnedMeshLayer,
    Theme,
)

from ..core.human_model import HumanState

RGB = tuple[float, float, float]

#: BetterRobot's ``LIGHT`` palette with ``+y`` up: SMPL, SMPL-X, MANO, FLAME, SOMA and MHR stand along ``+y``.
HUMAN_THEME: Final[Theme] = replace(LIGHT, up="+y")

# A bind whose ``R R^T`` is this close to the identity moves a vertex 1 m from its joint by under 10 um.
_RIGID_ATOL = 1e-5


def _numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


def mesh_layer(
    state: HumanState, *, name: str = "mesh", color: RGB = HUMAN_THEME.human, opacity: float = 1.0
) -> MeshLayer:
    """Exact posed mesh per step (correctives and streams included): ``(T, V, 3)`` vertices written each step."""
    return MeshLayer(
        name,
        _numpy(state.vertices),
        faces=_numpy(state.identity.human.spec.faces),
        color=color,
        opacity=opacity,
    )


def skinned_layer(
    state: HumanState, *, name: str = "mesh", color: RGB = HUMAN_THEME.human, opacity: float = 1.0
) -> SkinnedMeshLayer:
    """Client-side skinning: the identity's rest mesh once, the joint poses ``(T, J, 7)`` per step.

    Cheap on the wire; draws no pose correctives or stream offsets (:func:`mesh_layer` does). The bind poses
    ``(J, 7)`` invert the identity's ``inverse_bind_affines (J, 3, 4)``. Raises ``ShapeError`` for an identity batch
    and ``ArgumentError`` for binds that scale or shear, which rigid bones cannot draw.
    """
    identity = state.identity
    inverse_binds = identity.inverse_bind_affines.detach()
    if inverse_binds.ndim != 3:
        raise ShapeError(
            f"skinned_layer draws one identity; inverse_bind_affines is {tuple(inverse_binds.shape)}: pass the state "
            "of a single identity"
        )
    rotations = inverse_binds[..., :3]
    eye = torch.eye(3, dtype=rotations.dtype, device=rotations.device)
    if (rotations @ rotations.mT - eye).abs().max() > _RIGID_ATOL or (torch.linalg.det(rotations) <= 0).any():
        raise ArgumentError(
            f"skinned_layer needs rigid binds; {identity.human.spec.name}'s binds scale or shear: use mesh_layer"
        )
    bind_poses = se3.inverse(torch.cat((inverse_binds[..., 3], so3.from_matrix(rotations)), dim=-1))
    return SkinnedMeshLayer(
        name,
        _numpy(identity.rest_vertices),
        faces=_numpy(identity.human.spec.faces),
        weights=_numpy(identity.human.skin_matrix),
        bind_poses=_numpy(bind_poses),
        bone_poses=_numpy(state.joint_poses),
        color=color,
        opacity=opacity,
    )


def skeleton_layer(
    state: HumanState, *, name: str = "skeleton", theme: Theme = HUMAN_THEME, radius: float = 0.015
) -> SkeletonLayer:
    """Joints and bones of the native skeleton."""
    return SkeletonLayer(
        name,
        _numpy(state.joint_translations),
        parent_ids=state.identity.human.spec.parents,
        radius=radius,
        colors=theme.joint,
        bone_color=theme.link,
    )


def frames_layer(state: HumanState, *, name: str = "frames", axes_length: float = 0.05) -> FramesLayer:
    """Every model frame (contact frames included) as axes."""
    return FramesLayer(name, _numpy(state.frame_poses), axes_length=axes_length)


def forces_layer(
    state: HumanState,
    frame_names: Sequence[str],
    *,
    forces: torch.Tensor,
    name: str = "forces",
    scale: float = 1e-3,
    theme: Theme = HUMAN_THEME,
) -> ArrowsLayer:
    """World ``forces (T, C, 3)`` (or ``(C, 3)``) applied at the model frames ``frame_names (C,)``, drawn as arrows
    (``scale`` metres per newton).

    Raises ``UnknownNameError`` for a frame the model does not have, ``TypeError`` when ``forces`` is not a tensor and
    ``ShapeError`` when it does not hold one 3-vector per frame name.
    """
    if not isinstance(forces, torch.Tensor):
        raise TypeError(f"forces must be a torch.Tensor, got {type(forces).__name__}")
    ids = torch.tensor([state.model.frame_id(frame) for frame in frame_names], dtype=torch.long, device=state.q.device)
    if forces.shape[-2:] != (len(ids), 3):
        raise ShapeError(
            f"forces must be (T, C, 3) or (C, 3) with C = len(frame_names) = {len(ids)}, got {tuple(forces.shape)}"
        )
    origins = state.kinematics.frame_translations.index_select(-2, ids)
    return ArrowsLayer(name, _numpy(origins), vectors=_numpy(forces), scale=scale, colors=theme.force)


def human_layers(
    state: HumanState,
    *,
    mesh: Literal["exact", "skinned"] | None = "exact",
    skeleton: bool = True,
    frames: bool = False,
    forces: Mapping[str, torch.Tensor] | None = None,
    theme: Theme = HUMAN_THEME,
    color: RGB = HUMAN_THEME.human,
    opacity: float = 1.0,
) -> list[Layer]:
    """Layers for a posed clip: ``mesh`` is ``"exact"`` (:func:`mesh_layer`), ``"skinned"`` (:func:`skinned_layer`)
    or ``None``.

    ``forces`` maps a model frame name to world forces ``(T, 3)`` or ``(3,)`` (a ``(3,)`` force holds at every step).
    Raises ``TypeError`` for a force that is not a tensor and ``ShapeError`` for one of another shape. Add the result
    with ``viewer.add(layers, group="human")``.
    """
    layers: list[Layer] = []
    if mesh == "exact":
        layers.append(mesh_layer(state, color=color, opacity=opacity))
    elif mesh == "skinned":
        layers.append(skinned_layer(state, color=color, opacity=opacity))
    elif mesh is not None:
        raise ArgumentError(f"mesh must be 'exact', 'skinned' or None, got {mesh!r}")
    if skeleton:
        layers.append(skeleton_layer(state, theme=theme))
    if frames:
        layers.append(frames_layer(state))
    if forces:
        values = list(forces.values())
        if not all(isinstance(value, torch.Tensor) for value in values):
            raise TypeError("forces must map frame names to torch.Tensor values")
        steps = {value.shape[:-1] for value in values if value.ndim == 2}
        if any(value.shape[-1:] != (3,) or value.ndim > 2 for value in values) or len(steps) > 1:
            shapes = {frame: tuple(value.shape) for frame, value in forces.items()}
            raise ShapeError(f"forces must map frame names to (T, 3) or (3,) with one T, got {shapes}")
        stacked = torch.stack(torch.broadcast_tensors(*values), dim=-2)
        layers.append(forces_layer(state, list(forces), forces=stacked, theme=theme))
    return layers


@final
class HumanView:
    """Live human: layers of one pose (``T = 1``) that :meth:`update` rewrites in place (then call ``viewer.show()``).

    Same options as :func:`human_layers` except ``forces``; ``viewer.add(view, group=...)`` adds its layers.
    """

    def __init__(
        self,
        state: HumanState,
        *,
        mesh: Literal["exact", "skinned"] | None = "exact",
        skeleton: bool = True,
        frames: bool = False,
        theme: Theme = HUMAN_THEME,
        color: RGB = HUMAN_THEME.human,
        opacity: float = 1.0,
    ) -> None:
        self.layers = human_layers(
            state, mesh=mesh, skeleton=skeleton, frames=frames, theme=theme, color=color, opacity=opacity
        )
        if any(layer.nsteps != 1 for layer in self.layers):
            raise ShapeError("HumanView takes the state of a single pose, not a clip: use human_layers for a clip")

    def __iter__(self) -> Iterator[Layer]:
        return iter(self.layers)

    def update(self, state: HumanState) -> None:
        """Write the pose of ``state`` (a single pose of the same identity) into the layers.

        Raises ``ShapeError``, and writes nothing, for a clip or the state of a body of another size.
        """
        writes: list[tuple[np.ndarray, torch.Tensor]] = []
        for layer in self.layers:
            if isinstance(layer, MeshLayer):
                writes.append((layer.vertices, state.vertices))
            elif isinstance(layer, SkinnedMeshLayer):
                writes.append((layer.bone_poses, state.joint_poses))
            elif isinstance(layer, SkeletonLayer):
                writes.append((layer.joints, state.joint_translations))
            elif isinstance(layer, FramesLayer):
                writes.append((layer.poses, state.frame_poses))
        for drawn, value in writes:
            if value.shape not in (drawn.shape, drawn.shape[1:]):
                raise ShapeError(
                    f"HumanView.update takes a single pose of the drawn body: got {tuple(value.shape)} where the view "
                    f"draws {drawn.shape[1:]}"
                )
        for drawn, value in writes:
            drawn[...] = _numpy(value)


__all__ = [
    "HUMAN_THEME",
    "HumanView",
    "forces_layer",
    "frames_layer",
    "human_layers",
    "mesh_layer",
    "skeleton_layer",
    "skinned_layer",
]
