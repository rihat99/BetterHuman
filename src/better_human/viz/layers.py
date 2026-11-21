"""Human layers for the BetterRobot viewer: a :class:`HumanState` clip as a body, its native skeleton, frames and
forces.

Every function takes a state ``(T, ...)`` (a clip: the batch axis is the viewer's timeline) or a single pose, and
names frames by name. A body is drawn two ways: :func:`mesh_layer` sends the exact vertices of every step (pose
correctives and streams included), and BetterRobot's ``RobotVisual(identity.model, q, skin=skin(identity))`` skins
the rest mesh in the browser from the joint poses (light on the wire, the viewer's Style control, no correctives).
``HUMAN_THEME`` is BetterRobot's ``LIGHT`` with ``+y`` up, the axis the human families stand on, and a body colour
as its first hue.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import replace
from typing import Final, Literal, final

import numpy as np
import torch
from better_robot.exceptions import ArgumentError, ShapeError
from better_robot.lie import so3
from better_robot.viewer import (
    LIGHT,
    ArrowsLayer,
    FramesLayer,
    Layer,
    LinesLayer,
    MeshLayer,
    RobotVisual,
    Skin,
    SpheresLayer,
    Theme,
)

from ..core.human_model import HumanState, Identity

RGB = tuple[float, float, float]

#: BetterRobot's ``LIGHT`` with ``+y`` up (SMPL, SMPL-X, MANO, FLAME, SOMA and MHR stand along ``+y``) and a body
#: colour as the first hue, the one ``mesh_layer`` and the first ``RobotVisual`` of a viewer draw in.
HUMAN_THEME: Final[Theme] = replace(LIGHT, up="+y", hues=((0.85, 0.72, 0.62), *LIGHT.hues))

# How far, at q_neutral, a native joint may sit from its model joint and a skin affine from the identity (float32).
_BIND_ATOL = 1e-5


def _numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


def skin(identity: Identity) -> Skin:
    """The rest mesh of ``identity`` as a BetterRobot ``Skin`` over the joints of ``identity.model``, bound at
    ``q_neutral``: ``RobotVisual(identity.model, q, skin=skin(identity))`` draws it.

    The weights ``(V, njoints)`` are the family's skin weights in model joint order, the universe and the joints
    without a native counterpart at zero. The browser skins without pose correctives and stream offsets
    (:func:`mesh_layer` draws them). Raises ``ShapeError`` for an identity batch and ``ArgumentError`` for a body the
    model's joints cannot skin: one whose native joints turn away from their model joints (SOMA's), or whose binds are
    not its pose at ``q_neutral`` or scale, which rigid bones cannot draw.
    """
    human = identity.human
    if identity.rest_vertices.ndim != 2:
        raise ShapeError(
            f"skin draws one identity; its rest vertices are {tuple(identity.rest_vertices.shape)}: bake a single "
            "identity, or use mesh_layer"
        )
    with torch.no_grad():
        rest = identity.forward(identity.model.q_neutral)
        joints = rest.kinematics.joint_poses.index_select(-2, human.joint_ids)  # (J, 7) model joints, native order
        native, affines = rest.joint_transforms, rest.skin_affines  # (J, 3, 4) each
        bound = torch.eye(3, 4, dtype=affines.dtype, device=affines.device).expand_as(affines)  # [I | 0]
        follows = (
            torch.allclose(native[..., :3], so3.to_matrix(joints[..., 3:]), atol=_BIND_ATOL)
            and torch.allclose(native[..., 3], joints[..., :3], atol=_BIND_ATOL)
            and torch.allclose(affines, bound, atol=_BIND_ATOL)
        )
    if not follows:
        raise ArgumentError(
            f"{human.spec.name}'s skinning does not follow its model's joints from q_neutral (native joints turned "
            "away from the model's, or binds that scale or are not the q_neutral pose): use mesh_layer"
        )
    weights = np.zeros((human.skin_matrix.shape[0], identity.model.njoints), np.float32)
    weights[:, _numpy(human.joint_ids)] = _numpy(human.skin_matrix)
    return Skin(_numpy(identity.rest_vertices), _numpy(human.spec.faces), weights=weights)


def mesh_layer(
    state: HumanState,
    *,
    name: str = "mesh",
    color: RGB = HUMAN_THEME.hues[0],
    colors: np.ndarray | None = None,
    opacity: float = 1.0,
) -> MeshLayer:
    """The exact posed mesh per step (correctives and streams included): ``(T, V, 3)`` vertices written each step.

    ``colors (V, 3)`` paints each vertex instead of the one ``color`` (a segmentation), as ``MeshLayer`` takes it.
    """
    return MeshLayer(
        name,
        _numpy(state.vertices),
        faces=_numpy(state.identity.human.spec.faces),
        color=color,
        colors=colors,
        opacity=opacity,
    )


def _bones(state: HumanState) -> np.ndarray:
    """``(..., N, 2, 3)`` segments from each non-root native joint to its parent."""
    parents = state.identity.human.spec.parents
    children = [joint for joint, parent in enumerate(parents) if parent >= 0]
    joints = state.joint_translations
    return _numpy(torch.stack((joints[..., children, :], joints[..., [parents[j] for j in children], :]), dim=-2))


def skeleton_layers(
    state: HumanState, *, radius: float = 0.015, theme: Theme = HUMAN_THEME
) -> tuple[SpheresLayer, LinesLayer]:
    """The native skeleton (the family's joints and parents, MHR's included): a sphere of ``radius`` per joint in the
    theme's accent, named ``joints``, and a line per bone in its ink, named ``bones``."""
    joints = _numpy(state.joint_translations)
    radii = np.full(joints.shape[-2], radius, np.float32)
    return (
        SpheresLayer("joints", joints, radii=radii, colors=theme.accent, opacity=1.0),
        LinesLayer("bones", _bones(state)),
    )


def forces_layer(
    state: HumanState, frame_names: Sequence[str], *, forces: torch.Tensor, name: str = "forces", scale: float = 1e-3
) -> ArrowsLayer:
    """World ``forces (T, C, 3)`` (or ``(C, 3)``) applied at the model frames ``frame_names (C,)``, drawn as arrows
    (``scale`` metres per newton).

    Raises ``UnknownNameError`` for a frame the model does not have, ``TypeError`` when ``forces`` is not a
    tensor and ``ShapeError`` when it does not hold one 3-vector per frame name.
    """
    if not isinstance(forces, torch.Tensor):
        raise TypeError(f"forces must be a torch.Tensor, got {type(forces).__name__}")
    ids = torch.tensor([state.model.frame_id(frame) for frame in frame_names], dtype=torch.long, device=state.q.device)
    if forces.shape[-2:] != (len(ids), 3):
        raise ShapeError(
            f"forces must be (T, C, 3) or (C, 3) with C = len(frame_names) = {len(ids)}, got {tuple(forces.shape)}"
        )
    origins = state.kinematics.frame_translations.index_select(-2, ids)
    return ArrowsLayer(name, _numpy(origins), vectors=_numpy(forces), scale=scale)


def human_layers(
    state: HumanState,
    *,
    mesh: Literal["exact", "skinned"] | None = "exact",
    skeleton: bool = True,
    frames: bool = False,
    forces: Mapping[str, torch.Tensor] | None = None,
    theme: Theme = HUMAN_THEME,
    opacity: float = 1.0,
) -> list[Layer]:
    """Layers for a posed clip: ``mesh`` is ``"exact"`` (:func:`mesh_layer`), ``"skinned"`` (a ``RobotVisual`` with
    :func:`skin`, named ``mesh``) or ``None``; ``skeleton`` adds :func:`skeleton_layers`, ``frames`` a
    ``FramesLayer`` of every model frame (contact frames included).

    ``forces`` maps a model frame name to world forces ``(T, 3)`` or ``(3,)`` (a ``(3,)`` force holds at every step).
    Raises ``TypeError`` for a force that is not a tensor and ``ShapeError`` for one of another shape. Add the
    result with ``viewer.add(layers, tags="human")``.
    """
    layers: list[Layer] = []
    if mesh == "exact":
        layers.append(mesh_layer(state, color=theme.hues[0], opacity=opacity))
    elif mesh == "skinned":
        identity = state.identity
        layers.append(RobotVisual(identity.model, state.q, name="mesh", skin=skin(identity), opacity=opacity))
    elif mesh is not None:
        raise ArgumentError(f"mesh must be 'exact', 'skinned' or None, got {mesh!r}")
    if skeleton:
        layers.extend(skeleton_layers(state, theme=theme))
    if frames:
        layers.append(FramesLayer("frames", _numpy(state.frame_poses), axes_length=0.05))
    if forces:
        values = list(forces.values())
        if not all(isinstance(value, torch.Tensor) for value in values):
            raise TypeError("forces must map frame names to torch.Tensor values")
        steps = {value.shape[:-1] for value in values if value.ndim == 2}
        if any(value.shape[-1:] != (3,) or value.ndim > 2 for value in values) or len(steps) > 1:
            shapes = {frame: tuple(value.shape) for frame, value in forces.items()}
            raise ShapeError(f"forces must map frame names to (T, 3) or (3,) with one T, got {shapes}")
        stacked = torch.stack(torch.broadcast_tensors(*values), dim=-2)
        layers.append(forces_layer(state, list(forces), forces=stacked))
    return layers


@final
class HumanView:
    """Live human: the layers of one pose (``T = 1``) that :meth:`update` redraws with another pose or shape.

    Same options as :func:`human_layers` except ``forces``; ``viewer.add(view, tags=...)`` adds its layers. A
    skinned body keeps the rest shape it was built with: a view that follows shape changes draws the exact mesh.
    """

    def __init__(
        self,
        state: HumanState,
        *,
        mesh: Literal["exact", "skinned"] | None = "exact",
        skeleton: bool = True,
        frames: bool = False,
        theme: Theme = HUMAN_THEME,
        opacity: float = 1.0,
    ) -> None:
        self.layers = human_layers(state, mesh=mesh, skeleton=skeleton, frames=frames, theme=theme, opacity=opacity)
        if any(layer.nsteps != 1 for layer in self.layers):
            raise ShapeError("HumanView takes the state of a single pose, not a clip: use human_layers for a clip")

    def __iter__(self) -> Iterator[Layer]:
        return iter(self.layers)

    def __repr__(self) -> str:
        return f"HumanView(layers={[layer.name for layer in self.layers]})"

    def update(self, state: HumanState) -> None:
        """Draw the pose of ``state`` (a single pose of the drawn body) in the layers, and in the viewer they are in.

        Raises ``ShapeError``, and draws nothing, for a clip or the state of a body of another size.
        """
        writes: list[tuple[np.ndarray, torch.Tensor | np.ndarray]] = []
        for layer in self.layers:
            if isinstance(layer, MeshLayer):
                writes.append((layer.vertices, state.vertices))
            elif isinstance(layer, RobotVisual):
                writes.append((_numpy(layer.q), state.q))
            elif isinstance(layer, SpheresLayer):
                writes.append((layer.centers, state.joint_translations))
            elif isinstance(layer, LinesLayer):
                writes.append((layer.points, _bones(state)))
            elif isinstance(layer, FramesLayer):
                writes.append((layer.poses, state.frame_poses))
        for drawn, value in writes:
            if tuple(value.shape) not in (drawn.shape, drawn.shape[1:]):
                raise ShapeError(
                    f"HumanView.update takes a single pose of the drawn body: got {tuple(value.shape)} where the view "
                    f"draws {drawn.shape[1:]}"
                )
        viewer = self.layers[0].viewer if self.layers else None
        with nullcontext() if viewer is None else viewer.server.atomic():  # the browser draws the pose in one go
            for layer, (drawn, value) in zip(self.layers, writes, strict=True):
                if isinstance(layer, RobotVisual):
                    layer.update(state.q)  # forward kinematics, then the visual redraws itself
                else:
                    drawn[...] = value if isinstance(value, np.ndarray) else _numpy(value)
                    layer.redraw()


__all__ = [
    "HUMAN_THEME",
    "HumanView",
    "forces_layer",
    "human_layers",
    "mesh_layer",
    "skeleton_layers",
    "skin",
]
