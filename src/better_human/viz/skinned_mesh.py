"""Live skinned-mesh mode for the BetterRobot viewer."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import torch
from better_robot.viewer import RenderContext

if TYPE_CHECKING:
    from better_robot import Data, Model

    from better_human.core import ParametricBody


class SkinnedMeshMode:
    """Render one parametric body's posed vertices as a deforming mesh."""

    name: ClassVar[str] = "Skinned mesh"
    description: ClassVar[str] = "Live pose-corrected and skinned human surface"

    def __init__(
        self,
        body: ParametricBody,
        *,
        rgba: tuple[float, float, float, float] = (0.65, 0.78, 0.95, 1.0),
    ) -> None:
        self._body = body
        self._rgba = rgba
        self._context: RenderContext | None = None

    @classmethod
    def is_available(cls, model: Model, data: Data) -> bool:
        """Return true because availability is established by construction."""
        return True

    def attach(self, context: RenderContext, model: Model, data: Data) -> None:
        """Upload the shaped rest mesh once and bind the render context."""
        if model is not self._body.robot:
            raise ValueError("SkinnedMeshMode must attach to the body.robot it was constructed with")
        self._context = context
        self._upload(self._body.values.v_shaped)

    def update(self, data: Data) -> None:
        """Skin vertices from existing FK data and replace the displayed mesh."""
        if self._context is None:
            return
        self._upload(self._body.vertices_from_data(data))

    def set_visible(self, visible: bool) -> None:
        """Show or hide the skinned surface."""
        if self._context is not None:
            self._context.backend.set_visible(self._node_name, visible)

    def detach(self) -> None:
        """Remove the skinned surface from the backend."""
        if self._context is None:
            return
        self._context.backend.remove(self._node_name)
        self._context = None

    @property
    def _node_name(self) -> str:
        if self._context is None:
            raise RuntimeError("SkinnedMeshMode is not attached")
        return f"{self._context.namespace}/surface"

    def _upload(self, vertices: torch.Tensor) -> None:
        if self._context is None:
            return
        selected = _select_batch(vertices, self._context.batch_index)
        self._context.backend.add_mesh(
            self._node_name,
            selected,
            self._body.structure.faces,
            rgba=self._rgba,
        )


def _select_batch(vertices: torch.Tensor, batch_index: int) -> torch.Tensor:
    """Select one surface from arbitrary leading batch dimensions."""
    if vertices.ndim == 2:
        return vertices
    return vertices.reshape(-1, *vertices.shape[-2:])[batch_index]


__all__ = ["SkinnedMeshMode"]
