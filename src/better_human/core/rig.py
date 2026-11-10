"""``Rig``: how identity turns into joint placements and how a configuration feeds the corrective.

A rig is the one behavioural part of a :class:`~better_human.core.spec.FamilySpec`. The families define three:
``RegressedRig`` (SMPL family), ``MhrRig`` and ``SomaRig``. A rig implements the four abstract members; the three
hooks below have defaults for rigs whose streams move no joint and whose native joint frames are BetterRobot's.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, final

import torch
from better_robot import Model
from better_robot.exceptions import ArgumentError

if TYPE_CHECKING:
    from better_robot.kinematics import KinematicState

    from .spec import FamilySpec


@final
@dataclass(frozen=True)
class RigBake:
    """Identity-dependent kinematic values of one bake; leading dims are the identity batch ``I...``."""

    joint_placements: torch.Tensor  #: ``(I..., nj, 7)`` for the BetterRobot model (joint 0 is the universe).
    rest_vertices: torch.Tensor  #: ``(I..., V, 3)`` skinning rest mesh, root at the origin.
    rest_joint_translations: torch.Tensor  #: ``(I..., J, 3)`` native joints in the skinning rest frame.
    #: ``(I..., J, 3, 4)`` affine from the rest frame into each native joint's frame; skinning composes the world joint
    #: transforms with it (``skin_affines``).
    inverse_bind_affines: torch.Tensor
    root_offset: torch.Tensor  #: ``(I..., 3)`` root from the native origin; ``transl + root_offset`` places the root.


class Rig(ABC):
    """The behaviour of a family: its BetterRobot model, its bake, its joint numbering and its corrective features."""

    @abstractmethod
    def build_model(self, spec: FamilySpec, *, frames: Sequence[tuple[str, str, torch.Tensor]]) -> Model:
        """BetterRobot model in ``spec``'s dtype and device, bodies named after ``spec.joint_names``.

        ``frames`` are operational frames ``(name, parent joint name, placement (7,))`` to attach; a rig that cannot
        place them raises.
        """

    @abstractmethod
    def bake(
        self, spec: FamilySpec, model: Model, *, shaped_vertices: torch.Tensor, identity: Mapping[str, torch.Tensor]
    ) -> RigBake:
        """The bake of ``shaped_vertices (I..., V, 3)`` (template plus identity blend shapes) and the group values."""

    @abstractmethod
    def joint_ids(self, spec: FamilySpec, model: Model) -> torch.Tensor:
        """``(J,)`` BetterRobot joint id of each native joint."""

    @abstractmethod
    def corrective_features(
        self, spec: FamilySpec, model: Model, *, q: torch.Tensor, identity: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        """``(..., P)`` features the corrective reads, from ``q (..., nq)``.

        ``identity`` holds the identity group values ``(I..., n)`` of the bake, so a rig whose corrective depends on
        shape (STAR, SUPR) can read them.
        """

    def joint_transforms(self, spec: FamilySpec, model: Model, *, kinematics: KinematicState) -> torch.Tensor:
        """``(..., J, 3, 4)`` world ``[R | p]`` of the native joints: by default BetterRobot's joint transforms.

        Positions are BetterRobot's joint translations; a rig only post-processes rotations BetterRobot cannot model.
        """
        return kinematics.joint_transforms.index_select(-3, self.joint_ids(spec, model))

    def stream_joint_basis(self, spec: FamilySpec, name: str) -> torch.Tensor | None:
        """``(J, 3, n)`` joint motion of stream group ``name``; ``None`` (the default) when it moves no joint.

        A rig whose streams move joints (it returns a basis here and placements from :meth:`placements_from_rest`)
        binds every joint at its rest joint translation, so a stream shift of the rest joints shifts the binds.
        """
        return None

    def placements_from_rest(
        self, spec: FamilySpec, model: Model, *, rest_joint_translations: torch.Tensor
    ) -> torch.Tensor | None:
        """``(..., nj, 7)`` placements for shifted rest joints; ``None`` (the default) when streams move no joint."""
        return None


def gravity(up_axis: str, magnitude: float = 9.81) -> tuple[float, float, float]:
    """World gravity ``(3,)`` pointing against ``up_axis`` (``"+y"``, ``"-z"``, ...).

    Raises :class:`~better_robot.exceptions.ArgumentError` for another ``up_axis``.
    """
    if up_axis not in _UP_AXES:
        raise ArgumentError(f"up_axis={up_axis!r} is not one of {list(_UP_AXES)}; pass a sign and an axis, e.g. '+y'")
    sign, axis = up_axis[0], "xyz".index(up_axis[1])
    value = -magnitude if sign == "+" else magnitude
    return tuple(value if i == axis else 0.0 for i in range(3))


_UP_AXES = ("+x", "-x", "+y", "-y", "+z", "-z")

__all__ = ["Rig", "RigBake", "gravity"]
