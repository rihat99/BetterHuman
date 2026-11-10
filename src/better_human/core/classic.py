"""Classic parameters, the upstream libraries' own inputs, to and from ``(identity, q, streams)`` (decision record 0004).

Every family's spec carries one :class:`ClassicFormat` (``FamilySpec.classic``). :func:`from_classic` and
:func:`to_classic` are the one way in and out for every family: they check every tensor in one place, so a format maps
checked tensors only.
"""

from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import torch
from better_robot.exceptions import ArgumentError, ShapeError, UnknownNameError

from .human_model import HumanModel, Identity, _checked, _checked_groups

if TYPE_CHECKING:
    from .spec import FamilySpec


class ClassicFormat(ABC):
    """How one family's ``*Classic`` dataclass maps to ``(identity, q, streams)`` and back (decision record 0004).

    ``classic_type`` is that dataclass; each of its fields is a tensor or ``None`` (zero). The hooks see checked
    tensors only.
    """

    classic_type: type

    @abstractmethod
    def field_shapes(self, spec: FamilySpec) -> dict[str, tuple[int, ...]]:
        """Trailing shape of every field a model of ``spec`` takes; a field left out was not loaded and stays ``None``.

        Raises :class:`~better_robot.exceptions.ArgumentError` when the format cannot map ``spec``'s joints.
        """

    @abstractmethod
    def from_fields(
        self, human: HumanModel, fields: Mapping[str, torch.Tensor]
    ) -> tuple[Identity, torch.Tensor, dict[str, torch.Tensor]]:
        """``(identity, q (..., nq), streams)`` of ``fields`` ``(..., *shape)``; an omitted field arrives as zeros."""

    @abstractmethod
    def to_fields(
        self, identity: Identity, q: torch.Tensor, *, streams: Mapping[str, torch.Tensor]
    ) -> dict[str, torch.Tensor | None]:
        """The fields of ``q (..., nq)`` and every stream group ``(..., n)``, all at one batch that covers the identity's."""


def from_classic(human: HumanModel, classic: Any) -> tuple[Identity, torch.Tensor, dict[str, torch.Tensor]]:
    """``(identity, q (..., nq), streams)`` of a family's ``*Classic`` parameters; the identity is baked from them.

    An omitted field is zero. Raises :class:`~better_robot.exceptions.ArgumentError` when ``human`` is not a
    :class:`HumanModel`, when ``classic`` is not the family's ``*Classic`` or sets a field the model was loaded without
    (``hands=False``, ``face=False``, ``expression=False``); :class:`~better_robot.exceptions.ShapeError`,
    :class:`~better_robot.exceptions.DtypeMismatchError` or :class:`~better_robot.exceptions.DeviceMismatchError`
    for a field of another trailing shape, dtype or device, and ``ShapeError`` for fields whose batch shapes do not
    broadcast; ``TypeError`` for a field that is not a floating tensor.
    """
    if not isinstance(human, HumanModel):
        raise ArgumentError(f"from_classic takes a HumanModel first, got {type(human).__name__}")
    spec = human.spec
    classic_format = spec.classic
    if not isinstance(classic, classic_format.classic_type):
        raise ArgumentError(
            f"from_classic: a {spec.name} model takes {classic_format.classic_type.__name__}, "
            f"got {type(classic).__name__}"
        )
    shapes = classic_format.field_shapes(spec)
    exemplar = spec.template_vertices
    fields, batches = {}, {}
    for name in (field.name for field in dataclasses.fields(classic)):
        value = getattr(classic, name)
        if name in shapes:
            if value is None:
                fields[name] = exemplar.new_zeros(shapes[name])
                continue
            fields[name] = _checked(name, value, shapes[name], exemplar)
            batches[name] = value.shape[: value.dim() - len(shapes[name])]
        elif value is not None:
            raise ArgumentError(
                f"from_classic: this {spec.name} model was loaded without {name} (hands=False, face=False or "
                "expression=False); leave it None"
            )
    _broadcast("from_classic", batches)
    return classic_format.from_fields(human, fields)


def to_classic(identity: Identity, q: torch.Tensor, *, streams: Mapping[str, torch.Tensor] | None = None) -> Any:
    """The family's ``*Classic`` parameters of ``identity`` posed at ``q (..., nq)`` with stream groups ``(..., n)``.

    Every field comes back at the batch of ``q``, the streams and the identity broadcast together; an omitted stream
    group is zero, and a field the model was loaded without is ``None``. Raises
    :class:`~better_robot.exceptions.ArgumentError` when ``identity`` is not an :class:`Identity`,
    :class:`~better_robot.exceptions.UnknownNameError` for a stream group the family does not have, and the errors of
    :func:`from_classic` for ``q`` or a stream of another shape, dtype or device, or whose batch shapes do not
    broadcast with each other and the identity's.
    """
    if not isinstance(identity, Identity):
        raise ArgumentError(
            f"to_classic takes an Identity first (HumanModel.bake or from_classic returns one), "
            f"got {type(identity).__name__}"
        )
    spec = identity.human.spec
    exemplar = spec.template_vertices
    streams = dict(streams or {})
    for name in streams:
        if name not in spec.stream_bases:
            raise UnknownNameError("stream group", name, list(spec.stream_bases))
    q = _checked("q", q, (identity.model.nq,), exemplar)
    streams = _checked_groups(streams, spec.stream_groups, exemplar, require_all=False)
    batch = _broadcast(
        "to_classic",
        {
            "q": q.shape[:-1],
            "identity": identity.root_offset.shape[:-1],
            **{name: value.shape[:-1] for name, value in streams.items()},
        },
    )
    streams = {
        name: streams[name].expand(*batch, width) if name in streams else exemplar.new_zeros((*batch, width))
        for name, width in spec.stream_groups.items()
    }
    fields = spec.classic.to_fields(identity, q.expand(*batch, identity.model.nq), streams=streams)
    return spec.classic.classic_type(**fields)


def _broadcast(function: str, batches: Mapping[str, torch.Size]) -> torch.Size:
    """The leading batch shapes ``batches`` broadcast together; :class:`~better_robot.exceptions.ShapeError` naming
    each one when they do not."""
    try:
        return torch.broadcast_shapes(*batches.values())
    except RuntimeError:
        listed = ", ".join(f"{name} {tuple(shape)}" for name, shape in batches.items())
        raise ShapeError(
            f"{function}: the batch shapes {listed} do not broadcast; give them one batch shape, or 1 on the axes to "
            "broadcast"
        ) from None


__all__ = ["ClassicFormat", "from_classic", "to_classic"]
