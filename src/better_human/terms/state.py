"""The cached values the terms read: a baked identity and the posed state."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from better_robot.exceptions import ArgumentError, UnknownNameError
from better_robot.optim import TermFn

from better_human.core import HumanModel, HumanState, Identity


def baked_identity(human: HumanModel, *groups: str) -> TermFn:
    """Cached value: ``human.bake(**groups)`` from variables named after the identity groups (default: all).

    Raises :class:`~better_robot.exceptions.UnknownNameError` for a group the family does not have.
    """
    for name in groups:
        if name not in human.spec.identity_bases:
            raise UnknownNameError("identity group", name, list(human.spec.identity_bases))
    names = groups or tuple(human.spec.identity_groups)
    return TermFn(fn=lambda *values: human.bake(**dict(zip(names, values, strict=True))), reads=names)


def human_state(
    identity: Identity | None = None, *, q: str = "q", streams: Sequence[str] = (), identity_name: str | None = None
) -> TermFn:
    """Cached value: ``identity.forward(q, streams)``.

    Pass ``identity`` for a fixed subject, or ``identity_name`` to read a cached or variable identity (see
    :func:`baked_identity`). ``streams`` names the variables or data of per-frame groups (``"expression"``); a group
    the family does not have raises :class:`~better_robot.exceptions.UnknownNameError` here when ``identity`` is
    given, else on the first evaluation, when the identity is known. Raises
    :class:`~better_robot.exceptions.ArgumentError` unless exactly one of ``identity`` and ``identity_name`` is given.
    """
    if (identity is None) == (identity_name is None):
        raise ArgumentError("human_state: give exactly one of identity and identity_name")
    streams = tuple(streams)
    if identity is not None:
        known = identity.human.spec.stream_bases
        for name in streams:
            if name not in known:
                raise UnknownNameError("stream group", name, list(known))

    def fixed(pose: torch.Tensor, *values: torch.Tensor) -> HumanState:
        return identity.forward(pose, dict(zip(streams, values, strict=True)))

    def variable(subject: Identity, pose: torch.Tensor, *values: torch.Tensor) -> HumanState:
        return subject.forward(pose, dict(zip(streams, values, strict=True)))

    if identity is not None:
        return TermFn(fn=fixed, reads=(q, *streams))
    return TermFn(fn=variable, reads=(identity_name, q, *streams))
