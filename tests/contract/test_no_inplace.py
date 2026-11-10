"""Principles 1.4 and 2.6: no public call mutates a tensor its caller holds.

Each case calls one public entry point on the random-weight SMPL-X archive: ``HumanModel.from_spec``, ``bake``,
``forward`` with a stream, every public property and method of ``HumanState``, ``from_classic``, ``to_classic`` and
the function of every term of ``better_human.terms``. Every tensor reachable from the arguments (the pose, the
streams, the classic fields, the term values, and the tensor fields of the records passed in, their memos included)
keeps its version counter and its values.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping

import pytest
import torch
from better_robot.kinematics import integrate

import better_human as bh
from better_human import terms
from tests.support.random_archives import SMPLX_FRAMES, random_weight_smplx, random_weight_spec


def _betas(*batch: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    return 0.3 * torch.randn(*batch, random_weight_smplx().spec.identity_groups["betas"], generator=generator)


def _pose(*batch: int) -> torch.Tensor:
    model = random_weight_smplx().model
    generator = torch.Generator().manual_seed(1)
    return integrate(model, model.q_neutral, 0.2 * torch.randn(*batch, model.nv, generator=generator))


def _streams(*batch: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(2)
    width = random_weight_smplx().spec.stream_groups["expression"]
    return {"expression": 0.3 * torch.randn(*batch, width, generator=generator)}


def _identity() -> bh.Identity:
    return random_weight_smplx().bake(betas=_betas())


def _state() -> bh.HumanState:
    return _identity().forward(_pose(2), _streams(2))


def _term(term: Callable[[], object], *values: object) -> tuple[Callable[..., object], tuple[object, ...]]:
    return term().fn, values


STATE_MEMBERS: dict[str, Callable[[bh.HumanState], object]] = {
    "joint_poses": lambda state: state.joint_poses,
    "joint_translations": lambda state: state.joint_translations,
    "joint_transforms": lambda state: state.joint_transforms,
    "skin_affines": lambda state: state.skin_affines,
    "corrective_feature": lambda state: state.corrective_feature,
    "rest_vertices": lambda state: state.rest_vertices,
    "vertices": lambda state: state.vertices,
    "points": lambda state: state.points(torch.tensor([0, 10, 20])),
    "landmarks": lambda state: state.landmarks(),
    "frame_poses": lambda state: state.frame_poses,
    "contact_frame_poses": lambda state: state.contact_frame_poses(SMPLX_FRAMES),
}
TERMS: dict[str, Callable[[], tuple[Callable[..., object], tuple[object, ...]]]] = {
    "baked_identity": lambda: _term(lambda: terms.baked_identity(random_weight_smplx()), _betas(2)),
    "human_state": lambda: _term(
        lambda: terms.human_state(_identity(), streams=["expression"]), _pose(2), _streams(2)["expression"]
    ),
    "joints": lambda: _term(lambda: terms.joints(target="goal"), _state(), torch.zeros(2, 55, 3)),
    "points": lambda: _term(lambda: terms.points([0, 10], target="goal"), _state(), torch.zeros(2, 2, 3)),
    "landmarks": lambda: _term(lambda: terms.landmarks(target="goal"), _state(), torch.zeros(2, 21, 3)),
    "reprojection": lambda: _term(
        lambda: terms.reprojection(target="goal", confidence="weight"),
        _state(),
        torch.zeros(2, 21, 2),
        torch.eye(3, 4).expand(2, 3, 4) + torch.tensor([0.0, 0.0, 0.0, 5.0]),
        torch.ones(2, 21),
    ),
    "ground": lambda: _term(terms.ground, _state()),
    "pose_prior": lambda: _term(lambda: terms.pose_prior(_identity()), _pose(2)),
    "shape_prior": lambda: _term(terms.shape_prior, _betas(2)),
    "root_wrench": lambda: _term(
        lambda: terms.root_wrench(_identity(), list(SMPLX_FRAMES), dt=0.01), _pose(1, 4), torch.ones(1, 4, 2, 3)
    ),
}
CASES: dict[str, Callable[[], tuple[Callable[..., object], tuple[object, ...]]]] = {
    "HumanModel.from_spec": lambda: (
        lambda spec: bh.HumanModel.from_spec(spec, density=None, contact_frames=SMPLX_FRAMES),
        (random_weight_spec("smplx"),),
    ),
    "HumanModel.bake": lambda: (lambda human, betas: human.bake(betas=betas), (random_weight_smplx(), _betas(3))),
    "Identity.forward": lambda: (
        lambda identity, q, streams: identity.forward(q, streams),
        (_identity(), _pose(3), _streams(3)),
    ),
    **{f"HumanState.{name}": lambda member=member: (member, (_state(),)) for name, member in STATE_MEMBERS.items()},
    "from_classic": lambda: (
        bh.from_classic,
        (random_weight_smplx(), bh.to_classic(_identity(), _pose(2), streams=_streams(2))),
    ),
    "to_classic": lambda: (
        lambda identity, q, streams: bh.to_classic(identity, q, streams=streams),
        (_identity(), _pose(2), _streams(2)),
    ),
    **{f"terms.{name}": case for name, case in TERMS.items()},
}


def _tensors(value: object, seen: set[int]) -> list[torch.Tensor]:
    """Every tensor reachable from ``value`` through tensors, mappings, sequences and dataclass fields, each once."""
    if id(value) in seen:
        return []
    seen.add(id(value))
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, Mapping):
        children = list(value.values())
    elif isinstance(value, tuple | list):
        children = list(value)
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        children = [getattr(value, item.name) for item in dataclasses.fields(value)]
    else:
        return []
    return [tensor for child in children for tensor in _tensors(child, seen)]


def test_every_state_member_and_term_has_a_case() -> None:
    fields = {item.name for item in dataclasses.fields(bh.HumanState)}
    members = {name for name in vars(bh.HumanState) if name[0] != "_" and name not in fields}
    assert members == STATE_MEMBERS.keys()
    assert set(terms.__all__) == TERMS.keys()


@pytest.mark.parametrize("name", sorted(CASES))
def test_public_call_leaves_the_callers_tensors_untouched(name: str) -> None:
    function, arguments = CASES[name]()
    inputs = _tensors(arguments, set())
    assert inputs, f"{name}: no tensor reachable from its arguments"
    before = [(tensor._version, tensor.detach().clone()) for tensor in inputs]
    function(*arguments)
    changed = [
        tuple(tensor.shape)
        for tensor, (version, values) in zip(inputs, before, strict=True)
        if tensor._version != version or not torch.equal(tensor, values)
    ]
    assert not changed, f"{name} wrote into {len(changed)} of its callers' tensors, shapes {changed[:5]}"
