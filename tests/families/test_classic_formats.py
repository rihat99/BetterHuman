"""Decision record 0004: one round-trip checker runs over every family's ``ClassicFormat``, both ways.

Oracle: the input itself. ``bh.to_classic`` of ``bh.from_classic(human, classic)`` gives back ``classic`` (an omitted
field as zeros, a field the model was loaded without as ``None``), and ``bh.from_classic`` of that classic gives back
the same identity, ``q`` and streams. Random fields come from a generator seeded in the case id. FLAME and MANO run on
synthetic weight files (full tier); the other families, and MANO's real hand PCA, read the licensed weights (release
tier).
"""

from __future__ import annotations

import dataclasses
import inspect
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import torch
from better_robot.exceptions import ArgumentError

import better_human as bh
from better_human.core import ClassicFormat
from better_human.families import FLAMEClassic, MHRClassic, SMPLFamilyFormat, SOMAClassic
from better_human.io import weight_path
from tests.support.synthetic import flame_spec

BATCH = (2,)
F64 = torch.float64


def _mano_file(directory: Path) -> Path:
    """A MANO-shaped file: 16 joints (the wrist and five three-joint fingers), 50 vertices, an orthogonal hand PCA."""
    rng = np.random.default_rng(0)
    num_vertices, num_joints = 50, 16
    weights, regressor = rng.random((num_vertices, num_joints)), rng.random((num_joints, num_vertices))
    parents = [4294967295] + [0 if joint % 3 == 1 else joint - 1 for joint in range(1, num_joints)]
    path = directory / "MANO_RIGHT.npz"
    np.savez(
        path,
        v_template=rng.normal(size=(num_vertices, 3)) * 0.1,
        shapedirs=rng.normal(size=(num_vertices, 3, 10)) * 1e-3,
        posedirs=rng.normal(size=(num_vertices, 3, (num_joints - 1) * 9)) * 1e-3,
        J_regressor=regressor / regressor.sum(axis=1, keepdims=True),
        weights=weights / weights.sum(axis=1, keepdims=True),
        kintree_table=np.array([parents, list(range(num_joints))], dtype=np.uint32),
        f=rng.integers(0, num_vertices, size=(80, 3)).astype(np.uint32),
        hands_components=np.linalg.qr(rng.normal(size=(45, 45)))[0],
        hands_mean=rng.normal(size=45) * 0.1,
    )
    return path


def _licensed(loader: Callable[[], bh.FamilySpec]) -> Callable[[Path], bh.FamilySpec]:
    return lambda _: loader()


#: case id -> (spec of the case, the format it must carry, absolute tolerance of the round trip). The tolerances are
#: those of the per-family round-trip tests this checker replaces: float64 rounding, the PCA pseudo-inverse (SMPL-X
#: 1e-8, MANO 1e-9), float32 (MHR, SOMA 1e-6).
CASES: dict[str, tuple[Callable[[Path], bh.FamilySpec], type[ClassicFormat], float]] = {
    "flame": (flame_spec, SMPLFamilyFormat, 1e-10),
    "flame-without-jaw": (lambda path: bh.families.drop_joints(flame_spec(path), ["jaw"]), SMPLFamilyFormat, 1e-10),
    "mano-pca6": (lambda path: bh.families.mano(_mano_file(path), num_pca_comps=6, dtype=F64), SMPLFamilyFormat, 1e-9),
    "mano-flat": (
        lambda path: bh.families.mano(_mano_file(path), flat_hand_mean=True, dtype=F64),
        SMPLFamilyFormat,
        1e-9,
    ),
    "mano-pca6-weights": (
        _licensed(lambda: bh.families.mano(weight_path("mano", "MANO_RIGHT.npz"), num_pca_comps=6, dtype=F64)),
        SMPLFamilyFormat,
        1e-9,
    ),
    "smpl": (
        _licensed(lambda: bh.families.smpl(weight_path("smpl", "SMPL_NEUTRAL.npz"), dtype=F64)),
        SMPLFamilyFormat,
        1e-10,
    ),
    "smplh": (
        _licensed(lambda: bh.families.smplh(weight_path("smplh/smplh/neutral", "model.npz"), dtype=F64)),
        SMPLFamilyFormat,
        1e-10,
    ),
    "smplx-pca6": (
        _licensed(lambda: bh.families.smplx(weight_path("smplx", "SMPLX_NEUTRAL.npz"), num_pca_comps=6, dtype=F64)),
        SMPLFamilyFormat,
        1e-8,
    ),
    "smplx-pca6-flat": (
        _licensed(
            lambda: bh.families.smplx(
                weight_path("smplx", "SMPLX_NEUTRAL.npz"), num_pca_comps=6, flat_hand_mean=True, dtype=F64
            )
        ),
        SMPLFamilyFormat,
        1e-8,
    ),
    "smplx-body-only": (
        _licensed(
            lambda: bh.families.smplx(weight_path("smplx", "SMPLX_NEUTRAL.npz"), hands=False, face=False, dtype=F64)
        ),
        SMPLFamilyFormat,
        1e-10,
    ),
    "mhr": (
        _licensed(lambda: bh.families.mhr(weight_path("MHR/converted", "mhr_lod6.npz"), lod=6, correctives=False)),
        bh.families.MHRFormat,
        1e-6,
    ),
    "soma-float32": (
        _licensed(lambda: bh.families.soma(weight_path("SOMA", "soma_mid.npz"), correctives=False)),
        bh.families.SOMAFormat,
        1e-6,
    ),
    "soma-float64": (
        _licensed(lambda: bh.families.soma(weight_path("SOMA", "soma_mid.npz"), correctives=False, dtype=F64)),
        bh.families.SOMAFormat,
        1e-12,
    ),
}
SYNTHETIC = ("flame", "flame-without-jaw", "mano-pca6", "mano-flat")


def check_round_trip(human: bh.HumanModel, generator: torch.Generator, atol: float) -> None:
    """``to_classic`` inverts ``from_classic`` on a random classic of ``human``'s format, and the other way round.

    Every field the model takes is drawn ``(*BATCH, *shape)`` and comes back, an identity field exactly; every field of
    an empty classic comes back as zeros; a field the model was loaded without must raise when set and come back
    ``None``; another family's classic must raise.
    """
    spec = human.spec
    classic_type = spec.classic.classic_type
    shapes = spec.classic.field_shapes(spec)
    names = [field.name for field in dataclasses.fields(classic_type)]
    values = {
        name: 0.3 * torch.randn(*BATCH, *shapes[name], generator=generator, dtype=spec.dtype)
        for name in names
        if name in shapes
    }
    identity, q, streams = bh.from_classic(human, classic_type(**values))
    back = bh.to_classic(identity, q, streams=streams)
    empty_identity, empty_q, empty_streams = bh.from_classic(human, classic_type())
    empty = bh.to_classic(empty_identity, empty_q, streams=empty_streams)
    assert type(back) is classic_type and type(empty) is classic_type
    for name in names:
        if name not in shapes:
            assert getattr(back, name) is None and getattr(empty, name) is None, (
                f"{name}: the model was loaded without it, so to_classic returns None"
            )
            with pytest.raises(ArgumentError, match=f"loaded without {name}"):
                bh.from_classic(human, classic_type(**{name: torch.zeros(3, dtype=spec.dtype)}))
            continue
        torch.testing.assert_close(
            getattr(back, name),
            values[name],
            rtol=0,
            atol=0 if name in identity.values else atol,
            msg=lambda text, name=name: f"{name}: {text}",
        )
        torch.testing.assert_close(
            getattr(empty, name),
            torch.zeros(shapes[name], dtype=spec.dtype),
            rtol=0,
            atol=atol,
            msg=lambda text, name=name: f"{name} omitted: {text}",
        )

    identity_back, q_back, streams_back = bh.from_classic(human, back)
    torch.testing.assert_close(q_back, q, rtol=0, atol=atol)
    assert streams_back.keys() == streams.keys()
    for name, value in streams.items():
        torch.testing.assert_close(streams_back[name], value, rtol=0, atol=atol)
    for name, value in identity.values.items():
        torch.testing.assert_close(identity_back.values[name], value, rtol=0, atol=atol)

    wrong = MHRClassic if classic_type is not MHRClassic else SOMAClassic
    with pytest.raises(ArgumentError, match=f"takes {classic_type.__name__}, got {wrong.__name__}"):
        bh.from_classic(human, wrong())


@pytest.mark.parametrize(
    "case",
    [pytest.param(case, marks=() if case in SYNTHETIC else pytest.mark.licensed) for case in CASES],
)
def test_classic_round_trip(tmp_path: Path, case: str) -> None:
    load, expected_format, atol = CASES[case]
    human = bh.HumanModel.from_spec(load(tmp_path), density=None)
    assert type(human.spec.classic) is expected_format
    check_round_trip(human, torch.Generator().manual_seed(0), atol)


def test_the_cases_cover_every_family_and_format() -> None:
    """Every family loader and every exported ``ClassicFormat`` has a case (6.7: the checker sees every format)."""
    loaders = {
        name
        for name in bh.families.__all__
        if inspect.isfunction(getattr(bh.families, name))
        and next(iter(inspect.signature(getattr(bh.families, name)).parameters)) == "path"
    }
    assert {case.split("-")[0] for case in CASES} == loaders
    formats = {getattr(bh.families, name) for name in bh.families.__all__}
    formats = {value for value in formats if inspect.isclass(value) and issubclass(value, ClassicFormat)}
    assert {expected for _, expected, _ in CASES.values()} == formats


def test_smpl_family_axis_angles_keep_the_quaternion_sign(tmp_path: Path) -> None:
    """An angle beyond pi comes back as given, and a negated joint quaternion of ``q`` survives the round trip."""
    human = bh.HumanModel.from_spec(flame_spec(tmp_path), density=None)
    classic = FLAMEClassic(global_orient=torch.tensor([4.0, 0.0, 0.0], dtype=F64))
    identity, q, streams = bh.from_classic(human, classic)
    assert q[6] < 0, "4 rad is beyond pi: the root quaternion's scalar is negative"
    back = bh.to_classic(identity, q, streams=streams)
    torch.testing.assert_close(back.global_orient, classic.global_orient, rtol=0, atol=1e-12)
    negated = torch.cat((q[:7], -q[7:]))
    _, q_back, _ = bh.from_classic(human, bh.to_classic(identity, negated, streams=streams))
    torch.testing.assert_close(q_back, negated, rtol=0, atol=1e-12)


@pytest.mark.parametrize("num_pca_comps", (0, 46, 6.0, True))
def test_the_hand_pca_width_is_checked_at_load(tmp_path: Path, num_pca_comps: object) -> None:
    for loader in (bh.families.smplx, bh.families.mano):
        with pytest.raises(ArgumentError, match=r"num_pca_comps must be an int in \[1, 45\]"):
            loader(tmp_path / "unread.npz", num_pca_comps=num_pca_comps)
