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
from better_robot.exceptions import ArgumentError, ShapeError

import better_human as bh
from better_human.core import ClassicFormat
from better_human.families import FLAMEClassic, MHRClassic, SMPLFamilyFormat, SOMAClassic
from better_human.io import weight_path
from tests.support.synthetic import flame_spec

#: Leading batch shapes the drawn fields take in turn: two axes, broadcast between the fields to ``BROADCAST``.
BATCHES = ((2, 1), (3,))
BROADCAST = (2, 3)
F64, F32 = torch.float64, torch.float32


def _mano_file(directory: Path) -> Path:
    """A MANO-shaped file: 16 joints (the wrist and five three-joint fingers), 778 vertices, an orthogonal hand PCA."""
    rng = np.random.default_rng(0)
    num_vertices, num_joints = 778, 16
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
#: 1e-8, MANO 1e-9), float32 (MHR, SOMA 1e-6); the SMPL family's float32 cases take the same 1e-6.
CASES: dict[str, tuple[Callable[[Path], bh.FamilySpec], type[ClassicFormat], float]] = {
    "flame": (flame_spec, SMPLFamilyFormat, 1e-10),
    "flame-without-jaw": (lambda path: bh.families.drop_joints(flame_spec(path), ["jaw"]), SMPLFamilyFormat, 1e-10),
    "flame-float32": (lambda path: flame_spec(path, dtype=F32), SMPLFamilyFormat, 1e-6),
    "mano-pca6": (lambda path: bh.families.mano(_mano_file(path), num_pca_comps=6, dtype=F64), SMPLFamilyFormat, 1e-9),
    "mano-pca6-float32": (
        lambda path: bh.families.mano(_mano_file(path), num_pca_comps=6, dtype=F32),
        SMPLFamilyFormat,
        1e-6,
    ),
    "mano-flat": (
        lambda path: bh.families.mano(_mano_file(path), flat_hand_mean=True, dtype=F64),
        SMPLFamilyFormat,
        1e-9,
    ),
    "mano-without-index3": (
        lambda path: bh.families.drop_joints(bh.families.mano(_mano_file(path), dtype=F64), ["index3"]),
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
    "smplx-pca6-float32": (
        _licensed(lambda: bh.families.smplx(weight_path("smplx", "SMPLX_NEUTRAL.npz"), num_pca_comps=6, dtype=F32)),
        SMPLFamilyFormat,
        1e-6,
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
    "smplx-without-left_index3": (
        _licensed(
            lambda: bh.families.drop_joints(
                bh.families.smplx(weight_path("smplx", "SMPLX_NEUTRAL.npz"), dtype=F64), ["left_index3"]
            )
        ),
        SMPLFamilyFormat,
        1e-10,
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
SYNTHETIC = (
    "flame",
    "flame-without-jaw",
    "flame-float32",
    "mano-pca6",
    "mano-pca6-float32",
    "mano-flat",
    "mano-without-index3",
)


def check_round_trip(human: bh.HumanModel, generator: torch.Generator, atol: float) -> None:
    """``to_classic`` inverts ``from_classic`` on a random classic of ``human``'s format, and the other way round.

    Every field the model takes is drawn ``(*batch, *shape)``, ``batch`` taking each of ``BATCHES`` in turn, and comes
    back at ``BROADCAST``, an identity field exactly; every field of an empty classic comes back as zeros; a field the
    model was loaded without must raise when set and come back ``None``, and so must a joint the model was loaded
    without in a field it keeps (drawn zero, read zero); another family's classic must raise.
    """
    spec = human.spec
    classic_type = spec.classic.classic_type
    shapes = spec.classic.field_shapes(spec)
    names = [field.name for field in dataclasses.fields(classic_type)]
    taken = [name for name in names if name in shapes]
    values = {
        name: 0.3 * torch.randn(*BATCHES[index % len(BATCHES)], *shapes[name], generator=generator, dtype=spec.dtype)
        for index, name in enumerate(taken)
    }
    dropped_entries = {}  # field -> its axis-angle entries of joints the model was loaded without
    pose_fields = spec.classic.pose_fields if isinstance(spec.classic, SMPLFamilyFormat) else {}
    for name, joints in pose_fields.items():
        dropped = torch.tensor([joint not in spec.joint_names for joint in joints]).repeat_interleave(3)
        if name in values and bool(dropped.any()):
            dropped_entries[name] = dropped
            values[name] = values[name].masked_fill(dropped, 0.0)
    for name, dropped in dropped_entries.items():
        with pytest.raises(ArgumentError, match="loaded without"):
            bh.from_classic(human, classic_type(**{name: values[name].masked_fill(dropped, 0.3)}))
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
            values[name].expand(*BROADCAST, *shapes[name]),
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
        torch.testing.assert_close(
            identity_back.values[name], value.expand_as(identity_back.values[name]), rtol=0, atol=atol
        )

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


def test_misused_arguments_raise_library_errors(tmp_path: Path) -> None:
    """Principle 2.7: batches that do not broadcast raise ``ShapeError`` naming them, and the other function's first
    argument raises ``ArgumentError``, instead of torch's ``RuntimeError`` or an ``AttributeError``."""
    human = bh.HumanModel.from_spec(flame_spec(tmp_path), density=None)
    betas, global_orient = torch.zeros(2, 100, dtype=F64), torch.zeros(3, 3, dtype=F64)
    with pytest.raises(ShapeError, match=r"betas \(2,\), global_orient \(3,\) do not broadcast"):
        bh.from_classic(human, FLAMEClassic(betas=betas, global_orient=global_orient))
    identity = human.bake(betas=betas)
    with pytest.raises(ShapeError, match=r"q \(3,\), identity \(2,\) do not broadcast"):
        bh.to_classic(identity, human.model.q_neutral.expand(3, -1))
    with pytest.raises(ArgumentError, match="to_classic takes an Identity first"):
        bh.to_classic(human, human.model.q_neutral)
    with pytest.raises(ArgumentError, match="from_classic takes a HumanModel first"):
        bh.from_classic(identity, FLAMEClassic())


def test_a_hand_pca_field_cannot_lose_some_of_its_joints(tmp_path: Path) -> None:
    """A PCA coefficient moves every joint of the hand, so dropping one finger joint raises with the fix."""
    for flat_hand_mean in (False, True):
        spec = bh.families.mano(_mano_file(tmp_path), num_pca_comps=6, flat_hand_mean=flat_hand_mean, dtype=F64)
        with pytest.raises(ArgumentError, match=r"\['index3'\] cannot be left out; load the model with num_pca_comps"):
            bh.families.drop_joints(spec, ["index3"])


@pytest.mark.parametrize("num_pca_comps", (0, 46, 6.0, True))
def test_the_hand_pca_width_is_checked_at_load(tmp_path: Path, num_pca_comps: object) -> None:
    for loader in (bh.families.smplx, bh.families.mano):
        with pytest.raises(ArgumentError, match=r"num_pca_comps must be an int in \[1, 45\]"):
            loader(tmp_path / "unread.npz", num_pca_comps=num_pca_comps)
