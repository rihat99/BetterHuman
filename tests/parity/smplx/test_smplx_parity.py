"""Golden parity of SMPL, SMPL+H and SMPL-X against the pinned upstream smplx, on random and on licensed weights.

``tools/parity/smplx/dump_{smpl,smplh,smplx}.py`` wrote the oracle's outputs on one grid of cases per family, from two
weight sources. ``random``: the tracked archives ``tests/support/random_weights/<family>_random.npz`` (written by
``tools/parity/smplx/make_random_archive.py``, random weights in each family's upstream layout), goldens in
``tests/parity/smplx/random/<family>/``, full tier. ``licensed``: the neutral SMPL and SMPL-X weights of the models
directory, goldens in ``$BETTERHUMAN_MODELS_DIR/goldens/<family>_neutral/``, release tier (marked ``licensed``). One
evaluator runs both, 1e-6 m in float64 and float32. The models load with ``density=None``: mass does not enter
vertices or joints. Each random archive is read once per session (``tests.support.random_archives``); every option
set runs the full loader.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

import better_human as bh
from better_human.families import SMPLClassic, SMPLHClassic, SMPLXClassic
from better_human.io import weight_path
from tests.support.goldens import golden_dir
from tests.support.random_archives import random_weight_arrays, random_weight_spec

GOLDENS = Path(__file__).parent / "random"
ATOL = 1e-6
#: Least motion in metres per unit coefficient of every joint, the root included, with the first ten shape columns
#: and the first ten expression columns (norm of its ``(3, 10)`` block of ``J_regressor @ shapedirs``): a joint that
#: stays put would leave its shape-dependent pivot untested (the classic root pivots at ``transl + root_offset``).
MIN_JOINT_SHAPE_MOTION = 1e-3
#: First column of each blend group in an archive's ``shapedirs``; SMPL-X stores 300 shape then 100 expression columns.
_GROUP_COLUMNS = {"smpl": {"betas": 0}, "smplh": {"betas": 0}, "smplx": {"betas": 0, "expression": 300}}
_SMPL_GRID = (
    "beta_minus_two",
    "beta_plus_two",
    "beta_zero",
    *(f"random_{index:02d}" for index in range(8)),
    "root_wart",
)
#: family -> (classic type, the dump script's case names).
FAMILIES: dict[str, tuple[type, tuple[str, ...]]] = {
    "smpl": (SMPLClassic, (*_SMPL_GRID, "zero")),
    "smplh": (SMPLHClassic, (*_SMPL_GRID, "left_hand", "right_hand", "zero")),
    "smplx": (
        SMPLXClassic,
        (
            "beta_minus_two",
            "beta_plus_two",
            "expression_minus_two",
            "expression_plus_two",
            "face_all",
            "full_both",
            "full_left",
            "full_right",
            "jaw",
            "left_eye",
            "pca_both",
            "pca_left",
            "pca_right",
            "pca_twelve",
            *(f"random_{index:02d}" for index in range(8)),
            "reduced_shape_expression",
            "right_eye",
            "root_wart",
            "zero_full_flat",
            "zero_full_mean",
            "zero_pca_flat",
            "zero_pca_mean",
        ),
    ),
}
#: family -> the licensed neutral weights ``(loader, folder, file)``; SMPL+H has no licensed goldens.
LICENSED = {
    "smpl": (bh.families.smpl, "smpl", "SMPL_NEUTRAL.npz"),
    "smplx": (bh.families.smplx, "smplx", "SMPLX_NEUTRAL.npz"),
}
#: ``(source, family)``: every family on its random archive, and the families with licensed goldens.
SOURCES = [*(("random", family) for family in FAMILIES), *(("licensed", family) for family in LICENSED)]
_MODELS: dict[tuple[Any, ...], bh.HumanModel] = {}


def _case(source: str, *values: str) -> Any:
    """A test case of ``source``, marked ``licensed`` when its goldens are."""
    return pytest.param(
        source, *values, marks=pytest.mark.licensed if source == "licensed" else (), id="-".join((source, *values))
    )


def _goldens(source: str, family: str) -> Path:
    """The golden directory of ``source``; the licensed one is read only when a test runs, never at collection."""
    return GOLDENS / family if source == "random" else golden_dir(f"{family}_neutral")


def _model(
    source: str, family: str, dtype: torch.dtype, options: dict[str, Any], *, stream_joints: bool
) -> bh.HumanModel:
    key = (source, family, dtype, stream_joints, *sorted(options.items()))
    if key not in _MODELS:
        if source == "random":
            spec = random_weight_spec(family, dtype=dtype, **options)
        else:
            loader, folder, file = LICENSED[family]
            spec = loader(weight_path(folder, file), dtype=dtype, **options)
        _MODELS[key] = bh.HumanModel.from_spec(spec, density=None, stream_joints=stream_joints)
    return _MODELS[key]


def _options(golden: np.lib.npyio.NpzFile) -> dict[str, Any]:
    """The loader options a golden was dumped with: its shape width, and for SMPL-X the grid's stored options."""
    if "num_pca_comps" not in golden.files:
        return {"num_betas": golden["betas"].shape[-1]}
    stored_pca = int(golden["num_pca_comps"])
    return {
        "num_betas": int(golden["num_betas"]),
        "num_expression": int(golden["num_expression"]),
        "flat_hand_mean": bool(golden["flat_hand_mean"]),
        "num_pca_comps": None if stored_pca < 0 else stored_pca,
    }


def _errors(
    source: str, family: str, case: str, dtype: torch.dtype, *, stream_joints: bool = True
) -> tuple[float, float]:
    """Largest vertex and joint distances in metres between the model on ``case``'s classic inputs and its golden."""
    classic_type = FAMILIES[family][0]
    with np.load(_goldens(source, family) / f"{case}.npz", allow_pickle=False) as golden:
        human = _model(source, family, dtype, _options(golden), stream_joints=stream_joints)
        fields = {field.name for field in dataclasses.fields(classic_type)} & set(golden.files)
        classic = classic_type(**{name: torch.from_numpy(golden[name]).to(dtype) for name in fields})
        expected_vertices = torch.from_numpy(golden["vertices"]).to(dtype)
        expected_joints = torch.from_numpy(golden["joints"]).to(dtype)
    identity, q, streams = bh.from_classic(human, classic)
    state = identity.forward(q, streams)
    vertex_error = (state.vertices - expected_vertices).abs().max().item()
    return vertex_error, (state.joint_translations - expected_joints).abs().max().item()


@pytest.mark.parametrize(("source", "family"), [_case(*pair) for pair in SOURCES])
def test_goldens_are_the_dump_grid(source: str, family: str) -> None:
    directory = _goldens(source, family)
    found = {path.stem for path in directory.glob("*.npz")}
    assert found == set(FAMILIES[family][1]), f"parity goldens in {directory} are missing or unexpected"


@pytest.mark.parametrize("family", FAMILIES)
def test_every_random_weight_joint_moves_with_shape(family: str) -> None:
    arrays = random_weight_arrays(family)
    regressor, shapedirs = arrays["J_regressor"].double(), arrays["shapedirs"].double()
    for group, start in _GROUP_COLUMNS[family].items():
        joint_bases = torch.einsum("jv,vcs->jcs", regressor, shapedirs[..., start : start + 10])
        motion = torch.linalg.vector_norm(joint_bases, dim=(1, 2))
        still = torch.nonzero(motion < MIN_JOINT_SHAPE_MOTION).flatten().tolist()
        assert not still, (
            f"{family}: joints {still} move less than {MIN_JOINT_SHAPE_MOTION} m per unit {group} coefficient, so the "
            "random-weight gate cannot see their shape-dependent pivots; regenerate the archive"
        )


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32], ids=["float64", "float32"])
@pytest.mark.parametrize(
    ("source", "family", "case"),
    [_case(source, family, case) for source, family in SOURCES for case in FAMILIES[family][1]],
)
def test_parity(source: str, family: str, case: str, dtype: torch.dtype) -> None:
    vertex_error, joint_error = _errors(source, family, case, dtype)
    assert vertex_error <= ATOL, f"{source} {family}/{case}: vertex max error {vertex_error:.3e}"
    assert joint_error <= ATOL, f"{source} {family}/{case}: joint max error {joint_error:.3e}"


@pytest.mark.licensed
def test_smplx_static_joints_are_approximate() -> None:
    """``stream_joints=False`` keeps the expression-free joints: measured on ``expression_plus_two`` (float64) the
    joint error is 1.17e-3 m and the vertex error 5.8e-10 m."""
    error = max(_errors("licensed", "smplx", "expression_plus_two", torch.float64, stream_joints=False))
    assert 1e-5 < error < 2e-3, f"static-joint error {error:.3e}"
