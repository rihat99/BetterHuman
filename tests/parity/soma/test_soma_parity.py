"""Golden parity against the pinned float32 SOMA-X v0.3.4 oracle, mid LOD (``tools/parity/soma/dump_soma.py``)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest
import torch

from better_human import HumanModel, HumanState, families
from better_human.families.soma_family import SOMAClassic, soma_from_classic
from better_human.io import weight_path
from tests.support.goldens import golden_dir

pytestmark = pytest.mark.licensed
EXPECTED_CASES = {
    "big_shape",
    "correctives_off",
    *(f"identity_{index:02d}" for index in range(3)),
    *(f"pose_{index:02d}" for index in range(5)),
    "reflect_00",
    "twist_00",
    "twist_01",
    "zero",
}
KEYS = ("identity_coeffs", "poses", "transl", "vertices", "skin_transforms")
LABELS = ("vertex", "joint position", "rotation entry")

# Vertices [m], joint positions [m], rotation entries, float32 and float64: 2x the worst error over the 14 goldens and
# their identity batch, rounded up. A float32 joint regression alone rounds by up to 5.2e-6 m on these identities, so
# a correct float64 or batched float32 evaluation sits that far from the float32 oracle; no gate goes below that.
# Single reflect_00, a 5-sigma identity, sets its own vertex gate and not the general one: its float32 joint regression
# rounds further (5.4e-6 m; 3.0e-6 m with that regression in float64), as upstream's does (7.5e-6 m from its float64).
PARITY_ATOLS = (6.5e-6, 9e-6, 3.5e-6)
PARITY_ATOLS_BY_CASE = {"reflect_00": (1.1e-5, 9e-6, 3.5e-6)}
# Rest mesh [m], bind positions [m], bind rotation entries, float32 and float64: 2x the worst error over the 14
# goldens and their identity batch, rounded up; never below the float32 oracle's 5.2e-6 m joint regression rounding.
# Single reflect_00, a 5-sigma identity, sets its own rest mesh gate and not the general one: its float32 joint
# regression rounds further (5.9e-6 m; 3.2e-6 m with that regression in float64), as upstream's does (7.3e-6 m from
# its float64).
BAKE_ATOLS = (6e-6, 1e-5, 1.1e-6)
BAKE_ATOLS_BY_CASE = {"reflect_00": (1.2e-5, 1e-5, 1.1e-6)}
PARITY_CASES = tuple((dtype, case) for dtype in (torch.float32, torch.float64) for case in sorted(EXPECTED_CASES))


@pytest.fixture(scope="module")
def soma_goldens() -> Path:
    """The SOMA golden directory in the models directory, holding exactly the expected cases."""
    directory = golden_dir("soma_mid")
    found = {path.stem for path in directory.glob("*.npz")}
    assert found == EXPECTED_CASES, f"The SOMA parity grid in {directory} is missing or unexpected"
    return directory


@lru_cache(maxsize=2)
def _body(dtype: torch.dtype, correctives: bool) -> HumanModel:
    return HumanModel.from_spec(
        families.soma(weight_path("SOMA", "soma_mid.npz"), correctives=correctives, dtype=dtype), density=None
    )


def _golden(path: Path) -> tuple[bool, dict[str, torch.Tensor]]:
    """``(apply_correctives, float32 tensors of KEYS)`` of one golden case."""
    with np.load(path, allow_pickle=False) as golden:
        assert int(golden["format_version"]) == 2
        assert str(golden["case_name"]) == path.stem
        return bool(golden["apply_correctives"]), {name: torch.from_numpy(golden[name]) for name in KEYS}


def _errors(state: HumanState, expected: dict[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
    """Max vertex, joint position and rotation entry errors ``(...)`` per case, over all 110 skin joints.

    ``skin_transforms`` holds upstream's public ``joints`` and ``transforms`` bit for bit, plus the twist helpers.
    """
    skin = expected["skin_transforms"][..., :3, :]
    return (
        (state.vertices - expected["vertices"]).abs().amax((-2, -1)),
        (state.joint_translations - skin[..., 3]).abs().amax((-2, -1)),
        (state.joint_transforms[..., :3] - skin[..., :3]).abs().amax((-3, -2, -1)),
    )


@pytest.mark.parametrize(
    ("dtype", "case"),
    PARITY_CASES,
    ids=[f"{str(dtype).removeprefix('torch.')}-{case}" for dtype, case in PARITY_CASES],
)
def test_soma_parity(soma_goldens: Path, dtype: torch.dtype, case: str) -> None:
    """Vertices and the world transforms of every skin joint against one upstream case."""
    golden_path = soma_goldens / f"{case}.npz"
    correctives, golden = _golden(golden_path)
    expected = {name: value.to(dtype) for name, value in golden.items()}
    with torch.inference_mode():
        classic = SOMAClassic(expected["identity_coeffs"], expected["poses"], expected["transl"])
        identity, q, stream = soma_from_classic(_body(dtype, correctives), classic)
        errors = _errors(identity.forward(q, stream), expected)
    atols = PARITY_ATOLS_BY_CASE.get(golden_path.stem, PARITY_ATOLS)
    for label, error, atol in zip(LABELS, errors, atols, strict=True):
        assert error <= atol, f"{dtype} {golden_path.stem}: {label} max error {error.item():.3e}"


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64), ids=("float32", "float64"))
def test_soma_parity_identity_batch(soma_goldens: Path, dtype: torch.dtype) -> None:
    """Every correctives-on golden in one identity batch, one pose each: batched sums round in another order."""
    goldens = {case: _golden(soma_goldens / f"{case}.npz") for case in sorted(EXPECTED_CASES)}
    names = [name for name, (correctives, _) in goldens.items() if correctives]
    expected = {key: torch.stack([goldens[name][1][key] for name in names]).to(dtype) for key in KEYS}
    with torch.inference_mode():
        classic = SOMAClassic(expected["identity_coeffs"], expected["poses"], expected["transl"])
        identity, q, stream = soma_from_classic(_body(dtype, True), classic)
        errors = _errors(identity.forward(q, stream), expected)
    assert q.shape == (len(names), 311)
    for label, error, atol in zip(LABELS, errors, PARITY_ATOLS, strict=True):
        worst = int(error.argmax())
        assert error[worst] <= atol, f"{dtype} {names[worst]} in a batch: {label} max error {error[worst].item():.3e}"


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64), ids=("float32", "float64"))
def test_bake_matches_upstream_rest_shape_and_bind(soma_goldens: Path, dtype: torch.dtype) -> None:
    """Skeleton fit and repose for every golden identity: the skinning rest mesh and the bind poses."""
    body = HumanModel.from_spec(
        families.soma(weight_path("SOMA", "soma_mid.npz"), correctives=False, dtype=dtype), density=None
    )
    for case in sorted(EXPECTED_CASES):
        path = soma_goldens / f"{case}.npz"
        rest_atol, position_atol, rotation_atol = BAKE_ATOLS_BY_CASE.get(case, BAKE_ATOLS)
        with np.load(path, allow_pickle=False) as golden:
            names = ("identity_coeffs", "rest_shape", "bind_transforms")
            coefficients, rest_shape, expected = (torch.from_numpy(golden[name]).to(dtype) for name in names)
        identity = body.bake(shape=coefficients)
        inverse_bind = identity.inverse_bind_affines  # [R | t] of the inverse bind; the bind is [R^T | -R^T t]
        bind = torch.cat((inverse_bind[..., :3].mT, -(inverse_bind[..., :3].mT @ inverse_bind[..., 3:])), dim=-1)
        rest_error = (identity.rest_vertices - rest_shape).abs().amax().item()
        positions = torch.stack((identity.rest_joint_translations, bind[:, :3, 3]))
        position_error = (positions - expected[:, :3, 3]).abs().amax().item()
        rotation_error = (bind[:, :3, :3] - expected[:, :3, :3]).abs().amax().item()
        assert rest_error <= rest_atol, f"{path.stem}: rest mesh max error {rest_error:.3e}"
        assert position_error <= position_atol, f"{path.stem}: bind position max error {position_error:.3e}"
        assert rotation_error <= rotation_atol, f"{path.stem}: bind rotation max error {rotation_error:.3e}"
