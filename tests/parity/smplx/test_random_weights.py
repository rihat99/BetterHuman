"""Golden parity of SMPL, SMPL+H and SMPL-X on tracked random weights against the pinned upstream smplx.

Decision record 0016: ``tests/support/random_weights/<family>_random.npz`` (written by
``tools/parity/smplx/make_random_archive.py``) holds random weights in each family's upstream layout, and
``tools/parity/smplx/dump_{smpl,smplh,smplx}.py`` wrote the oracle's outputs on them to
``tests/parity/smplx/random/<family>/``. The gate is the licensed one, 1e-6 m in float64 and float32. The models load
with ``density=None``, as in the licensed parity tests: mass does not enter vertices or joints.
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

ARCHIVES = Path(__file__).parents[2] / "support" / "random_weights"
GOLDENS = Path(__file__).parent / "random"
ATOL = 1e-6
_SMPL_GRID = (
    "beta_minus_two",
    "beta_plus_two",
    "beta_zero",
    *(f"random_{index:02d}" for index in range(8)),
    "root_wart",
)
#: family -> (loader, classic type, the dump script's case names).
FAMILIES: dict[str, tuple[Any, type, tuple[str, ...]]] = {
    "smpl": (bh.families.smpl, SMPLClassic, (*_SMPL_GRID, "zero")),
    "smplh": (bh.families.smplh, SMPLHClassic, (*_SMPL_GRID, "left_hand", "right_hand", "zero")),
    "smplx": (
        bh.families.smplx,
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
_MODELS: dict[tuple[Any, ...], bh.HumanModel] = {}


def _model(family: str, dtype: torch.dtype, options: dict[str, Any]) -> bh.HumanModel:
    key = (family, dtype, *sorted(options.items()))
    if key not in _MODELS:
        spec = FAMILIES[family][0](ARCHIVES / f"{family}_random.npz", dtype=dtype, **options)
        _MODELS[key] = bh.HumanModel.from_spec(spec, density=None)
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


@pytest.mark.parametrize("family", FAMILIES)
def test_random_weight_goldens_are_the_dump_grid(family: str) -> None:
    found = {path.stem for path in (GOLDENS / family).glob("*.npz")}
    assert found == set(FAMILIES[family][2]), f"random-weight goldens in {GOLDENS / family} are missing or unexpected"


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32], ids=["float64", "float32"])
@pytest.mark.parametrize(
    ("family", "case"), [(family, case) for family, (*_, cases) in FAMILIES.items() for case in cases]
)
def test_random_weight_parity(family: str, case: str, dtype: torch.dtype) -> None:
    golden_path = GOLDENS / family / f"{case}.npz"
    classic_type = FAMILIES[family][1]
    with np.load(golden_path, allow_pickle=False) as golden:
        human = _model(family, dtype, _options(golden))
        fields = {field.name for field in dataclasses.fields(classic_type)} & set(golden.files)
        classic = classic_type(**{name: torch.from_numpy(golden[name]).to(dtype) for name in fields})
        expected_vertices = torch.from_numpy(golden["vertices"]).to(dtype)
        expected_joints = torch.from_numpy(golden["joints"]).to(dtype)
    identity, q, streams = bh.from_classic(human, classic)
    state = identity.forward(q, streams)
    vertex_error = (state.vertices - expected_vertices).abs().max().item()
    joint_error = (state.joint_translations - expected_joints).abs().max().item()
    assert vertex_error <= ATOL, f"{family}/{case}: vertex max error {vertex_error:.3e}"
    assert joint_error <= ATOL, f"{family}/{case}: joint max error {joint_error:.3e}"
