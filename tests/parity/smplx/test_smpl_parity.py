"""Golden parity against the pinned upstream smplx SMPL (``tools/parity/smplx/dump_smpl.py``)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

import better_human as bh
from better_human.families import SMPLClassic
from better_human.io import weight_path
from tests.support.goldens import golden_dir

pytestmark = pytest.mark.licensed
EXPECTED_GOLDENS = {
    "beta_minus_two",
    "beta_plus_two",
    "beta_zero",
    *(f"random_{index:02d}" for index in range(8)),
    "root_wart",
    "zero",
}
FLOAT64_ATOL = 1e-6
FLOAT32_ATOL = 1e-6


@pytest.fixture(scope="module")
def smpl_goldens() -> Path:
    """The SMPL golden directory in the models directory, holding exactly the expected cases."""
    directory = golden_dir("smpl_neutral")
    found = {path.stem for path in directory.glob("*.npz")}
    assert found == EXPECTED_GOLDENS, f"SMPL parity goldens in {directory} are missing or unexpected"
    return directory


@pytest.fixture(scope="module")
def body_float64() -> bh.HumanModel:
    """Load the licensed local neutral weights in oracle precision."""
    return bh.HumanModel.from_spec(
        bh.families.smpl(weight_path("smpl", "SMPL_NEUTRAL.npz"), dtype=torch.float64), density=None
    )


@pytest.fixture(scope="module")
def body_float32() -> bh.HumanModel:
    """Load the licensed local neutral weights in runtime precision."""
    return bh.HumanModel.from_spec(
        bh.families.smpl(weight_path("smpl", "SMPL_NEUTRAL.npz"), dtype=torch.float32), density=None
    )


def _evaluate(body: bh.HumanModel, golden_path: Path) -> tuple[torch.Tensor, torch.Tensor, np.lib.npyio.NpzFile]:
    golden = np.load(golden_path, allow_pickle=False)
    dtype = body.spec.dtype
    identity, q, stream = bh.from_classic(
        body,
        SMPLClassic(
            betas=torch.from_numpy(golden["betas"]).to(dtype=dtype),
            global_orient=torch.from_numpy(golden["global_orient"]).to(dtype=dtype),
            body_pose=torch.from_numpy(golden["body_pose"]).to(dtype=dtype),
            transl=torch.from_numpy(golden["transl"]).to(dtype=dtype),
        ),
    )
    state = identity.forward(q, stream)
    return state.vertices, state.joint_translations, golden


@pytest.mark.parametrize("case", sorted(EXPECTED_GOLDENS))
def test_smpl_float64_parity(body_float64: bh.HumanModel, smpl_goldens: Path, case: str) -> None:
    golden_path = smpl_goldens / f"{case}.npz"
    vertices, joints, golden = _evaluate(body_float64, golden_path)
    vertex_error = (vertices - torch.from_numpy(golden["vertices"])).abs().max().item()
    joint_error = (joints - torch.from_numpy(golden["joints"])).abs().max().item()
    golden.close()
    assert vertex_error <= FLOAT64_ATOL, f"{golden_path.stem}: vertex max error {vertex_error:.3e}"
    assert joint_error <= FLOAT64_ATOL, f"{golden_path.stem}: joint max error {joint_error:.3e}"


@pytest.mark.parametrize("case", sorted(EXPECTED_GOLDENS))
def test_smpl_float32_parity(body_float32: bh.HumanModel, smpl_goldens: Path, case: str) -> None:
    golden_path = smpl_goldens / f"{case}.npz"
    vertices, joints, golden = _evaluate(body_float32, golden_path)
    expected_vertices = torch.from_numpy(golden["vertices"]).float()
    expected_joints = torch.from_numpy(golden["joints"]).float()
    vertex_error = (vertices - expected_vertices).abs().max().item()
    joint_error = (joints - expected_joints).abs().max().item()
    golden.close()
    assert vertex_error <= FLOAT32_ATOL, f"{golden_path.stem}: vertex max error {vertex_error:.3e}"
    assert joint_error <= FLOAT32_ATOL, f"{golden_path.stem}: joint max error {joint_error:.3e}"
