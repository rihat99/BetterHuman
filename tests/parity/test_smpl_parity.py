"""Golden parity against the pinned upstream smplx SMPL implementation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

import better_human as bh
from better_human.families.classic import SMPLClassic, smpl_from_classic

ROOT = Path(__file__).parents[2]
MODEL_PATH = ROOT / "models" / "smpl" / "SMPL_NEUTRAL.npz"
GOLDEN_DIR = Path(__file__).parent / "goldens" / "smpl_neutral"
GOLDEN_FILES = tuple(sorted(GOLDEN_DIR.glob("*.npz")))
EXPECTED_GOLDENS = {
    "beta_minus_two",
    "beta_plus_two",
    "beta_zero",
    *(f"random_{index:02d}" for index in range(8)),
    "root_wart",
    "zero",
}
assert {path.stem for path in GOLDEN_FILES} == EXPECTED_GOLDENS, "SMPL parity goldens are missing or unexpected"
FLOAT64_ATOL = 1e-6
FLOAT32_ATOL = 1e-6


@pytest.fixture(scope="module")
def body_float64() -> bh.Body:
    """Load the licensed local neutral weights in oracle precision."""
    if not MODEL_PATH.is_file():
        pytest.skip(f"licensed SMPL fixture is absent: {MODEL_PATH}")
    return bh.Body(bh.families.smpl(MODEL_PATH, dtype=torch.float64), mass=False)


@pytest.fixture(scope="module")
def body_float32() -> bh.Body:
    """Load the licensed local neutral weights in runtime precision."""
    if not MODEL_PATH.is_file():
        pytest.skip(f"licensed SMPL fixture is absent: {MODEL_PATH}")
    return bh.Body(bh.families.smpl(MODEL_PATH, dtype=torch.float32), mass=False)


def _evaluate(body: bh.Body, golden_path: Path) -> tuple[torch.Tensor, torch.Tensor, np.lib.npyio.NpzFile]:
    golden = np.load(golden_path, allow_pickle=False)
    dtype = body.dtype
    identity = body.bake(betas=torch.from_numpy(golden["betas"]).to(dtype=dtype))
    q, stream = smpl_from_classic(
        identity,
        SMPLClassic(
            global_orient=torch.from_numpy(golden["global_orient"]).to(dtype=dtype),
            body_pose=torch.from_numpy(golden["body_pose"]).to(dtype=dtype),
            transl=torch.from_numpy(golden["transl"]).to(dtype=dtype),
        ),
    )
    state = identity.forward(q, stream)
    return state.vertices, state.joints, golden


@pytest.mark.parametrize("golden_path", GOLDEN_FILES, ids=lambda path: path.stem)
def test_smpl_float64_parity(body_float64: bh.Body, golden_path: Path) -> None:
    vertices, joints, golden = _evaluate(body_float64, golden_path)
    vertex_error = (vertices - torch.from_numpy(golden["vertices"])).abs().max().item()
    joint_error = (joints - torch.from_numpy(golden["joints"])).abs().max().item()
    golden.close()
    assert vertex_error <= FLOAT64_ATOL, f"{golden_path.stem}: vertex max error {vertex_error:.3e}"
    assert joint_error <= FLOAT64_ATOL, f"{golden_path.stem}: joint max error {joint_error:.3e}"


@pytest.mark.parametrize("golden_path", GOLDEN_FILES, ids=lambda path: path.stem)
def test_smpl_float32_parity(body_float32: bh.Body, golden_path: Path) -> None:
    vertices, joints, golden = _evaluate(body_float32, golden_path)
    expected_vertices = torch.from_numpy(golden["vertices"]).float()
    expected_joints = torch.from_numpy(golden["joints"]).float()
    vertex_error = (vertices - expected_vertices).abs().max().item()
    joint_error = (joints - expected_joints).abs().max().item()
    golden.close()
    assert vertex_error <= FLOAT32_ATOL, f"{golden_path.stem}: vertex max error {vertex_error:.3e}"
    assert joint_error <= FLOAT32_ATOL, f"{golden_path.stem}: joint max error {joint_error:.3e}"
