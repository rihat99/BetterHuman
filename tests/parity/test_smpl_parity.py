"""Golden parity against the pinned upstream smplx SMPL implementation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from better_human import SMPL
from better_human.bodies import SMPLClassic

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
def body_float64() -> SMPL:
    """Load the licensed local neutral weights in oracle precision."""
    if not MODEL_PATH.is_file():
        pytest.skip(f"licensed SMPL fixture is absent: {MODEL_PATH}")
    return SMPL(MODEL_PATH, compute_mass=False, dtype=torch.float64)


@pytest.fixture(scope="module")
def body_float32() -> SMPL:
    """Load the licensed local neutral weights in runtime precision."""
    if not MODEL_PATH.is_file():
        pytest.skip(f"licensed SMPL fixture is absent: {MODEL_PATH}")
    return SMPL(MODEL_PATH, compute_mass=False, dtype=torch.float32)


def _evaluate(body: SMPL, golden_path: Path) -> tuple[torch.Tensor, torch.Tensor, np.lib.npyio.NpzFile]:
    golden = np.load(golden_path, allow_pickle=False)
    dtype = body.values.v_template.dtype
    shaped = body.with_shape(betas=torch.from_numpy(golden["betas"]).to(dtype=dtype))
    q = shaped.from_classic(
        SMPLClassic(
            global_orient=torch.from_numpy(golden["global_orient"]).to(dtype=dtype),
            body_pose=torch.from_numpy(golden["body_pose"]).to(dtype=dtype),
            transl=torch.from_numpy(golden["transl"]).to(dtype=dtype),
        )
    )
    data = shaped.fk(q)
    return shaped.vertices_from_data(data), data.joint_pose_world[..., 1:, :3], golden


@pytest.mark.parametrize("golden_path", GOLDEN_FILES, ids=lambda path: path.stem)
def test_smpl_float64_parity(body_float64: SMPL, golden_path: Path) -> None:
    vertices, joints, golden = _evaluate(body_float64, golden_path)
    vertex_error = (vertices - torch.from_numpy(golden["vertices"])).abs().max().item()
    joint_error = (joints - torch.from_numpy(golden["joints"])).abs().max().item()
    golden.close()
    assert vertex_error <= FLOAT64_ATOL, f"{golden_path.stem}: vertex max error {vertex_error:.3e}"
    assert joint_error <= FLOAT64_ATOL, f"{golden_path.stem}: joint max error {joint_error:.3e}"


@pytest.mark.parametrize("golden_path", GOLDEN_FILES, ids=lambda path: path.stem)
def test_smpl_float32_parity(body_float32: SMPL, golden_path: Path) -> None:
    vertices, joints, golden = _evaluate(body_float32, golden_path)
    expected_vertices = torch.from_numpy(golden["vertices"]).float()
    expected_joints = torch.from_numpy(golden["joints"]).float()
    vertex_error = (vertices - expected_vertices).abs().max().item()
    joint_error = (joints - expected_joints).abs().max().item()
    golden.close()
    assert vertex_error <= FLOAT32_ATOL, f"{golden_path.stem}: vertex max error {vertex_error:.3e}"
    assert joint_error <= FLOAT32_ATOL, f"{golden_path.stem}: joint max error {joint_error:.3e}"
