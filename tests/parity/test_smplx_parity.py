"""Regenerate-on-demand parity against upstream smplx SMPL-X."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from better_human import SMPLX
from better_human.bodies import SMPLXClassic

ROOT = Path(__file__).parents[2]
MODEL_PATH = ROOT / "models" / "smplx" / "SMPLX_NEUTRAL.npz"
GOLDEN_DIR = Path(__file__).parent / "goldens" / "smplx_neutral"
GOLDEN_FILES = tuple(sorted(GOLDEN_DIR.glob("*.npz")))
EXPECTED_GOLDENS = {
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
}
if GOLDEN_FILES:
    assert {path.stem for path in GOLDEN_FILES} == EXPECTED_GOLDENS, "SMPL-X parity grid is incomplete"
PARITY_CASES: tuple[Path | None, ...] = GOLDEN_FILES if GOLDEN_FILES else (None,)
FLOAT64_ATOL = 1e-6
FLOAT32_ATOL = 1e-6
_BODY_CACHE: dict[tuple[torch.dtype, int, int, bool], SMPLX] = {}


def _body(dtype: torch.dtype, num_betas: int, num_expression: int, flat_hand_mean: bool) -> SMPLX:
    key = (dtype, num_betas, num_expression, flat_hand_mean)
    if key not in _BODY_CACHE:
        _BODY_CACHE[key] = SMPLX(
            MODEL_PATH,
            num_betas=num_betas,
            num_expression=num_expression,
            flat_hand_mean=flat_hand_mean,
            compute_mass=False,
            dtype=dtype,
        )
    return _BODY_CACHE[key]


def _evaluate(
    dtype: torch.dtype,
    golden_path: Path | None,
) -> tuple[torch.Tensor, torch.Tensor, np.lib.npyio.NpzFile]:
    if golden_path is None:
        pytest.skip(
            "SMPL-X goldens are absent; regenerate with the command in tools/parity/smplx/README.md"
        )
    if not MODEL_PATH.is_file():
        pytest.skip(f"licensed SMPL-X fixture is absent: {MODEL_PATH}")
    golden = np.load(golden_path, allow_pickle=False)
    num_betas = int(golden["num_betas"].item())
    num_expression = int(golden["num_expression"].item())
    flat_hand_mean = bool(golden["flat_hand_mean"].item())
    stored_pca = int(golden["num_pca_comps"].item())
    num_pca_comps = None if stored_pca < 0 else stored_pca
    body = _body(dtype, num_betas, num_expression, flat_hand_mean)
    expression = torch.from_numpy(golden["expression"]).to(dtype=dtype)
    shaped = body.with_shape(
        betas=torch.from_numpy(golden["betas"]).to(dtype=dtype),
        expression=expression,
    )
    q = shaped.from_classic(
        SMPLXClassic(
            global_orient=torch.from_numpy(golden["global_orient"]).to(dtype=dtype),
            body_pose=torch.from_numpy(golden["body_pose"]).to(dtype=dtype),
            jaw_pose=torch.from_numpy(golden["jaw_pose"]).to(dtype=dtype),
            leye_pose=torch.from_numpy(golden["leye_pose"]).to(dtype=dtype),
            reye_pose=torch.from_numpy(golden["reye_pose"]).to(dtype=dtype),
            left_hand_pose=torch.from_numpy(golden["left_hand_pose"]).to(dtype=dtype),
            right_hand_pose=torch.from_numpy(golden["right_hand_pose"]).to(dtype=dtype),
            expression=expression,
            transl=torch.from_numpy(golden["transl"]).to(dtype=dtype),
            num_pca_comps=num_pca_comps,
        )
    )
    data = shaped.fk(q)
    return shaped.vertices_from_data(data), data.joint_pose_world[..., 1:, :3], golden


@pytest.mark.parametrize("golden_path", PARITY_CASES, ids=lambda path: "missing" if path is None else path.stem)
def test_smplx_float64_parity(golden_path: Path | None) -> None:
    vertices, joints, golden = _evaluate(torch.float64, golden_path)
    vertex_error = (vertices - torch.from_numpy(golden["vertices"])).abs().max().item()
    joint_error = (joints - torch.from_numpy(golden["joints"])).abs().max().item()
    golden.close()
    assert vertex_error <= FLOAT64_ATOL, f"{golden_path.stem}: vertex max error {vertex_error:.3e}"
    assert joint_error <= FLOAT64_ATOL, f"{golden_path.stem}: joint max error {joint_error:.3e}"


@pytest.mark.parametrize("golden_path", PARITY_CASES, ids=lambda path: "missing" if path is None else path.stem)
def test_smplx_float32_parity(golden_path: Path | None) -> None:
    vertices, joints, golden = _evaluate(torch.float32, golden_path)
    expected_vertices = torch.from_numpy(golden["vertices"]).float()
    expected_joints = torch.from_numpy(golden["joints"]).float()
    vertex_error = (vertices - expected_vertices).abs().max().item()
    joint_error = (joints - expected_joints).abs().max().item()
    golden.close()
    assert vertex_error <= FLOAT32_ATOL, f"{golden_path.stem}: vertex max error {vertex_error:.3e}"
    assert joint_error <= FLOAT32_ATOL, f"{golden_path.stem}: joint max error {joint_error:.3e}"
