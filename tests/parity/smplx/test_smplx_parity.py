"""Golden parity against the pinned upstream smplx SMPL-X (``tools/parity/smplx/dump_smplx.py``)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

import better_human as bh
from better_human.families import SMPLXClassic
from better_human.io import weight_path
from tests.support.goldens import golden_dir

pytestmark = pytest.mark.licensed
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
FLOAT64_ATOL = 1e-6
FLOAT32_ATOL = 1e-6
_BODY_CACHE: dict[tuple[torch.dtype, int, int, bool, int | None, bool], bh.HumanModel] = {}


@pytest.fixture(scope="module")
def smplx_goldens() -> Path:
    """The SMPL-X golden directory in the models directory, holding exactly the expected grid."""
    directory = golden_dir("smplx_neutral")
    found = {path.stem for path in directory.glob("*.npz")}
    assert found == EXPECTED_GOLDENS, f"SMPL-X parity grid in {directory} is incomplete"
    return directory


def _body(
    dtype: torch.dtype,
    num_betas: int,
    num_expression: int,
    flat_hand_mean: bool,
    num_pca_comps: int | None,
    stream_joints: bool = True,
) -> bh.HumanModel:
    key = (dtype, num_betas, num_expression, flat_hand_mean, num_pca_comps, stream_joints)
    if key not in _BODY_CACHE:
        spec = bh.families.smplx(
            weight_path("smplx", "SMPLX_NEUTRAL.npz"),
            num_betas=num_betas,
            num_expression=num_expression,
            flat_hand_mean=flat_hand_mean,
            num_pca_comps=num_pca_comps,
            dtype=dtype,
        )
        _BODY_CACHE[key] = bh.HumanModel.from_spec(spec, density=None, stream_joints=stream_joints)
    return _BODY_CACHE[key]


def _evaluate(
    dtype: torch.dtype,
    golden_path: Path,
    stream_joints: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, np.lib.npyio.NpzFile]:
    golden = np.load(golden_path, allow_pickle=False)
    num_betas = int(golden["num_betas"].item())
    num_expression = int(golden["num_expression"].item())
    flat_hand_mean = bool(golden["flat_hand_mean"].item())
    stored_pca = int(golden["num_pca_comps"].item())
    num_pca_comps = None if stored_pca < 0 else stored_pca
    body = _body(dtype, num_betas, num_expression, flat_hand_mean, num_pca_comps, stream_joints)
    expression = torch.from_numpy(golden["expression"]).to(dtype=dtype)
    identity, q, stream = bh.from_classic(
        body,
        SMPLXClassic(
            betas=torch.from_numpy(golden["betas"]).to(dtype=dtype),
            global_orient=torch.from_numpy(golden["global_orient"]).to(dtype=dtype),
            body_pose=torch.from_numpy(golden["body_pose"]).to(dtype=dtype),
            jaw_pose=torch.from_numpy(golden["jaw_pose"]).to(dtype=dtype),
            leye_pose=torch.from_numpy(golden["leye_pose"]).to(dtype=dtype),
            reye_pose=torch.from_numpy(golden["reye_pose"]).to(dtype=dtype),
            left_hand_pose=torch.from_numpy(golden["left_hand_pose"]).to(dtype=dtype),
            right_hand_pose=torch.from_numpy(golden["right_hand_pose"]).to(dtype=dtype),
            expression=expression,
            transl=torch.from_numpy(golden["transl"]).to(dtype=dtype),
        ),
    )
    state = identity.forward(q, stream)
    return state.vertices, state.joint_translations, golden


@pytest.mark.parametrize("case", sorted(EXPECTED_GOLDENS))
def test_smplx_float64_parity(smplx_goldens: Path, case: str) -> None:
    golden_path = smplx_goldens / f"{case}.npz"
    vertices, joints, golden = _evaluate(torch.float64, golden_path)
    vertex_error = (vertices - torch.from_numpy(golden["vertices"])).abs().max().item()
    joint_error = (joints - torch.from_numpy(golden["joints"])).abs().max().item()
    golden.close()
    assert vertex_error <= FLOAT64_ATOL, f"{golden_path.stem}: vertex max error {vertex_error:.3e}"
    assert joint_error <= FLOAT64_ATOL, f"{golden_path.stem}: joint max error {joint_error:.3e}"


@pytest.mark.parametrize("case", sorted(EXPECTED_GOLDENS))
def test_smplx_float32_parity(smplx_goldens: Path, case: str) -> None:
    golden_path = smplx_goldens / f"{case}.npz"
    vertices, joints, golden = _evaluate(torch.float32, golden_path)
    expected_vertices = torch.from_numpy(golden["vertices"]).float()
    expected_joints = torch.from_numpy(golden["joints"]).float()
    vertex_error = (vertices - expected_vertices).abs().max().item()
    joint_error = (joints - expected_joints).abs().max().item()
    golden.close()
    assert vertex_error <= FLOAT32_ATOL, f"{golden_path.stem}: vertex max error {vertex_error:.3e}"
    assert joint_error <= FLOAT32_ATOL, f"{golden_path.stem}: joint max error {joint_error:.3e}"


def test_smplx_static_joints_are_approximate(smplx_goldens: Path) -> None:
    """``stream_joints=False`` keeps the expression-free joints: measured on ``expression_plus_two`` (float64) the
    joint error is 1.17e-3 m and the vertex error 5.8e-10 m."""
    vertices, joints, golden = _evaluate(torch.float64, smplx_goldens / "expression_plus_two.npz", stream_joints=False)
    vertex_error = (vertices - torch.from_numpy(golden["vertices"])).abs().max().item()
    joint_error = (joints - torch.from_numpy(golden["joints"])).abs().max().item()
    golden.close()
    error = max(vertex_error, joint_error)
    assert 1e-5 < error < 2e-3, f"static-joint error {error:.3e}"
