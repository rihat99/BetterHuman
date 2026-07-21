"""Secondary MHR oracle cross-check against the released TorchScript model."""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).parents[2]
MODEL_PATH = ROOT / "models" / "MHR" / "assets" / "mhr_model.pt"
GOLDEN_DIR = Path(__file__).parent / "goldens" / "mhr_lod1"
CASES = ("identity_00", "combined_00", "zero_correctives_off", "expression_none")
VERTEX_ATOL_METERS = 2e-6
POSITION_ATOL_METERS = 1e-6
QUATERNION_ATOL = 1e-6
SCALE_ATOL = 1e-7


def _sign_invariant_quaternion_error(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> float:
    direct = (actual - expected).abs().amax(dim=-1)
    sign_flipped = (actual + expected).abs().amax(dim=-1)
    return torch.minimum(direct, sign_flipped).amax().item()


def test_released_torchscript_agrees_with_pymomentum_goldens() -> None:
    if not MODEL_PATH.is_file():
        pytest.skip(f"released MHR TorchScript oracle is absent: {MODEL_PATH}")

    model = torch.jit.load(MODEL_PATH, map_location="cpu")
    model.eval()
    try:
        with torch.inference_mode():
            for case in CASES:
                golden_path = GOLDEN_DIR / f"{case}.npz"
                assert golden_path.is_file(), f"committed MHR golden is absent: {golden_path}"
                with np.load(golden_path, allow_pickle=False) as golden:
                    length_scale = float(golden["length_scale_to_meters"])
                    identity = torch.from_numpy(golden["identity_coeffs"])[None]
                    model_parameters = torch.from_numpy(golden["model_parameters"])[None]
                    expression = (
                        torch.from_numpy(golden["face_expression"])[None]
                        if bool(golden["face_expression_present"])
                        else torch.zeros(1, 72, dtype=torch.float32)
                    )
                    apply_correctives = bool(golden["apply_correctives"])
                    expected_vertices = torch.from_numpy(golden["vertices"]) * length_scale
                    expected_state = torch.from_numpy(golden["skel_state"])

                vertices_cm, state = model(
                    identity,
                    model_parameters,
                    expression,
                    apply_correctives,
                )
                assert vertices_cm.shape == (1, 18439, 3)
                assert state.shape == (1, 127, 8)
                assert vertices_cm.dtype == torch.float32
                assert state.dtype == torch.float32
                vertices = vertices_cm[0] * length_scale
                state = state[0]
                vertex_error = (vertices - expected_vertices).abs().amax().item()
                position_error = (state[..., :3] - expected_state[..., :3]).abs().amax().item() * length_scale
                quaternion_error = _sign_invariant_quaternion_error(
                    state[..., 3:7],
                    expected_state[..., 3:7],
                )
                scale_error = (state[..., 7] - expected_state[..., 7]).abs().amax().item()

                assert vertex_error <= VERTEX_ATOL_METERS, f"{case}: TorchScript vertex max error {vertex_error:.9e} m"
                assert position_error <= POSITION_ATOL_METERS, (
                    f"{case}: TorchScript position max error {position_error:.9e} m"
                )
                assert quaternion_error <= QUATERNION_ATOL, (
                    f"{case}: TorchScript quaternion max error {quaternion_error:.9e}"
                )
                assert scale_error <= SCALE_ATOL, f"{case}: TorchScript scale max error {scale_error:.9e}"
    finally:
        del model
        gc.collect()
