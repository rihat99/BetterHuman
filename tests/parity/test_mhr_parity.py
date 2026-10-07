"""Golden parity against the pinned float32 pymomentum MHR oracle."""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import pytest
import torch

from better_human import families
from better_human.core.body import Body
from better_human.families.mhr import MHRClassic, mhr_from_classic

ROOT = Path(__file__).parents[2]
MODEL_DIR = ROOT / "models" / "MHR" / "converted"
GOLDEN_ROOT = Path(__file__).parent / "goldens"
EXPECTED_GOLDEN_COUNTS = (3, 30, 3, 3, 3, 3, 3)
SMOKE_CASES = {"fully_random", "proportions_hand_scale", "zero"}
EXPECTED_CASES = {
    0: SMOKE_CASES,
    1: {
        "combined_00",
        "combined_01",
        "combined_02",
        "combined_03",
        "combined_04",
        "combined_05",
        "combined_06",
        "expression_00",
        "expression_01",
        "expression_02",
        "expression_none",
        "hand_scale_crossed",
        "hand_scale_max",
        "hand_scale_min",
        "identity_00",
        "identity_01",
        "identity_02",
        "proportions_00",
        "proportions_01",
        "proportions_02",
        "random_pose_00",
        "random_pose_01",
        "random_pose_02",
        "random_pose_03",
        "random_pose_04",
        "root_transform_00",
        "root_transform_01",
        "tpose",
        "zero_correctives_off",
        "zero_correctives_on",
    },
    **{lod: SMOKE_CASES for lod in range(2, 7)},
}
LOD_VERTEX_COUNTS = (73639, 18439, 10661, 4899, 2461, 971, 595)

# (vertices [m], joint positions [m], quaternion coefficients, joint scales).
# The float64 implementation is intentionally compared with a float32 oracle.
PARITY_ATOLS = {
    torch.float32: {lod: (5e-6, 1e-6, 1e-6, 1e-7) for lod in range(7)},
    torch.float64: {lod: (2e-6, 5e-7, 5e-7, 1e-7) for lod in range(7)},
}

PARITY_CONFIGS = tuple(
    (lod, dtype, True, True) for dtype in (torch.float32, torch.float64) for lod in range(7)
) + tuple(
    config for dtype in (torch.float32, torch.float64) for config in ((1, dtype, False, True), (1, dtype, True, False))
)
PARITY_IDS = tuple(
    f"lod{lod}-{str(dtype).removeprefix('torch.')}-"
    f"{'correctives' if use_correctives else 'no-correctives'}-"
    f"{'expression' if use_expression else 'no-expression'}"
    for lod, dtype, use_correctives, use_expression in PARITY_CONFIGS
)

GOLDEN_FILES = {lod: tuple(sorted((GOLDEN_ROOT / f"mhr_lod{lod}").glob("*.npz"))) for lod in range(7)}
assert tuple(len(GOLDEN_FILES[lod]) for lod in range(7)) == EXPECTED_GOLDEN_COUNTS, (
    "The committed MHR parity grid is missing or unexpected"
)
assert all({path.stem for path in GOLDEN_FILES[lod]} == EXPECTED_CASES[lod] for lod in range(7)), (
    "The committed MHR parity case names are missing or unexpected"
)


def _model_path(lod: int) -> Path:
    path = MODEL_DIR / f"mhr_lod{lod}.npz"
    if not path.is_file():
        pytest.skip(f"converted MHR LOD{lod} fixture is absent: {path}")
    return path


def _selected_goldens(
    lod: int,
    use_correctives: bool,
    use_expression: bool,
) -> tuple[Path, ...]:
    selected: list[Path] = []
    for path in GOLDEN_FILES[lod]:
        with np.load(path, allow_pickle=False) as golden:
            if bool(golden["apply_correctives"]) != use_correctives:
                continue
            if not use_expression and bool(golden["face_expression_present"]):
                continue
        selected.append(path)
    return tuple(selected)


def _sign_invariant_quaternion_error(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> float:
    direct = (actual - expected).abs().amax(dim=-1)
    sign_flipped = (actual + expected).abs().amax(dim=-1)
    return torch.minimum(direct, sign_flipped).amax().item()


@pytest.mark.parametrize(
    ("lod", "dtype", "use_correctives", "use_expression"),
    PARITY_CONFIGS,
    ids=PARITY_IDS,
)
def test_mhr_parity_grid(
    lod: int,
    dtype: torch.dtype,
    use_correctives: bool,
    use_expression: bool,
) -> None:
    """Stream one oracle group through one body to bound peak memory."""
    paths = _selected_goldens(lod, use_correctives, use_expression)
    assert paths, "parity configuration selected no golden cases"
    body = Body(
        families.mhr(
            _model_path(lod), lod=lod, correctives=use_correctives, expression=use_expression, dtype=dtype
        )
    )
    vertex_atol, position_atol, quaternion_atol, scale_atol = PARITY_ATOLS[dtype][lod]
    baked = q = state = actual_vertices = actual_poses = actual_scales = None

    try:
        with torch.inference_mode():
            for golden_path in paths:
                with np.load(golden_path, allow_pickle=False) as golden:
                    assert int(golden["format_version"]) == 1
                    assert int(golden["lod"]) == lod
                    assert str(golden["case_name"]) == golden_path.stem
                    assert str(golden["oracle"]) == "MHR reference package via pymomentum"
                    assert str(golden["oracle_dtype"]) == "float32"
                    assert str(golden["native_unit"]) == "centimeter"
                    assert str(golden["native_pose_format"]) == "euler_xyz"
                    assert str(golden["skeleton_state_format"]) == ("translation_xyz_quaternion_xyzw_scale")
                    assert golden["length_scale_to_meters"].dtype == np.float32
                    assert float(golden["length_scale_to_meters"]) == pytest.approx(0.01)
                    expected_fields = {
                        "identity_coeffs": (45,),
                        "model_parameters": (204,),
                        "face_expression": (72,),
                        "vertices": (LOD_VERTEX_COUNTS[lod], 3),
                        "skel_state": (127, 8),
                    }
                    for field, shape in expected_fields.items():
                        assert golden[field].dtype == np.float32
                        assert golden[field].shape == shape
                    length_scale = float(golden["length_scale_to_meters"])
                    identity = torch.from_numpy(golden["identity_coeffs"]).to(dtype=dtype)
                    model_parameters = torch.from_numpy(golden["model_parameters"]).to(dtype=dtype)
                    expression = (
                        torch.from_numpy(golden["face_expression"]).to(dtype=dtype)
                        if bool(golden["face_expression_present"])
                        else None
                    )
                    expected_vertices = torch.from_numpy(golden["vertices"]).to(dtype=dtype) * length_scale
                    expected_state = torch.from_numpy(golden["skel_state"]).to(dtype=dtype)

                baked, q, stream = mhr_from_classic(body, MHRClassic(identity, model_parameters, expression))
                state = baked.forward(q, stream)
                actual_vertices = state.vertices()
                actual_poses = state.joint_poses
                actual_scales = baked.joint_scales

                assert actual_vertices.dtype == dtype
                assert actual_poses.dtype == dtype
                assert actual_scales.dtype == dtype
                vertex_error = (actual_vertices - expected_vertices).abs().amax().item()
                position_error = (actual_poses[..., :3] - expected_state[..., :3] * length_scale).abs().amax().item()
                quaternion_error = _sign_invariant_quaternion_error(
                    actual_poses[..., 3:7],
                    expected_state[..., 3:7],
                )
                scale_error = (actual_scales - expected_state[..., 7]).abs().amax().item()
                context = f"LOD{lod} {dtype} {golden_path.stem}"
                assert vertex_error <= vertex_atol, f"{context}: vertex max error {vertex_error:.9e}"
                assert position_error <= position_atol, f"{context}: position max error {position_error:.9e}"
                assert quaternion_error <= quaternion_atol, f"{context}: quaternion max error {quaternion_error:.9e}"
                assert scale_error <= scale_atol, f"{context}: scale max error {scale_error:.9e}"
    finally:
        del actual_scales, actual_poses, actual_vertices, state, q, baked, body
        gc.collect()
