"""Dump deterministic float64 SMPL-X goldens with upstream smplx 0.1.28."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
from smplx import SMPLX

SEED = 20260720
RANDOM_CASES = 8
DEFAULT_BETAS = 16
DEFAULT_EXPRESSION = 10
DEFAULT_PCA = 6


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path", type=Path, help="Neutral SMPL-X .npz")
    parser.add_argument("output_dir", type=Path, help="Directory for per-case .npz files")
    return parser.parse_args()


def _axis_angles(generator: torch.Generator, count: int, maximum: float = 1.5) -> torch.Tensor:
    directions = torch.randn(count, 3, generator=generator, dtype=torch.float64)
    directions = directions / directions.norm(dim=-1, keepdim=True)
    magnitudes = torch.rand(count, 1, generator=generator, dtype=torch.float64) * maximum
    return directions * magnitudes


def _zero_case(
    *,
    num_betas: int = DEFAULT_BETAS,
    num_expression: int = DEFAULT_EXPRESSION,
    flat_hand_mean: bool = False,
    num_pca_comps: int | None = DEFAULT_PCA,
) -> dict[str, Any]:
    hand_width = 45 if num_pca_comps is None else num_pca_comps
    return {
        "num_betas": num_betas,
        "num_expression": num_expression,
        "flat_hand_mean": flat_hand_mean,
        "num_pca_comps": num_pca_comps,
        "betas": torch.zeros(num_betas, dtype=torch.float64),
        "expression": torch.zeros(num_expression, dtype=torch.float64),
        "global_orient": torch.zeros(3, dtype=torch.float64),
        "body_pose": torch.zeros(63, dtype=torch.float64),
        "jaw_pose": torch.zeros(3, dtype=torch.float64),
        "leye_pose": torch.zeros(3, dtype=torch.float64),
        "reye_pose": torch.zeros(3, dtype=torch.float64),
        "left_hand_pose": torch.zeros(hand_width, dtype=torch.float64),
        "right_hand_pose": torch.zeros(hand_width, dtype=torch.float64),
        "transl": torch.zeros(3, dtype=torch.float64),
    }


def _case_grid() -> dict[str, dict[str, Any]]:
    generator = torch.Generator().manual_seed(SEED)
    cases = {
        "zero_pca_mean": _zero_case(),
        "zero_pca_flat": _zero_case(flat_hand_mean=True),
        "zero_full_mean": _zero_case(num_pca_comps=None),
        "zero_full_flat": _zero_case(flat_hand_mean=True, num_pca_comps=None),
        "beta_plus_two": _zero_case(),
        "beta_minus_two": _zero_case(),
        "expression_plus_two": _zero_case(),
        "expression_minus_two": _zero_case(),
        "reduced_shape_expression": _zero_case(num_betas=5, num_expression=3),
        "jaw": _zero_case(),
        "left_eye": _zero_case(),
        "right_eye": _zero_case(),
        "face_all": _zero_case(),
        "pca_left": _zero_case(),
        "pca_right": _zero_case(),
        "pca_both": _zero_case(),
        "pca_twelve": _zero_case(num_pca_comps=12),
        "full_left": _zero_case(num_pca_comps=None),
        "full_right": _zero_case(num_pca_comps=None),
        "full_both": _zero_case(num_pca_comps=None),
        "root_wart": _zero_case(),
    }
    cases["beta_plus_two"]["betas"].fill_(2.0)
    cases["beta_minus_two"]["betas"].fill_(-2.0)
    cases["expression_plus_two"]["expression"].fill_(2.0)
    cases["expression_minus_two"]["expression"].fill_(-2.0)
    cases["reduced_shape_expression"]["betas"] = torch.linspace(-1.0, 1.0, 5, dtype=torch.float64)
    cases["reduced_shape_expression"]["expression"] = torch.tensor([0.5, -0.25, 0.75], dtype=torch.float64)
    cases["jaw"]["jaw_pose"] = torch.tensor([0.4, -0.2, 0.1], dtype=torch.float64)
    cases["left_eye"]["leye_pose"] = torch.tensor([0.2, 0.1, -0.15], dtype=torch.float64)
    cases["right_eye"]["reye_pose"] = torch.tensor([-0.1, 0.2, 0.15], dtype=torch.float64)
    cases["face_all"]["jaw_pose"] = _axis_angles(generator, 1)[0]
    cases["face_all"]["leye_pose"] = _axis_angles(generator, 1, 0.4)[0]
    cases["face_all"]["reye_pose"] = _axis_angles(generator, 1, 0.4)[0]
    cases["face_all"]["expression"] = torch.linspace(-1.0, 1.0, DEFAULT_EXPRESSION, dtype=torch.float64)
    cases["pca_left"]["left_hand_pose"] = torch.linspace(-0.7, 0.7, DEFAULT_PCA, dtype=torch.float64)
    cases["pca_right"]["right_hand_pose"] = torch.linspace(0.6, -0.6, DEFAULT_PCA, dtype=torch.float64)
    cases["pca_both"]["left_hand_pose"] = torch.randn(DEFAULT_PCA, generator=generator, dtype=torch.float64)
    cases["pca_both"]["right_hand_pose"] = torch.randn(DEFAULT_PCA, generator=generator, dtype=torch.float64)
    cases["pca_twelve"]["left_hand_pose"] = torch.randn(12, generator=generator, dtype=torch.float64) * 0.4
    cases["pca_twelve"]["right_hand_pose"] = torch.randn(12, generator=generator, dtype=torch.float64) * 0.4
    cases["full_left"]["left_hand_pose"] = _axis_angles(generator, 15).flatten()
    cases["full_right"]["right_hand_pose"] = _axis_angles(generator, 15).flatten()
    cases["full_both"]["left_hand_pose"] = _axis_angles(generator, 15).flatten()
    cases["full_both"]["right_hand_pose"] = _axis_angles(generator, 15).flatten()
    cases["root_wart"]["betas"] = torch.linspace(-1.5, 1.5, DEFAULT_BETAS, dtype=torch.float64)
    cases["root_wart"]["global_orient"] = torch.tensor([0.8, -0.6, 0.4], dtype=torch.float64)
    cases["root_wart"]["transl"] = torch.tensor([1.25, -0.75, 0.5], dtype=torch.float64)

    for index in range(RANDOM_CASES):
        num_pca_comps = DEFAULT_PCA if index % 2 == 0 else None
        case = _zero_case(flat_hand_mean=index % 3 == 0, num_pca_comps=num_pca_comps)
        axes = _axis_angles(generator, 25)
        case["betas"] = torch.randn(DEFAULT_BETAS, generator=generator, dtype=torch.float64).clamp(-2.0, 2.0)
        case["expression"] = torch.randn(DEFAULT_EXPRESSION, generator=generator, dtype=torch.float64).clamp(-2.0, 2.0)
        case["global_orient"] = axes[0]
        case["body_pose"] = axes[1:22].flatten()
        case["jaw_pose"], case["leye_pose"], case["reye_pose"] = axes[22], axes[23], axes[24]
        hand_width = DEFAULT_PCA if num_pca_comps is not None else 45
        case["left_hand_pose"] = torch.randn(hand_width, generator=generator, dtype=torch.float64) * 0.35
        case["right_hand_pose"] = torch.randn(hand_width, generator=generator, dtype=torch.float64) * 0.35
        case["transl"] = torch.randn(3, generator=generator, dtype=torch.float64) * 0.25
        cases[f"random_{index:02d}"] = case
    return cases


def _make_model(model_path: Path, case: dict[str, Any]) -> SMPLX:
    num_pca_comps = case["num_pca_comps"]
    return SMPLX(
        model_path=str(model_path),
        num_betas=case["num_betas"],
        num_expression_coeffs=case["num_expression"],
        dtype=torch.float64,
        use_pca=num_pca_comps is not None,
        num_pca_comps=DEFAULT_PCA if num_pca_comps is None else num_pca_comps,
        flat_hand_mean=case["flat_hand_mean"],
        create_betas=False,
        create_global_orient=False,
        create_body_pose=False,
        create_left_hand_pose=False,
        create_right_hand_pose=False,
        create_transl=False,
        create_expression=False,
        create_jaw_pose=False,
        create_leye_pose=False,
        create_reye_pose=False,
    )


def main() -> None:
    """Evaluate the fixed SMPL-X grid and write one archive per case."""
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    models: dict[tuple[int, int, bool, int | None], SMPLX] = {}
    for name, case in _case_grid().items():
        key = (
            case["num_betas"],
            case["num_expression"],
            case["flat_hand_mean"],
            case["num_pca_comps"],
        )
        if key not in models:
            models[key] = _make_model(args.model_path, case)
        model = models[key]
        with torch.no_grad():
            output = model(
                betas=case["betas"][None],
                expression=case["expression"][None],
                global_orient=case["global_orient"][None],
                body_pose=case["body_pose"][None],
                jaw_pose=case["jaw_pose"][None],
                leye_pose=case["leye_pose"][None],
                reye_pose=case["reye_pose"][None],
                left_hand_pose=case["left_hand_pose"][None],
                right_hand_pose=case["right_hand_pose"][None],
                transl=case["transl"][None],
            )
        np.savez_compressed(
            args.output_dir / f"{name}.npz",
            num_betas=np.int64(case["num_betas"]),
            num_expression=np.int64(case["num_expression"]),
            flat_hand_mean=np.bool_(case["flat_hand_mean"]),
            num_pca_comps=np.int64(-1 if case["num_pca_comps"] is None else case["num_pca_comps"]),
            betas=case["betas"].numpy(),
            expression=case["expression"].numpy(),
            global_orient=case["global_orient"].numpy(),
            body_pose=case["body_pose"].numpy(),
            jaw_pose=case["jaw_pose"].numpy(),
            leye_pose=case["leye_pose"].numpy(),
            reye_pose=case["reye_pose"].numpy(),
            left_hand_pose=case["left_hand_pose"].numpy(),
            right_hand_pose=case["right_hand_pose"].numpy(),
            transl=case["transl"].numpy(),
            vertices=output.vertices[0].numpy(),
            joints=output.joints[0, :55].numpy(),
        )


if __name__ == "__main__":
    main()
