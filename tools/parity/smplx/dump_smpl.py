"""Dump deterministic float64 SMPL goldens with upstream smplx 0.1.28."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.sparse import csr_matrix
from smplx import SMPL
from smplx.utils import Struct

SEED = 20260720
RANDOM_CASES = 8
NUM_BETAS = 10


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path", type=Path, help="Converted neutral SMPL .npz")
    parser.add_argument("output_dir", type=Path, help="Directory for per-case .npz files")
    return parser.parse_args()


def _load_data_struct(model_path: Path) -> Struct:
    with np.load(model_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    if "J_regressor" not in arrays:
        arrays["J_regressor"] = csr_matrix(
            (
                arrays.pop("J_regressor_data"),
                arrays.pop("J_regressor_indices"),
                arrays.pop("J_regressor_indptr"),
            ),
            shape=tuple(int(value) for value in arrays.pop("J_regressor_shape")),
        )
    return Struct(**arrays)


def _axis_angles(generator: torch.Generator, count: int) -> torch.Tensor:
    directions = torch.randn(count, 3, generator=generator, dtype=torch.float64)
    directions = directions / directions.norm(dim=-1, keepdim=True)
    magnitudes = torch.rand(count, 1, generator=generator, dtype=torch.float64) * 1.5
    return directions * magnitudes


def _case_grid() -> dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    generator = torch.Generator().manual_seed(SEED)
    zeros_betas = torch.zeros(NUM_BETAS, dtype=torch.float64)
    zeros_orient = torch.zeros(3, dtype=torch.float64)
    zeros_pose = torch.zeros(69, dtype=torch.float64)
    zeros_transl = torch.zeros(3, dtype=torch.float64)
    cases = {"zero": (zeros_betas, zeros_orient, zeros_pose, zeros_transl)}

    for index in range(RANDOM_CASES):
        axes = _axis_angles(generator, 24)
        betas = torch.randn(NUM_BETAS, generator=generator, dtype=torch.float64).clamp(-2.0, 2.0)
        transl = torch.randn(3, generator=generator, dtype=torch.float64) * 0.25
        cases[f"random_{index:02d}"] = (betas, axes[0], axes[1:].flatten(), transl)

    beta_pose = _axis_angles(generator, 23).flatten()
    cases["beta_zero"] = (zeros_betas, zeros_orient, beta_pose, zeros_transl)
    cases["beta_plus_two"] = (torch.full((NUM_BETAS,), 2.0, dtype=torch.float64), zeros_orient, beta_pose, zeros_transl)
    cases["beta_minus_two"] = (
        torch.full((NUM_BETAS,), -2.0, dtype=torch.float64),
        zeros_orient,
        beta_pose,
        zeros_transl,
    )
    cases["root_wart"] = (
        torch.linspace(-1.5, 1.5, NUM_BETAS, dtype=torch.float64),
        torch.tensor([0.8, -0.6, 0.4], dtype=torch.float64),
        zeros_pose,
        torch.tensor([1.25, -0.75, 0.5], dtype=torch.float64),
    )
    return cases


def main() -> None:
    """Evaluate the fixed grid and write one compressed archive per case."""
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_struct = _load_data_struct(args.model_path)
    model = SMPL(
        model_path=str(args.model_path),
        data_struct=data_struct,
        num_betas=NUM_BETAS,
        dtype=torch.float64,
        create_betas=False,
        create_global_orient=False,
        create_body_pose=False,
        create_transl=False,
    )
    joint_count = int(data_struct.weights.shape[-1])

    for name, (betas, global_orient, body_pose, transl) in _case_grid().items():
        with torch.no_grad():
            output = model(
                betas=betas[None],
                global_orient=global_orient[None],
                body_pose=body_pose[None],
                transl=transl[None],
            )
        np.savez_compressed(
            args.output_dir / f"{name}.npz",
            betas=betas.numpy(),
            global_orient=global_orient.numpy(),
            body_pose=body_pose.numpy(),
            transl=transl.numpy(),
            vertices=output.vertices[0].numpy(),
            joints=output.joints[0, :joint_count].numpy(),
        )


if __name__ == "__main__":
    main()
