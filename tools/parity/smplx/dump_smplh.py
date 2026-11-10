"""Dump deterministic float64 SMPL+H goldens with upstream smplx 0.1.28.

The SMPL+H file BetterHuman reads (AMASS's layout) ships no hand PCA, so its hand poses are per-joint axis-angles
with no mean pose: the oracle runs with ``use_pca=False`` and ``flat_hand_mean=True``. smplx's ``SMPLH`` still reads
``hands_components*`` and ``hands_mean*`` when it is built, so the dump hands it zeros, which those two options never
use. smplx 0.1.28 reads at most 10 betas from a file with fewer than 300 shape columns (AMASS's has 16), so the grid
draws 10.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import torch
from smplx import SMPLH
from smplx.utils import Struct

SEED = 20260720
RANDOM_CASES = 8
NUM_BETAS = 10
HAND_DIM = 45


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("model_path", type=Path, help="SMPL+H .npz (AMASS layout)")
    parser.add_argument("output_dir", type=Path, help="Directory for per-case .npz files")
    return parser.parse_args()


def _asset_sha256(directory: Path, names: tuple[str, ...]) -> str:
    """Provenance ``asset_sha256``: sha256 of the ``sha256sum`` listing of the oracle's asset files, names sorted.

    Shell equivalent: ``cd DIR && sha256sum NAMES | sha256sum`` (decision record 0014).
    """
    lines = []
    for name in sorted(names):
        digest = hashlib.sha256()
        with (directory / name).open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 24), b""):
                digest.update(block)
        lines.append(f"{digest.hexdigest()}  {name}\n")
    return hashlib.sha256("".join(lines).encode()).hexdigest()


def _load_data_struct(model_path: Path) -> Struct:
    with np.load(model_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    for side in "lr":
        arrays[f"hands_components{side}"] = np.zeros((HAND_DIM, HAND_DIM))
        arrays[f"hands_mean{side}"] = np.zeros(HAND_DIM)
    return Struct(**arrays)


def _axis_angles(generator: torch.Generator, count: int) -> torch.Tensor:
    directions = torch.randn(count, 3, generator=generator, dtype=torch.float64)
    directions = directions / directions.norm(dim=-1, keepdim=True)
    magnitudes = torch.rand(count, 1, generator=generator, dtype=torch.float64) * 1.5
    return directions * magnitudes


def _case(
    *,
    betas: torch.Tensor | None = None,
    global_orient: torch.Tensor | None = None,
    body_pose: torch.Tensor | None = None,
    left_hand_pose: torch.Tensor | None = None,
    right_hand_pose: torch.Tensor | None = None,
    transl: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """One case; an omitted field is zero."""
    fields = {
        "betas": (betas, NUM_BETAS),
        "global_orient": (global_orient, 3),
        "body_pose": (body_pose, 63),
        "left_hand_pose": (left_hand_pose, HAND_DIM),
        "right_hand_pose": (right_hand_pose, HAND_DIM),
        "transl": (transl, 3),
    }
    return {
        name: torch.zeros(width, dtype=torch.float64) if value is None else value
        for name, (value, width) in fields.items()
    }


def _case_grid() -> dict[str, dict[str, torch.Tensor]]:
    generator = torch.Generator().manual_seed(SEED)
    cases = {"zero": _case()}
    for index in range(RANDOM_CASES):
        axes = _axis_angles(generator, 52)
        cases[f"random_{index:02d}"] = _case(
            betas=torch.randn(NUM_BETAS, generator=generator, dtype=torch.float64).clamp(-2.0, 2.0),
            global_orient=axes[0],
            body_pose=axes[1:22].flatten(),
            left_hand_pose=axes[22:37].flatten(),
            right_hand_pose=axes[37:52].flatten(),
            transl=torch.randn(3, generator=generator, dtype=torch.float64) * 0.25,
        )

    pose = _axis_angles(generator, 51).flatten()
    body_pose, left_hand_pose, right_hand_pose = pose[:63], pose[63:108], pose[108:]
    for name, beta in (("beta_zero", 0.0), ("beta_plus_two", 2.0), ("beta_minus_two", -2.0)):
        cases[name] = _case(
            betas=torch.full((NUM_BETAS,), beta, dtype=torch.float64),
            body_pose=body_pose,
            left_hand_pose=left_hand_pose,
            right_hand_pose=right_hand_pose,
        )
    cases["left_hand"] = _case(left_hand_pose=_axis_angles(generator, 15).flatten())
    cases["right_hand"] = _case(right_hand_pose=_axis_angles(generator, 15).flatten())
    cases["root_wart"] = _case(
        betas=torch.linspace(-1.5, 1.5, NUM_BETAS, dtype=torch.float64),
        global_orient=torch.tensor([0.8, -0.6, 0.4], dtype=torch.float64),
        transl=torch.tensor([1.25, -0.75, 0.5], dtype=torch.float64),
    )
    return cases


def main() -> None:
    """Evaluate the fixed grid and write one compressed archive per case."""
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_struct = _load_data_struct(args.model_path)
    model = SMPLH(
        model_path=str(args.model_path),
        data_struct=data_struct,
        num_betas=NUM_BETAS,
        use_pca=False,
        flat_hand_mean=True,
        dtype=torch.float64,
        create_betas=False,
        create_global_orient=False,
        create_body_pose=False,
        create_left_hand_pose=False,
        create_right_hand_pose=False,
        create_transl=False,
    )
    joint_count = int(data_struct.weights.shape[-1])
    provenance = {
        "oracle": f"smplx=={importlib.metadata.version('smplx')}",
        "generator": "tools/parity/smplx/dump_smplh.py",
        "asset_sha256": _asset_sha256(args.model_path.parent, (args.model_path.name,)),
        "dtype": "float64",
        "seed": SEED,
    }

    for name, case in _case_grid().items():
        with torch.no_grad():
            output = model(**{field: value[None] for field, value in case.items()})
        np.savez_compressed(
            args.output_dir / f"{name}.npz",
            **{field: value.numpy() for field, value in case.items()},
            vertices=output.vertices[0].numpy(),
            joints=output.joints[0, :joint_count].numpy(),
            provenance=np.asarray(json.dumps(provenance, sort_keys=True)),
        )


if __name__ == "__main__":
    main()
