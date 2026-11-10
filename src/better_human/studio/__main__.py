"""Command-line entry point: ``python -m better_human.studio --data-root DIR``; weights come from ``io.models_dir()``."""

from __future__ import annotations

import argparse

import numpy as np

from .app import DEFAULT_HOST, DEFAULT_PORT, launch


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the HumanStudio viser app.")
    parser.add_argument("--family", default="smpl", choices=("smpl", "smplx", "mhr"))
    parser.add_argument("--gender", default="neutral", choices=("neutral", "male", "female"))
    parser.add_argument("--lod", type=int, default=1, help="MHR level of detail")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--data-root", required=True, help="Directory the studio reads and writes its segmentations, frames and poses"
    )
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--seed", type=int, default=0, help="Seed of the random shapes the Body tab draws")
    return parser.parse_args()


def main() -> None:
    """Parse CLI arguments and run the studio (blocking)."""
    args = _parse_args()
    launch(
        family=args.family,
        gender=args.gender,
        lod=args.lod,
        port=args.port,
        host=args.host,
        data_root=args.data_root,
        generator=np.random.default_rng(args.seed),
    )


if __name__ == "__main__":
    main()
