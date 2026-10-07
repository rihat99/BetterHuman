"""Command-line entry point: ``python -m better_human.viz.studio``."""

from __future__ import annotations

import argparse

from .app import launch


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the HumanStudio viser app.")
    parser.add_argument("--family", default="smpl", choices=("smpl", "smplx", "mhr"))
    parser.add_argument("--gender", default="neutral", choices=("neutral", "male", "female"))
    parser.add_argument("--lod", type=int, default=1, help="MHR level of detail")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--data-dir", default="studio_data", help="Root for authored data")
    parser.add_argument("--models-dir", default=None, help="Weight root; else $BETTERHUMAN_MODELS_DIR or repo models/")
    parser.add_argument("--host", default="127.0.0.1")
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
        data_dir=args.data_dir,
        models_dir=args.models_dir,
    )


if __name__ == "__main__":
    main()
