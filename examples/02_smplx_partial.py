"""Compare full and body-only SMPL-X spawns side by side.

Run without a viewer for CI or smoke checks::

    uv run python examples/02_smplx_partial.py --headless --frames 1
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import torch
from better_robot.viewer import RenderContext, ViserBackend

import better_human as bh
from better_human.viz import SkinnedMeshMode


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, help="SMPL-X .npz; otherwise use BETTERHUMAN_MODELS_DIR")
    parser.add_argument("--headless", action="store_true", help="Use BetterRobot's recording backend")
    parser.add_argument("--frames", type=int, help="Stop after this many frames; headless defaults to one")
    parser.add_argument("--port", type=int, default=8080)
    arguments = parser.parse_args()
    if arguments.frames is not None and arguments.frames < 1:
        parser.error("--frames must be positive")
    return arguments


def _backend(headless: bool, port: int) -> Any:
    if headless:
        from better_robot.viewer.renderers.testing import MockBackend  # noqa: PLC0415

        return MockBackend()
    return ViserBackend(port=port)


def _translated_neutral(body: bh.ParametricBody, x: float) -> torch.Tensor:
    tangent = body.robot.q_neutral.new_zeros(body.robot.nv)
    tangent[0] = x
    return body.robot.integrate(body.robot.q_neutral, tangent)


def main() -> None:
    """Run the full-versus-partial SMPL-X demo."""
    arguments = _arguments()
    frame_limit = arguments.frames if arguments.frames is not None else (1 if arguments.headless else None)
    backend = _backend(arguments.headless, arguments.port)
    full = bh.SMPLX(arguments.model_path, compute_mass=False)
    body_only = bh.SMPLX(
        arguments.model_path,
        use_hands=False,
        use_face=False,
        compute_mass=False,
    )
    full_q = _translated_neutral(full, -0.65)
    body_only_q = _translated_neutral(body_only, 0.65)
    full_mode = SkinnedMeshMode(full, rgba=(0.55, 0.75, 0.95, 1.0))
    body_mode = SkinnedMeshMode(body_only, rgba=(0.95, 0.68, 0.50, 1.0))
    full_mode.attach(
        RenderContext(backend=backend, namespace="/smplx/full"),
        full.robot,
        full.fk(full_q),
    )
    body_mode.attach(
        RenderContext(backend=backend, namespace="/smplx/body_only"),
        body_only.robot,
        body_only.fk(body_only_q),
    )

    rendered = 0
    started = time.perf_counter()
    try:
        while frame_limit is None or rendered < frame_limit:
            frame_started = time.perf_counter()
            full_mode.update(full.fk(full_q))
            body_mode.update(body_only.fk(body_only_q))
            rendered += 1
            if not arguments.headless:
                time.sleep(max(0.0, 1.0 / 60.0 - (time.perf_counter() - frame_started)))
    except KeyboardInterrupt:
        pass
    finally:
        full_mode.detach()
        body_mode.detach()

    elapsed = time.perf_counter() - started
    print(
        f"Rendered {rendered} comparison frames in {elapsed:.3f} s; "
        f"native joints: full={len(full.structure.joint_names)}, body-only={len(body_only.structure.joint_names)}"
    )


if __name__ == "__main__":
    main()
