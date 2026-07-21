"""Render an identity-shaped and articulated MHR LOD1 body.

Run without a viewer for CI or smoke checks::

    uv run python examples/04_mhr_shape_pose.py --headless --frames 1
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time
from typing import Any

import torch
from better_robot.viewer import RenderContext, ViserBackend

import better_human as bh
from better_human.viz import SkinnedMeshMode


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, help="Converted MHR LOD1 .npz; otherwise use BETTERHUMAN_MODELS_DIR")
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


def _set_named_pose(q: torch.Tensor, body: bh.MHR, joint_name: str, value: float) -> None:
    joint_id = body.robot.joint_id(joint_name)
    q[body.robot.idx_qs[joint_id]] = value


def main() -> None:
    """Run the MHR shape, hand-scale, and named-pose demo."""
    arguments = _arguments()
    frame_limit = arguments.frames if arguments.frames is not None else (1 if arguments.headless else None)
    backend = _backend(arguments.headless, arguments.port)
    neutral = bh.MHR(arguments.model_path, lod=1, compute_mass=False)
    identity = torch.zeros(45)
    identity[0] = 0.8
    proportions = torch.zeros(73)
    proportions[11] = 0.12
    proportions[14] = 0.18
    body = neutral.with_shape(
        identity=identity,
        expression=torch.zeros(72),
        proportions=proportions,
    )
    q = body.robot.q_neutral.clone()
    for joint_name, value in (
        ("l_uparm_rz", -0.6),
        ("l_elbow_bend", 0.8),
        ("r_uparm_rz", 0.45),
    ):
        _set_named_pose(q, body, joint_name, value)

    mode = SkinnedMeshMode(body, rgba=(0.58, 0.78, 0.96, 1.0))
    mode.attach(
        RenderContext(backend=backend, namespace="/mhr"),
        body.robot,
        body.fk(q),
    )
    rendered = 0
    started = time.perf_counter()
    try:
        while frame_limit is None or rendered < frame_limit:
            frame_started = time.perf_counter()
            mode.update(body.fk(q))
            rendered += 1
            if not arguments.headless:
                time.sleep(max(0.0, 1.0 / 60.0 - (time.perf_counter() - frame_started)))
    except KeyboardInterrupt:
        pass
    finally:
        mode.detach()

    elapsed = time.perf_counter() - started
    print(f"Rendered {rendered} MHR LOD1 frames in {elapsed:.3f} s with {body.structure.num_vertices} vertices")


if __name__ == "__main__":
    main()
