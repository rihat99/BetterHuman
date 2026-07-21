"""Interactively change SMPL shape and pose on the BetterRobot manifold.

Run without a viewer for CI or smoke checks::

    uv run python examples/01_smpl_shape_pose.py --headless --frames 1
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from better_robot.viewer import RenderContext, ViserBackend

import better_human as bh
from better_human.viz import SkinnedMeshMode


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, help="Converted SMPL .npz; otherwise use BETTERHUMAN_MODELS_DIR")
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


def main() -> None:
    """Run the live SMPL shape-and-pose demo."""
    arguments = _arguments()
    frame_limit = arguments.frames if arguments.frames is not None else (1 if arguments.headless else None)
    backend = _backend(arguments.headless, arguments.port)
    neutral = bh.SMPL(arguments.model_path, compute_mass=False)
    beta_values = [0.0] * 10
    tangent = torch.zeros(neutral.robot.nv)
    state = {"shape_dirty": False}

    def set_beta(index: int) -> Callable[[Any], None]:
        def callback(event: Any) -> None:
            beta_values[index] = float(event.target.value)
            state["shape_dirty"] = True

        return callback

    for index in range(10):
        backend.add_gui_slider(
            f"beta {index}",
            min=-3.0,
            max=3.0,
            step=0.1,
            value=0.0,
            callback=set_beta(index),
        )

    pose_controls = (
        ("left shoulder", "left_shoulder", 0),
        ("left elbow", "left_elbow", 1),
        ("right shoulder", "right_shoulder", 0),
    )
    for label, joint_name, axis in pose_controls:
        joint_id = neutral.robot.joint_id(joint_name)
        tangent_index = neutral.robot.idx_vs[joint_id] + axis

        def set_pose(event: Any, *, index: int = tangent_index) -> None:
            tangent[index] = float(event.target.value)

        backend.add_gui_slider(
            label,
            min=-1.5,
            max=1.5,
            step=0.05,
            value=0.0,
            callback=set_pose,
        )

    body = neutral
    mode = SkinnedMeshMode(body)
    context = RenderContext(backend=backend, namespace="/smpl")
    mode.attach(context, body.robot, body.fk(body.robot.q_neutral))
    rendered = 0
    started = time.perf_counter()
    try:
        while frame_limit is None or rendered < frame_limit:
            frame_started = time.perf_counter()
            if state["shape_dirty"]:
                mode.detach()
                body = neutral.with_shape(betas=torch.tensor(beta_values))
                mode = SkinnedMeshMode(body)
                mode.attach(context, body.robot, body.fk(body.robot.q_neutral))
                state["shape_dirty"] = False
            q = body.robot.integrate(body.robot.q_neutral, tangent)
            mode.update(body.fk(q))
            rendered += 1
            if not arguments.headless:
                time.sleep(max(0.0, 1.0 / 60.0 - (time.perf_counter() - frame_started)))
    except KeyboardInterrupt:
        pass
    finally:
        mode.detach()

    elapsed = time.perf_counter() - started
    print(f"Rendered {rendered} SMPL frames in {elapsed:.3f} s ({rendered / elapsed:.1f} FPS)")


if __name__ == "__main__":
    main()
