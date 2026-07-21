"""Drive an SMPL wrist to a draggable target with BetterRobot IK.

Run without a viewer for CI or smoke checks::

    uv run python examples/03_ik_reach.py --headless --frames 1
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import torch
from better_robot import forward_kinematics
from better_robot.tasks import IKCostConfig, OptimizerConfig, solve_ik
from better_robot.viewer import RenderContext, ViserBackend

import better_human as bh
from better_human.viz import SkinnedMeshMode

WRIST_FRAME = "body_left_wrist"


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
    """Run the draggable wrist-target IK demo."""
    arguments = _arguments()
    frame_limit = arguments.frames if arguments.frames is not None else (1 if arguments.headless else None)
    backend = _backend(arguments.headless, arguments.port)
    body = bh.SMPL(arguments.model_path, compute_mass=False)
    q = body.robot.q_neutral.clone()
    neutral_data = forward_kinematics(body.robot, q, compute_frames=True)
    target = neutral_data.frame_pose_world[body.robot.frame_id(WRIST_FRAME)].clone()
    target[:3] += target.new_tensor([0.08, 0.08, 0.04])
    state = {"target": target, "dirty": True}

    def set_target(pose: torch.Tensor) -> None:
        state["target"] = pose.to(dtype=q.dtype, device=q.device)
        state["dirty"] = True

    backend.add_transform_control("/smpl/wrist_target", target, scale=0.12, on_update=set_target)
    mode = SkinnedMeshMode(body)
    mode.attach(
        RenderContext(backend=backend, namespace="/smpl"),
        body.robot,
        body.fk(q),
    )
    cost = IKCostConfig(pos_weight=10.0, ori_weight=0.2, limit_weight=0.0, rest_weight=0.005)
    optimizer = OptimizerConfig(max_iter=20, jacobian_strategy="analytic")
    rendered = 0
    solves = 0
    last_error = float("nan")
    started = time.perf_counter()
    try:
        while frame_limit is None or rendered < frame_limit:
            frame_started = time.perf_counter()
            if state["dirty"]:
                result = solve_ik(
                    body.robot,
                    {WRIST_FRAME: state["target"]},
                    initial_q=q,
                    cost_cfg=cost,
                    optimizer_cfg=optimizer,
                )
                q = result.q
                data = result.fk()
                wrist = data.frame_pose_world[body.robot.frame_id(WRIST_FRAME)]
                last_error = float((wrist[:3] - state["target"][:3]).norm())
                state["dirty"] = False
                solves += 1
            else:
                data = forward_kinematics(body.robot, q, compute_frames=True)
            mode.update(data)
            rendered += 1
            if not arguments.headless:
                time.sleep(max(0.0, 1.0 / 60.0 - (time.perf_counter() - frame_started)))
    except KeyboardInterrupt:
        pass
    finally:
        mode.detach()

    elapsed = time.perf_counter() - started
    print(
        f"Rendered {rendered} IK frames with {solves} solve(s) in {elapsed:.3f} s; "
        f"last wrist position error={last_error:.6f} m"
    )


if __name__ == "__main__":
    main()
