"""Drag a target, the SMPL left wrist follows (LM on ``better_human.terms`` + ``better_robot.terms``).

    uv run python examples/03_ik_reach.py [--check]
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

import better_human as bh
from better_human import terms
from better_human.viz import HumanView
from better_robot.optim import LevenbergMarquardt, Problem, solve
from better_robot.terms import RobotManifold
from better_robot.viewer import LIGHT, GroundLayer, Viewer
from better_robot.viewer.tools import PoseGizmo

WRIST = 20  # native joint index of left_wrist


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    body = bh.load("smpl", args.model_path, num_betas=10, mass=False)
    identity = body.bake()
    model = identity.model
    q = identity.q_neutral[None].clone()
    state = identity.forward(q)
    view = HumanView(state)
    viewer = Viewer(port=args.port, theme=replace(LIGHT, up="+y"))
    viewer.add(GroundLayer(plane="xz"), group="scene")
    for layer in view:
        viewer.add(layer, group="smpl")

    target = state.joint_poses[0, WRIST].clone()
    problem = Problem({"q": q}, manifolds={"q": RobotManifold(model)}, data={"target": target[None, :3]})
    problem.cached(terms.human_state(identity), name="state")
    problem.residual("wrist", terms.joints(target="target", joint_ids=[WRIST]))
    problem.residual("prior", terms.pose_prior(identity), weight=0.05)
    current = {"q": q}

    def follow(pose: np.ndarray) -> None:
        problem.data["target"] = torch.from_numpy(pose[:3]).to(q)[None]
        result = solve(problem, LevenbergMarquardt(), iters=20, x=current)
        current["q"] = result.x["q"]
        view.update(identity.forward(current["q"]))
        viewer.show()

    gizmo = PoseGizmo(viewer, "wrist target", target.numpy(), on_change=follow, scale=0.1)
    if args.check:
        moved = target.clone()
        moved[:3] += torch.tensor([0.15, 0.10, 0.05])
        follow(moved.numpy())
        reached = identity.forward(current["q"]).joints[0, WRIST]
        print(f"target moved, residual {float((reached - moved[:3]).norm()):.1e} m")
        viewer.server.stop()
        return
    print(f"drag the gizmo (at {gizmo.pose[:3]}) at {viewer.url}; Ctrl-C to exit")
    viewer.server.sleep_forever()


if __name__ == "__main__":
    main()
