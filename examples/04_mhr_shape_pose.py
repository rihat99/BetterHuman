"""MHR LOD1: identity sliders and a left upper-arm swing on the BetterRobot viewer.

uv run python examples/04_mhr_shape_pose.py [--check]   # --check: on the stand-in server, swing the arm once, check, exit
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

import better_human as bh
from better_human.viz import HUMAN_THEME, HumanView
from better_robot import Viewer
from better_robot.kinematics import integrate
from better_robot.testing import FakeServer
from better_robot.viewer import GroundLayer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--port", type=int, help="default 8080, or the next free port")
    args = parser.parse_args()

    human = bh.load("mhr", args.model_path, density=None)
    coefficients = torch.zeros(45)
    tangent = torch.zeros(human.model.nv)
    identity = human.bake(identity=coefficients)
    state = identity.forward(identity.model.q_neutral)
    view = HumanView(state)
    viewer = Viewer(port=args.port, theme=HUMAN_THEME, ground=False, server=FakeServer() if args.check else None)
    viewer.add(GroundLayer(height=float(state.vertices[:, 1].min())))  # MHR stands on y = 0
    viewer.add(view, tags="mhr")

    def refresh() -> None:
        view.update(identity.forward(integrate(identity.model, identity.model.q_neutral, tangent)))

    def on_identity(value: float, index: int) -> None:
        nonlocal identity
        coefficients[index] = value
        identity = human.bake(identity=coefficients)
        refresh()

    for index in range(8):
        viewer.sidebar.slider(
            "Identity", f"identity {index}", low=-3.0, high=3.0, step=0.1, value=0.0,
            on_change=lambda value, index=index: on_identity(value, index),
        )  # fmt: skip
    # MHR's shoulder is its upper-arm joint, one revolute joint per axis.
    arm = human.model.idx_vs[human.model.joint_id("l_uparm_rz")]
    viewer.sidebar.slider(
        "Pose", "left upper arm", low=-1.5, high=1.5, step=0.05, value=0.0,
        on_change=lambda value: (tangent.__setitem__(arm, value), refresh()),
    )  # fmt: skip

    if args.check:
        coefficients[0] = 2.0
        identity = human.bake(identity=coefficients)
        refresh()
        joints, wrist = view.layers[1], human.spec.joint_names.index("l_wrist")
        before = joints.centers[0, wrist].copy()
        tangent[arm] = 1.0
        refresh()
        moved = float(np.linalg.norm(joints.centers[0, wrist] - before))
        if moved < 0.1:
            sys.exit(
                f"--check: the drawn left wrist moved {moved:.4f} m after the upper-arm swing; expected over 0.1 m"
            )
        print(
            f"MHR nq={human.model.nq}, {human.spec.num_vertices} vertices; identity0 = 2, left upper arm swung, "
            f"drawn wrist moved {moved:.2f} m"
        )
        return
    print(f"open {viewer.url}; Ctrl-C to exit")
    viewer.serve()


if __name__ == "__main__":
    main()
