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
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    human = bh.load("mhr", args.model_path, density=None)
    coefficients = torch.zeros(45)
    tangent = torch.zeros(human.model.nv)
    identity = human.bake(identity=coefficients)
    view = HumanView(identity.forward(identity.model.q_neutral))
    viewer = Viewer(port=args.port, theme=HUMAN_THEME, server=FakeServer() if args.check else None)
    viewer.add([GroundLayer(plane="xz")], group="scene")
    viewer.add(view, group="mhr")

    def refresh() -> None:
        view.update(identity.forward(integrate(identity.model, identity.model.q_neutral, tangent)))
        viewer.show()

    gui = viewer.server.gui
    with gui.add_folder("Identity"):
        for index in range(8):
            slider = gui.add_slider(f"identity {index}", min=-3.0, max=3.0, step=0.1, initial_value=0.0)

            def on_identity(_: object, index: int = index, slider: object = slider) -> None:
                nonlocal identity
                coefficients[index] = float(slider.value)
                identity = human.bake(identity=coefficients)
                refresh()

            slider.on_update(on_identity)
    # MHR's shoulder is its upper-arm joint, one revolute joint per axis.
    arm = human.model.idx_vs[human.model.joint_id("l_uparm_rz")]
    with gui.add_folder("Pose"):
        slider = gui.add_slider("left upper arm", min=-1.5, max=1.5, step=0.05, initial_value=0.0)
        slider.on_update(lambda _: (tangent.__setitem__(arm, float(slider.value)), refresh()))

    if args.check:
        coefficients[0] = 2.0
        identity = human.bake(identity=coefficients)
        refresh()
        skeleton, wrist = view.layers[1], human.spec.joint_names.index("l_wrist")
        before = skeleton.joints[0, wrist].copy()
        tangent[arm] = 1.0
        refresh()
        moved = float(np.linalg.norm(skeleton.joints[0, wrist] - before))
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
    viewer.server.sleep_forever()


if __name__ == "__main__":
    main()
