"""SMPL shape and pose sliders on the BetterRobot viewer.

uv run python examples/01_smpl_shape_pose.py            # open the printed URL
uv run python examples/01_smpl_shape_pose.py --check    # build the scene on the stand-in server, move sliders once, exit
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

import better_human as bh
from better_human.viz import HUMAN_THEME, HumanView
from better_robot import Viewer
from better_robot.kinematics import integrate
from better_robot.testing import FakeServer
from better_robot.viewer import GroundLayer

POSE_SLIDERS = (
    ("left shoulder", "left_shoulder", 0),
    ("left elbow", "left_elbow", 1),
    ("right shoulder", "right_shoulder", 0),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, help="SMPL .npz; otherwise $BETTERHUMAN_MODELS_DIR")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    human = bh.HumanModel.from_spec(bh.families.smpl(args.model_path, num_betas=10), density=None)
    betas = torch.zeros(10)
    tangent = torch.zeros(human.model.nv)
    identity = human.bake(betas=betas)
    view = HumanView(identity.forward(identity.model.q_neutral))

    viewer = Viewer(port=args.port, theme=HUMAN_THEME, server=FakeServer() if args.check else None)
    viewer.add([GroundLayer(plane="xz")], group="scene")
    viewer.add(view, group="smpl")

    def refresh() -> None:
        nonlocal identity
        view.update(identity.forward(integrate(identity.model, identity.model.q_neutral, tangent)))
        viewer.show()

    gui = viewer.server.gui
    with gui.add_folder("Shape"):
        for index in range(10):
            slider = gui.add_slider(f"beta {index}", min=-3.0, max=3.0, step=0.1, initial_value=0.0)

            def on_beta(_: object, index: int = index, slider: object = slider) -> None:
                nonlocal identity
                betas[index] = float(slider.value)
                identity = human.bake(betas=betas)
                refresh()

            slider.on_update(on_beta)
    with gui.add_folder("Pose"):
        for label, joint, axis in POSE_SLIDERS:
            coordinate = human.model.idx_vs[human.model.joint_id(joint)] + axis
            slider = gui.add_slider(label, min=-1.5, max=1.5, step=0.05, initial_value=0.0)

            def on_pose(_: object, coordinate: int = coordinate, slider: object = slider) -> None:
                tangent[coordinate] = float(slider.value)
                refresh()

            slider.on_update(on_pose)

    if args.check:
        betas[0] = 2.0
        identity = human.bake(betas=betas)
        tangent[human.model.idx_vs[human.model.joint_id("left_elbow")] + 1] = 1.0
        refresh()
        print(f"scene with {len(view.layers)} layers; elbow bent, beta0 = 2")
        return
    print(f"open {viewer.url}; Ctrl-C to exit")
    viewer.server.sleep_forever()


if __name__ == "__main__":
    main()
