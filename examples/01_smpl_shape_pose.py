"""SMPL shape and pose sliders on the BetterRobot viewer.

uv run python examples/01_smpl_shape_pose.py            # open the printed URL
uv run python examples/01_smpl_shape_pose.py --check    # on the stand-in server: bend the elbow once, check it, exit
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

POSE_SLIDERS = (
    ("left shoulder", "left_shoulder", 0),
    ("left elbow", "left_elbow", 1),
    ("right shoulder", "right_shoulder", 0),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, help="SMPL .npz; otherwise $BETTERHUMAN_MODELS_DIR")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--port", type=int, help="default 8080, or the next free port")
    args = parser.parse_args()

    human = bh.HumanModel.from_spec(bh.families.smpl(args.model_path, num_betas=10), density=None)
    betas = torch.zeros(10)
    tangent = torch.zeros(human.model.nv)
    identity = human.bake(betas=betas)
    state = identity.forward(identity.model.q_neutral)
    view = HumanView(state)

    viewer = Viewer(port=args.port, theme=HUMAN_THEME, ground=False, server=FakeServer() if args.check else None)
    viewer.add(GroundLayer(height=float(state.vertices[:, 1].min())))  # under the feet: the pelvis is at the origin
    viewer.add(view, tags="smpl")

    def refresh() -> None:
        view.update(identity.forward(integrate(identity.model, identity.model.q_neutral, tangent)))

    def on_beta(value: float, index: int) -> None:
        nonlocal identity
        betas[index] = value
        identity = human.bake(betas=betas)
        refresh()

    def on_pose(value: float, coordinate: int) -> None:
        tangent[coordinate] = value
        refresh()

    for index in range(10):
        viewer.sidebar.slider(
            "Shape", f"beta {index}", low=-3.0, high=3.0, step=0.1, value=0.0,
            on_change=lambda value, index=index: on_beta(value, index),
        )  # fmt: skip
    for label, joint, axis in POSE_SLIDERS:
        coordinate = human.model.idx_vs[human.model.joint_id(joint)] + axis
        viewer.sidebar.slider(
            "Pose", label, low=-1.5, high=1.5, step=0.05, value=0.0,
            on_change=lambda value, coordinate=coordinate: on_pose(value, coordinate),
        )  # fmt: skip

    if args.check:
        betas[0] = 2.0
        identity = human.bake(betas=betas)
        refresh()
        mesh = view.layers[0]
        before = mesh.vertices.copy()
        tangent[human.model.idx_vs[human.model.joint_id("left_elbow")] + 1] = 1.0
        refresh()
        # The bend alone (shape drawn before it): the drawn forearm moves 0.48 m on SMPL, 0.73 m on the random archive.
        bend = float(np.linalg.norm(mesh.vertices - before, axis=-1).max())
        if bend < 0.05:
            sys.exit(f"--check: the elbow bend moved the drawn mesh {bend:.4f} m; expected over 0.05 m")
        print(f"scene with {len(view.layers)} layers; beta0 = 2, elbow bent: drawn mesh moved {bend:.2f} m")
        return
    print(f"open {viewer.url}; Ctrl-C to exit")
    viewer.serve()


if __name__ == "__main__":
    main()
