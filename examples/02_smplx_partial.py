"""Full SMPL-X next to a body-only SMPL-X (hands and face dropped), same pose, one metre apart.

uv run python examples/02_smplx_partial.py [--check]   # --check: build on the stand-in server, show one frame, exit
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

import better_human as bh
from better_human.families import SMPLXClassic
from better_human.viz import HUMAN_THEME, human_layers
from better_robot.testing import FakeServer
from better_robot.viewer import GroundLayer, Viewer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    spec = bh.families.smplx(args.model_path, num_betas=10, num_expression=10, num_pca_comps=6)
    full = bh.HumanModel.from_spec(spec, density=None)
    # The face and hand joints follow the 22 body joints; dropping them from the same spec reads the file once.
    partial = bh.HumanModel.from_spec(bh.families.drop_joints(spec, spec.joint_names[22:]), density=None)
    torch.manual_seed(0)
    frames = 90
    phase = torch.linspace(0.0, 2.0 * torch.pi, frames)
    body_pose = torch.zeros(frames, 63)
    body_pose[:, 3 * 15 + 2] = 0.8 * torch.sin(phase)  # left shoulder (joint 16) about z
    body_pose[:, 3 * 17 + 1] = 0.6 * torch.sin(phase)  # left elbow (joint 18) about y
    hands = 0.5 * torch.sin(phase)[:, None] * torch.ones(frames, 6)
    expression = 2.0 * torch.sin(phase)[:, None] * torch.ones(frames, 10)

    viewer = Viewer(port=args.port, theme=HUMAN_THEME, server=FakeServer() if args.check else None)
    viewer.add([GroundLayer(plane="xz")], group="scene")
    for body, name, classic, offset in (
        (
            full,
            "full",
            SMPLXClassic(body_pose=body_pose, left_hand_pose=hands, right_hand_pose=hands, expression=expression),
            (-0.5, 0.0, 0.0),
        ),
        (partial, "body only", SMPLXClassic(body_pose=body_pose), (0.5, 0.0, 0.0)),
    ):
        identity, q, stream = bh.from_classic(body, classic)
        viewer.add(human_layers(identity.forward(q, stream)), group=name, offset=offset)
    if args.check:
        viewer.show(45)
        print(f"{frames} frames, {len(viewer.layers)} layers; full nq={full.model.nq}, body-only nq={partial.model.nq}")
        return
    print(f"open {viewer.url}; Ctrl-C to exit")
    viewer.play()


if __name__ == "__main__":
    main()
