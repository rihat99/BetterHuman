"""Full SMPL-X next to a body-only SMPL-X (hands and face dropped), same pose, one metre apart.

    uv run python examples/02_smplx_partial.py [--check]
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import torch

import better_human as bh
from better_human.families.classic import SMPLXClassic, smplx_from_classic
from better_human.viz import human_layers
from better_robot.viewer import DARK, GroundLayer, Viewer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    full = bh.load("smplx", args.model_path, num_betas=10, num_expression=10, mass=False)
    partial = bh.load("smplx", args.model_path, num_betas=10, hands=False, face=False, mass=False)
    torch.manual_seed(0)
    frames = 90
    phase = torch.linspace(0.0, 2.0 * torch.pi, frames)
    body_pose = torch.zeros(frames, 63)
    body_pose[:, 3 * 15 + 2] = 0.8 * torch.sin(phase)  # left shoulder (joint 16) about z
    body_pose[:, 3 * 17 + 1] = 0.6 * torch.sin(phase)  # left elbow (joint 18) about y
    hands = 0.5 * torch.sin(phase)[:, None] * torch.ones(frames, 6)
    expression = 2.0 * torch.sin(phase)[:, None] * torch.ones(frames, 10)

    viewer = Viewer(port=args.port, theme=replace(DARK, up="+y"))
    viewer.add(GroundLayer(plane="xz"), group="scene")
    for body, name, classic, offset in (
        (full, "full", SMPLXClassic(body_pose=body_pose, left_hand_pose=hands, right_hand_pose=hands, expression=expression, num_pca_comps=6), (-0.5, 0.0, 0.0)),
        (partial, "body only", SMPLXClassic(body_pose=body_pose), (0.5, 0.0, 0.0)),
    ):
        identity = body.bake()
        q, stream = smplx_from_classic(identity, classic)
        for layer in human_layers(identity.forward(q, stream)):
            viewer.add(layer, group=name, offset=offset)
    if args.check:
        viewer.show(45)
        print(f"{frames} frames, {len(viewer.layers)} layers; full nq={full.model.nq}, body-only nq={partial.model.nq}")
        viewer.server.stop()
        return
    print(f"open {viewer.url}; Ctrl-C to exit")
    viewer.play()


if __name__ == "__main__":
    main()
