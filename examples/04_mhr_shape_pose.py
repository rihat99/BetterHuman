"""MHR LOD1: identity sliders and an arm swing on the BetterRobot viewer.

    uv run python examples/04_mhr_shape_pose.py [--check]
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import torch

import better_human as bh
from better_human.viz import HumanView
from better_robot.viewer import LIGHT, GroundLayer, Viewer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    body = bh.load("mhr", args.model_path, lod=1, mass=False)
    coefficients = torch.zeros(45)
    tangent = torch.zeros(body.model.nv)
    identity = body.bake(identity=coefficients)
    view = HumanView(identity.forward(identity.q_neutral))
    viewer = Viewer(port=args.port, theme=replace(LIGHT, up="+y"))
    viewer.add(GroundLayer(plane="xz"), group="scene")
    for layer in view:
        viewer.add(layer, group="mhr")

    def refresh() -> None:
        view.update(identity.forward(identity.model.integrate(identity.q_neutral, tangent)))
        viewer.show()

    gui = viewer.server.gui
    with gui.add_folder("Identity"):
        for index in range(8):
            slider = gui.add_slider(f"identity {index}", min=-3.0, max=3.0, step=0.1, initial_value=0.0)

            def on_identity(_: object, index: int = index, slider: object = slider) -> None:
                nonlocal identity
                coefficients[index] = float(slider.value)
                identity = body.bake(identity=coefficients)
                refresh()

            slider.on_update(on_identity)
    arm = body.model.idx_vs[body.model.joint_id("l_shoulder_rx")] if "l_shoulder_rx" in body.model.joint_names else 6
    with gui.add_folder("Pose"):
        slider = gui.add_slider("left shoulder", min=-1.5, max=1.5, step=0.05, initial_value=0.0)
        slider.on_update(lambda _: (tangent.__setitem__(arm, float(slider.value)), refresh()))

    if args.check:
        coefficients[0] = 2.0
        identity = body.bake(identity=coefficients)
        tangent[arm] = 1.0
        refresh()
        print(f"MHR nq={body.model.nq}, {body.num_vertices} vertices; identity0 = 2, shoulder bent")
        viewer.server.stop()
        return
    print(f"open {viewer.url}; Ctrl-C to exit")
    viewer.server.sleep_forever()


if __name__ == "__main__":
    main()
