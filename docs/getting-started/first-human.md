# Your first human

This page installs BetterHuman, loads an SMPL body, bakes a subject, poses a batch of four frames and shows them in
the BetterRobot viewer. It needs no licensed weights: it reads the random-weight SMPL archive that the repository
ships for its tests, `tests/support/random_weights/smpl_random.npz` (decision record
[0016](../contracts/decisions/0016-random-weight-goldens.md)). Its arrays are random in SMPL's layout, so the body
looks like noise, but every call, name and shape on this page is the one the licensed SMPL file gives.

## Install with uv

BetterHuman depends on [BetterRobot](https://github.com/rihat99/BetterRobot); `uv sync` fetches the commit pinned in
`uv.lock`. The `studio` extra adds the browser viewer and HumanStudio.

```bash
cd BetterHuman
uv sync --extra studio
```

Run the programs below from the repository root, where the archive's relative path resolves.

## Load a body and bake a subject

`load` reads a weight file and builds one `HumanModel` for the family. `bake` applies an identity (the shape
coefficients `betas`) once and returns an `Identity`, whose `model` is a BetterRobot `Model`.

```python
import torch
import better_human as bh

human = bh.load("smpl", "tests/support/random_weights/smpl_random.npz")
subject = human.bake(betas=torch.zeros(10))
print(type(subject.model).__module__, subject.model.nq, subject.model.nv)
print(human.spec.joint_names[:4])
```
```output
better_robot.model.model 99 75
('pelvis', 'left_hip', 'right_hip', 'spine1')
```

Every SMPL joint is a ball joint (4 numbers in `q`, 3 in `v`) under a free-flyer root (7 and 6), hence
`nq = 7 + 23 * 4` and `nv = 6 + 23 * 3`.

## Pose a batch and run forward kinematics

`q` has `nq` entries and the tangent `v` has `nv`; build poses with `kinematics.integrate`, never by adding to `q`.
`forward` runs BetterRobot forward kinematics once for the whole batch.

```python
from better_robot.kinematics import integrate

model = subject.model
v = torch.zeros(4, model.nv)
elbow = model.idx_vs[model.joint_id("left_elbow")]
v[:, elbow + 1] = torch.linspace(0.0, 1.5, 4)
state = subject.forward(integrate(model, model.q_neutral, v))
print(type(state.kinematics).__name__, tuple(state.joint_translations.shape), tuple(state.vertices.shape))
```
```output
KinematicState (4, 24, 3) (4, 6890, 3)
```

`state.kinematics` is BetterRobot's `KinematicState`, so frame Jacobians and dynamics work on it unchanged.
`vertices` skins the mesh on first access and keeps the result.

## View the clip

`human_layers` turns a posed batch into viewer layers (the exact mesh and the native skeleton); the batch axis is the
timeline. The pelvis sits at the origin, so the ground goes under the lowest vertex.

```python
from better_robot import Viewer
from better_robot.viewer import GroundLayer
from better_human.viz import HUMAN_THEME, human_layers

viewer = Viewer(theme=HUMAN_THEME, ground=False)  # viser on 127.0.0.1:8080, +y up
viewer.add(GroundLayer(height=float(state.vertices[..., 1].min())))
viewer.add(human_layers(state), tags="smpl")
viewer.show(3)  # jump to the most bent frame
```

Open `http://127.0.0.1:8080` in a browser while the program runs; in a script, end with `viewer.serve()`.
`human_layers(state, mesh="skinned")` draws the body as BetterRobot's `RobotVisual` instead, skinned in the browser
from the joint poses (lighter on a long clip, without pose correctives), with the viewer's Style control.

## Where next

- [Load a family](../guides/load-a-family.md): your own weights, every family and its options.
- [Use the studio](../guides/use-the-studio.md): shape sliders, part painting and contact frames in the browser.
- [Architecture](../concepts/architecture.md): what BetterHuman adds on top of BetterRobot.
