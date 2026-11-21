# BetterHuman

BetterHuman loads parametric human bodies (SMPL, SMPL-H, SMPL-X, MANO, FLAME, MHR and SOMA) as batched,
differentiable [BetterRobot](https://github.com/rihat99/BetterRobot) models: bake a subject's shape once, then pose
whole batches with BetterRobot's kinematics, dynamics, solvers and browser viewer.

![SMPL-X in the viewer: the exact mesh with its skeleton, and the same pose skinned in the browser](docs/images/viewer_smplx.png)

> **Status:** this is an early version (0.1.0) in active development, and breaking changes may happen. Development
> happens on the `dev` branch.

## Install

```bash
git clone -b dev https://github.com/rihat99/BetterRobot   # this branch reads BetterRobot's dev from beside it
git clone https://github.com/rihat99/BetterHuman && cd BetterHuman
uv sync --extra studio   # Python 3.12+, PyTorch, BetterRobot, the browser viewer and the studio
```

## Quickstart

This needs no licensed weights: it reads the random-weight SMPL archive the tests use.

```python
import torch
import better_robot as br
import better_human as bh

human = bh.load("smpl", "tests/support/random_weights/smpl_random.npz")
subject = human.bake(betas=torch.zeros(10))  # the shape, baked once into a BetterRobot Model with part masses
q = subject.model.q_neutral.expand(4, -1).clone()  # (4, nq): a free-flyer root and 23 ball joints
q[:, 1] = torch.linspace(0.0, 1.5, 4)  # lift the root by 0 to 1.5 m (y up)
state = subject.forward(q)  # one batched query for the four poses
print(tuple(state.vertices.shape), tuple(state.joint_translations.shape))
print(br.center_of_mass(subject.model, state.kinematics)[:, 1])  # BetterRobot dynamics on the baked masses
```
```output
(4, 6890, 3) (4, 24, 3)
tensor([0.1660, 0.6660, 1.1660, 1.6660])
```

The archive holds random numbers in SMPL's layout, so the body looks like noise, but every name and shape is the
one the licensed SMPL file gives. `subject.model` is a plain BetterRobot `Model` and `state.kinematics` its
`KinematicState`, so Jacobians, dynamics and solvers apply unchanged.

## The families

BetterHuman ships no body weights. Download them from their authors, put them in one folder in the layout of
[docs/install.md](docs/install.md) and set `BETTERHUMAN_MODELS_DIR` to it; `bh.load(family)` then finds them.

| family | loader | identity groups | stream groups |
|---|---|---|---|
| SMPL | `bh.load("smpl")` | `betas` | |
| SMPL-H | `bh.load("smplh")` | `betas` | |
| SMPL-X | `bh.load("smplx")`; `bh.families.smplx(hands=, face=)` | `betas` | `expression` |
| MANO | `bh.load("mano")` | `betas` | |
| FLAME | `bh.load("flame")` | `betas` | `expression` |
| MHR | `bh.load("mhr")` (level of detail 1; `bh.families.mhr(lod=)`) | `identity`, `proportions` | `expression` |
| SOMA | `bh.load("soma")` (mid level of detail) | `shape` | |

Identity groups go to `bake`, stream groups to `forward` (`subject.forward(q, streams={"expression": ...})`).

## What is inside

| Package | What it does |
|---|---|
| `families` | One loader per family: weight file in, `FamilySpec` (mesh, skinning, joints, blend shapes) out. |
| `core` | `HumanModel`, `Identity` (a baked subject with its BetterRobot `Model`) and `HumanState` (a posed batch). |
| `mass` | Part masses and inertias from the closed mesh, so BetterRobot's dynamics apply. |
| `terms` | Fitting terms for BetterRobot's solvers: joints, markers, landmarks, reprojection, ground, priors, physics. |
| `io` | Where the weights are, contact-frame files, and the versioned formats every BetterHuman file carries. |
| `viz` | Bodies, skeletons and forces in BetterRobot's browser viewer. |
| `studio` | HumanStudio, a browser app to pose, shape, paint body parts and place contact frames. |

## Documentation

- The documentation starts at [docs/index.md](docs/index.md); `uv run mkdocs serve` serves it as a site.
- [examples/](examples/) holds four programs: shape and pose sliders on SMPL, partial SMPL-X, a reaching IK solve
  and MHR. Each takes `--check` to build its scene and exit.

## Licence

Code: Apache-2.0 ([LICENSE](LICENSE)). Body-model weights and data derived from them keep their own licences.
