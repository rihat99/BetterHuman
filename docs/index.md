# BetterHuman

BetterHuman loads parametric human bodies (SMPL, SMPL-H, SMPL-X, MANO, FLAME and MHR) as
BetterRobot `Model`s. A body is then just another robot: the same batched, differentiable
forward kinematics, Jacobians, dynamics, optimisers and viewer apply to it.

## A body is a BetterRobot model

- **Load once.** `better_human.load(family)` reads converted weights and builds one `Body` per family.
- **Bake a subject.** `body.bake(betas=...)` shapes the mesh and returns an `Identity` whose
  `model` is a plain BetterRobot `Model` with the subject's joint placements and mass.
- **Pose in batches.** `identity.forward(q)` runs BetterRobot forward kinematics and skins the
  mesh; vertices are computed on first use.
- **Differentiable end to end.** Shape, pose and mass are torch tensors; fit them with
  `better_robot.optim` and the terms in `better_human.terms`.
- **Shared contracts.** Poses are `[tx, ty, tz, qx, qy, qz, qw]`, tangents are
  `[linear, angular]`, and `nq` differs from `nv`: change `q` with `model.integrate`.

```python
import torch
import better_human as bh

body = bh.load("smpl")                          # $BETTERHUMAN_MODELS_DIR/smpl/
subject = body.bake(betas=torch.zeros(10))      # shape once
model = subject.model                           # a BetterRobot Model
joints, vertices = len(body.joint_names), body.num_vertices
print(f"nq={model.nq} nv={model.nv} joints={joints} vertices={vertices}")

v = torch.zeros(4, model.nv)                    # a batch of 4 tangent offsets
elbow = model.idx_vs[model.joint_name_to_id["left_elbow"]]
v[:, elbow + 1] = torch.linspace(0.0, 1.5, 4)   # flex the left elbow
state = subject.forward(model.integrate(subject.q_neutral, v))
print("vertices", tuple(state.vertices.shape), "joints", tuple(state.joints.shape))
```
```output
nq=99 nv=75 joints=24 vertices=6890
vertices (4, 6890, 3) joints (4, 24, 3)
```

Every SMPL joint is a ball joint (4 numbers in `q`, 3 in `v`) under a free-flyer root
(7 and 6), hence `nq = 7 + 23·4` and `nv = 6 + 23·3`.

<div class="grid cards" markdown>

-   **Getting started**

    ---

    Install, point at the weights, load, pose and view a body.

    [Getting started](getting-started.md)

-   **Concepts**

    ---

    How the layers sit on BetterRobot, and how bodies split into parts.

    [Architecture](concepts/architecture.md)

-   **Studio**

    ---

    A browser app to look at a body, paint its parts and place contact frames.

    [HumanStudio](studio.md)

</div>
