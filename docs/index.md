# BetterHuman

BetterHuman loads parametric human bodies (SMPL, SMPL-H, SMPL-X, MANO, FLAME, MHR and SOMA) as BetterRobot
`Model`s. A body is then just another robot: the same batched, differentiable forward kinematics, Jacobians,
dynamics, optimisers and viewer apply to it.

- **Load once.** `better_human.load(family)` reads converted weights and builds one `HumanModel` per family.
- **Bake a subject.** `human.bake(betas=...)` shapes the mesh and returns an `Identity` whose `model` is a plain
  BetterRobot `Model` with the subject's joint placements and mass.
- **Pose in batches.** `identity.forward(q)` runs BetterRobot forward kinematics and skins the mesh; vertices are
  computed on first use.
- **Differentiable end to end.** Shape, pose and mass are torch tensors; fit them with `better_robot.optim` and the
  terms in `better_human.terms`.

The program below runs on the random-weight SMPL archive the repository ships for its tests, so it needs no
licensed weights:

```python
import torch
import better_human as bh
from better_robot.kinematics import integrate

human = bh.load("smpl", "tests/support/random_weights/smpl_random.npz")
subject = human.bake(betas=torch.zeros(10))  # shape once
model = subject.model  # a BetterRobot Model
joints, vertices = len(human.spec.joint_names), human.spec.num_vertices
print(f"nq={model.nq} nv={model.nv} joints={joints} vertices={vertices}")

v = torch.zeros(4, model.nv)  # a batch of 4 tangent offsets
elbow = model.idx_vs[model.joint_id("left_elbow")]
v[:, elbow + 1] = torch.linspace(0.0, 1.5, 4)  # flex the left elbow
state = subject.forward(integrate(model, model.q_neutral, v))
print("vertices", tuple(state.vertices.shape), "joints", tuple(state.joint_translations.shape))
```
```output
nq=99 nv=75 joints=24 vertices=6890
vertices (4, 6890, 3) joints (4, 24, 3)
```

## The documentation

| Part | What it holds |
|------|---------------|
| [Getting started](getting-started/first-human.md) | install, then load, pose and view a body |
| Guides: [load a family](guides/load-a-family.md), [convert weights](guides/convert-weights.md), [use the studio](guides/use-the-studio.md), [run the tests](guides/run-the-tests.md), [run the benchmarks](guides/run-the-benchmarks.md) | one task each |
| Concepts: [architecture](concepts/architecture.md), [skinning](concepts/skinning.md), [segmentation](concepts/segmentation.md) | how BetterHuman works and why |
| Reference: [API](reference/api/better_human.md), [file formats](reference/formats/frame_set.md), [test suite](reference/tests.md), [benchmarks](reference/benchmarks.md), [glossary](reference/glossary.md), [roadmap](reference/roadmap.md) | names, shapes and files, looked up |
| [Contracts](contracts/index.md) | the rules the code follows, each with the test that enforces it |
