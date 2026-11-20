# Fitting terms

`terms` turns a body into least-squares problems for BetterRobot's solvers (`better_robot.optim`). You register the
posed state once per problem as a cached value; every term reads it, so one forward pass serves them all. Variables
are poses `q (B, nq)`, trajectories `q (B, T, nq)` and, for shape fits, the identity groups such as `betas`.

The names you will use:

| Name | What it does |
|---|---|
| `terms.human_state(identity)` | the cached `HumanState` of a fixed subject; `identity_name=` reads a fitted one |
| `terms.baked_identity(human, "betas")` | the cached `Identity` baked from shape variables |
| `terms.joints`, `terms.points`, `terms.landmarks` | 3D joints, mesh points (markers) and the family's landmarks against data |
| `terms.reprojection` | the same points in 2D, through a pinhole camera |
| `terms.ground` | mesh points below the ground plane |
| `terms.pose_prior`, `terms.shape_prior` | keep the pose near a mean and the shape near zero |
| `terms.root_wrench` | the root rows of inverse dynamics along a trajectory with contact forces: zero for a motion physics allows |
| `terms.contact_wrenches(model, state, forces=, frames=)` | world forces at named frames as the `external_forces` of BetterRobot's `inverse_dynamics` |

## Fit poses to joints

Eight random poses of an SMPL body give target joints; Levenberg-Marquardt recovers the poses from the neutral
pose, all eight in one problem:

```python
import torch
import better_human as bh
from better_human import terms
from better_robot.kinematics import integrate
from better_robot.optim import LevenbergMarquardt, Problem, solve
from better_robot.terms import RobotManifold

human = bh.load("smpl", "tests/support/random_weights/smpl_random.npz", density=None)
subject = human.bake()
model = subject.model
targets = subject.forward(integrate(model, model.q_neutral, 0.3 * torch.randn(8, model.nv))).joint_translations

q0 = model.q_neutral.expand(8, -1).clone()
problem = Problem({"q": q0}, manifolds={"q": RobotManifold(model)}, data={"joints": targets})
problem.cached(terms.human_state(subject), name="state")
problem.residual("joints", terms.joints(target="joints"))
result = solve(problem, LevenbergMarquardt(), iterations=30)
error = (subject.forward(result.x["q"]).joint_translations - targets).norm(dim=-1)
print(f"largest joint error: {float(error.max()):.1e} m")
```
```output
largest joint error: 1.1e-07 m
```

## Fit shape and pose to markers

Make `betas` a variable too: `baked_identity` bakes it inside the problem and `human_state` poses the result. Here
138 mesh points of a shaped and posed subject are the markers:

```python
markers = torch.arange(0, 6890, 50)  # vertex ids
truth = human.bake(betas=0.5 * torch.randn(10))
data = truth.forward(integrate(model, model.q_neutral, 0.1 * torch.randn(1, model.nv))).points(markers)

variables = {"q": model.q_neutral[None].clone(), "betas": torch.zeros(1, 10)}
problem = Problem(variables, manifolds={"q": RobotManifold(model)}, data={"markers": data})
problem.cached(terms.baked_identity(human, "betas"), name="identity")
problem.cached(terms.human_state(identity_name="identity"), name="state")
problem.residual("markers", terms.points(markers, target="markers"))
result = solve(problem, LevenbergMarquardt(), iterations=50)
fit = human.bake(betas=result.x["betas"]).forward(result.x["q"]).points(markers)
print(f"largest marker error: {float((fit - data).norm(dim=-1).max()):.1e} m")
```
```output
largest marker error: 1.5e-07 m
```

`examples/03_ik_reach.py` runs a pose fit live in the viewer: drag a target and the wrist follows.

See also: [Bodies](bodies.md) for the state the terms read, and BetterRobot's optimisation page for problems and
solvers.
