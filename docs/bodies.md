# Bodies

`core` holds the records of a body. A `HumanModel` is a family ready to shape subjects. `human.bake(betas=...)`
applies a shape once and returns an `Identity`: the subject's mesh, joints and masses, and `identity.model`, a
BetterRobot `Model`. `identity.forward(q)` poses the subject and returns a `HumanState`. A state computes each output
the first time you read it and keeps it; a new pose is a new state.

The pose `q` is a BetterRobot configuration: a free-flyer root `[tx, ty, tz, qx, qy, qz, qw]`, then the family's
joints. Units are metres and radians, and the world is y-up in every family. `q` has `nq` numbers and a velocity
`nv`, so change a pose with `kinematics.integrate(model, q, v)`, never with `q + v`.

The names you will use:

| Name | What it does |
|---|---|
| `human.bake(**groups)` | an `Identity` from the identity groups (`betas`; MHR's `identity` and `proportions`; SOMA's `shape`) |
| `identity.forward(q, streams=)` | the `HumanState` of `q (B..., nq)`; `streams` holds per-frame groups such as `expression` |
| `HumanState.vertices` | `(B..., V, 3)` the posed mesh, pose correctives included |
| `HumanState.joint_translations`, `joint_poses` | `(B..., J, 3)` and `(B..., J, 7)`, the joints in the family's own order |
| `HumanState.contact_frame_poses(frames)`, `landmarks()`, `points(ids)` | named points on the mesh |
| `HumanState.kinematics` | BetterRobot's `KinematicState`, for Jacobians and dynamics |

## A batch of poses

Give `q` any leading batch axes; every output keeps them. Here are 2 x 5 poses of one SMPL-X subject:

```python
import torch
import better_human as bh
from better_robot.kinematics import integrate

human = bh.load("smplx", "tests/support/random_weights/smplx_random.npz")
subject = human.bake(betas=torch.zeros(16))
model = subject.model
q = integrate(model, model.q_neutral, 0.2 * torch.randn(2, 5, model.nv))
state = subject.forward(q)
print(state.vertices.shape)
print(state.joint_poses.shape)
print(type(state.kinematics).__name__, state.kinematics.frame_poses.shape)
```
```output
torch.Size([2, 5, 10475, 3])
torch.Size([2, 5, 55, 7])
KinematicState torch.Size([2, 5, 56, 7])
```

One identity drives a whole clip; a batch of identities can each take a pose too, since `q (B..., nq)` broadcasts
against the identity's own batch axes.

## Expressions and other streams

Identity groups describe the subject and are baked once. Stream groups change from frame to frame: SMPL-X, FLAME and
MHR have an `expression`, passed next to `q`. In SMPL-X it moves the jaw and the eyes too, as upstream does:

```python
q = model.q_neutral.expand(3, -1)
still = subject.forward(q, streams={"expression": torch.zeros(3, 10)})
smile = subject.forward(q, streams={"expression": torch.ones(3, 10)})
print(smile.stream_offsets.shape)
print(bool((smile.joint_translations != still.joint_translations).any()))
```
```output
torch.Size([3, 10475, 3])
True
```

A misspelled group raises `UnknownNameError` with the family's groups; a group you leave out is zero.

## Gradients reach shape and pose

Bake, forward kinematics, skinning and the masses are torch operations, so a loss on the vertices reaches the shape
coefficients and `q` through autograd:

```python
betas = torch.zeros(16, requires_grad=True)
q = model.q_neutral.clone().requires_grad_(True)
state = human.bake(betas=betas).forward(q)
state.vertices[:, 1].mean().backward()  # the mean height of the mesh
print(betas.grad.shape, q.grad.shape)
```
```output
torch.Size([16]) torch.Size([223])
```

## Faster repeated calls

When you call the same function many times, in a fitting loop or over a long clip in chunks, BetterRobot's
`kinematics.compiled` compiles it once with `torch.compile` and reuses the graph. It takes any function of tensors,
so a body's forward pass goes in as it is:

```python
from better_robot import kinematics

vertices = kinematics.compiled(lambda q: subject.forward(q).vertices)  # build it once
q = model.q_neutral.expand(64, -1)
with torch.no_grad():
    vertices(q)  # the first call runs eagerly
    vertices(q)  # the second compiles, which takes seconds
    print(torch.allclose(vertices(q), subject.forward(q).vertices, atol=1e-5))
```
```output
True
```

Gradients flow through the compiled call too; with gradients on it compiles a graph of its own. A new batch size
compiles again, at most three graphs per key (the first size, one for every later size, and batch 1).

On the licensed SMPL-X body at batch 64 (float32, one call), the compiled forward pass takes 8.3 ms against 16 ms on
8 threads of an AMD EPYC 9654 and its gradient 36 ms against 49 ms; MHR gains less (40 ms against 44 ms). On an
NVIDIA RTX PRO 6000 that another job shared, the compiled gradient took a quarter to a half of the eager time
(SMPL-X 1.2 to 4.9 ms against 4.6 to 10 ms, MHR 2.9 to 5.8 ms against 11 to 19 ms), and the forward pass's gain
depended on that job's load. Compiling takes from a few seconds to about a minute per graph. On CPU, torch 2.13
aborts the process when it runs MHR's compiled backward at a batch above 1 (a fault in its vectorised CPU kernels):
compile MHR without gradients on CPU, or run it on a GPU.

See also: [Mass](mass.md), which gives the subject its inertias, and [Fitting terms](terms.md), which fits `q` and
`betas` to data.
