# Skinning

A body's mesh follows its joints. This page explains the two kinds of shape input a family has, how BetterHuman
turns joint motion into vertex motion, and why there is exactly one way to do it for every family.

## Identity groups and stream groups

A family's blend shapes come in named groups of coefficients:

- **Identity groups** (`betas` for the SMPL family, `identity` and `proportions` for MHR, `shape` for SOMA) describe
  the subject. They are baked once: `human.bake(betas=...)` shapes the template mesh, regresses or fits the joints,
  places them in the subject's BetterRobot `Model` and integrates its masses. Changing them means a new bake.
- **Stream groups** (`expression` for SMPL-X, FLAME and MHR) change from frame to frame. They go to every
  `forward` call next to `q`, as `streams={"expression": coefficients}`, and add vertex offsets to the rest mesh of
  that frame.

A misspelled group name raises `UnknownNameError` naming the family's groups; an omitted group is zero.

## One skinning path

Every rig skins the same way (decision record [0003](../contracts/decisions/0003-one-skinning-path.md)). Bake gives
each native joint an inverse-bind affine, the map from the skinning rest frame into that joint's frame. Forward
kinematics gives each joint's world transform. Their product is the joint's skin affine, from the rest frame to the
world, and linear blend skinning mixes the skin affines of each vertex's joints with its skinning weights. An affine
may scale: MHR folds its joint scales into the inverse binds, so the same path serves it. Pose correctives, the
family's learned vertex offsets driven by the joint rotations, are added to the rest mesh before skinning.

The skin affines carry the baked rest joints to their posed positions, which is what makes the mesh follow the
skeleton. On the random-weight SMPL archive:

```python
import torch
import better_human as bh

human = bh.load("smpl", "tests/support/random_weights/smpl_random.npz")
betas = torch.zeros(10, requires_grad=True)
subject = human.bake(betas=betas)
state = subject.forward(subject.model.q_neutral)
affines, rest = state.skin_affines, subject.rest_joint_translations
moved = (affines[..., :3] @ rest[..., None]).squeeze(-1) + affines[..., 3]
print(tuple(affines.shape), torch.allclose(moved, state.joint_translations, atol=1e-6))
```
```output
(24, 3, 4) True
```

## Differentiable in shape and pose

Bake, forward kinematics, skinning and the mass integration are torch operations, so a loss on the vertices reaches
the identity coefficients, `q` and the stream coefficients through autograd. Continuing the program above:

```python
state.vertices[:, 1].sum().backward()
print(tuple(betas.grad.shape))
```
```output
(10,)
```

A `HumanState` computes each output (joint transforms, skin affines, the corrective features, the vertices) on
first use and keeps it, so a loss that reads the vertices twice skins once. A state is never changed: a new pose is
a new `forward`.

## Streams that move joints

In SMPL-X the expression moves the jaw and the eyes as well as the face. With `stream_joints=True` (the default)
the regressed rig shifts the rest joints by the stream's joint basis and re-derives the joint placements of that
frame's model, as upstream does; with `stream_joints=False` the stream moves vertices only, and the result differs
from upstream by about a millimetre (`tests/parity/smplx/test_smplx_parity.py` holds that band). On the random-weight
SMPL-X archive, without its hands:

```python
from better_human import HumanModel, families

spec = families.smplx("tests/support/random_weights/smplx_random.npz", hands=False)
human = HumanModel.from_spec(spec, density=None)
subject = human.bake(betas=torch.zeros(16))
q = subject.model.q_neutral.expand(3, -1)
still = subject.forward(q, streams={"expression": torch.zeros(3, 10)})
smile = subject.forward(q, streams={"expression": torch.ones(3, 10)})
print(tuple(smile.stream_offsets.shape), bool((smile.joint_translations != still.joint_translations).any()))
```
```output
(3, 10475, 3) True
```

## Where next

- [Architecture](architecture.md): the four records and the rig.
- [Segmentation](segmentation.md): how the skinned mesh splits into parts with masses.
