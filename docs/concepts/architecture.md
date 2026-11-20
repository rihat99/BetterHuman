# Architecture

Every parametric body model ships in its own repository with its own conventions: its own forward pass, pose
format, file layout and quirks (SMPL's root translation is not at the root joint; MHR has its own rig; chumpy-era
weight files need an old NumPy to open). None of them speaks robotics: no configuration manifold, no Jacobians, no
dynamics, no solver, only a function from parameters to vertices. BetterHuman gives them one API, and builds each
body as a BetterRobot model, so the robotics comes from BetterRobot.

## A body is a BetterRobot model

A family is data. One `Rig` object holds the only family-specific behaviour. Shape is baked once per subject into a
BetterRobot `Model`; pose is a batched, differentiable BetterRobot query on it. BetterHuman adds only what is
human-specific; a capability that a robot would use too belongs in BetterRobot.

| Shared with BetterRobot | Human-specific (BetterHuman) |
|---|---|
| `Model`, `ModelBuilder`, `io.build_model` | weight-file readers, family tables, the MHR and SOMA rigs |
| `KinematicState`, Jacobians, `dynamics` | blend shapes, pose correctives, top-K skinning |
| `lie.so3`, `lie.se3`, `spatial.inertia_of_mesh` | per-part mass from mesh volume, contact frames |
| `optim` solvers, `Problem`, `TermFn` | `better_human.terms` (markers, reprojection, ground, ...) |
| `viewer` layers and `Viewer` | `better_human.viz` (`human_layers`, `HumanView`) and the studio |
| `exceptions` (`ShapeError`, `UnknownNameError`, ...) | `FormatError`, `AssetNotFoundError`, both under `BetterRobotError` |

Mass and inertia are first-class: models whose coordinates are independent support solve-based dynamics directly,
and meshes and skinning are outputs, not the organising principle.

## The package map

Each unit of `better_human` imports only the units of its row. This copy of the edge tables is
checked against `ALLOWED` and `BR_ALLOWED` in `tests/contract/test_layer_dependencies.py` by
`tests/docs/test_codemap.py`:

```text
# among BetterHuman's units
io           -> nothing
mass         -> nothing
core         -> mass
families     -> core io mass
terms        -> core
viz          -> core
studio       -> better_human core families io mass viz
better_human -> core families io mass studio terms viz
```

```text
# into BetterRobot
io           -> better_robot.exceptions
mass         -> better_robot.exceptions better_robot.lie better_robot.spatial
core         -> better_robot better_robot.exceptions better_robot.kinematics better_robot.lie
families     -> better_robot better_robot.exceptions better_robot.io better_robot.kinematics better_robot.lie
terms        -> better_robot better_robot.dynamics better_robot.exceptions better_robot.kinematics better_robot.optim
viz          -> better_robot.exceptions better_robot.lie better_robot.viewer
studio       -> better_robot better_robot.exceptions better_robot.lie better_robot.viewer
better_human -> better_robot.exceptions
```

`better_human` is the package root, and it lists every subpackage. The body model (`core`) and the file readers (`io`) stay usable without any
family table or viewer; `mass` reads no file: the family loaders read its data through `io` and hand it over.
`studio`, the optional application, sits on top of
everything but `terms`. Every BetterRobot import goes through BetterRobot's `__all__` tree, and a first-level name
such as `Viewer` comes from BetterRobot's root (`tests/contract/test_br_imports.py`).

## Four records

```text
weight file --families.<f>(path, dtype=, device=)--> FamilySpec --HumanModel.from_spec--> HumanModel
            --bake(**identity)--> Identity --forward(q, streams)--> HumanState
```

- **`FamilySpec`** (`src/better_human/core/spec.py`) is frozen, typed data: the template mesh, the skinning weights,
  the joint tree, the identity and stream blend shapes, the pose corrective, the parts and their density categories,
  the landmarks and the classic format. Its behaviour lives in `rig`; its methods (`joint_id`, `skin_matrix`) and
  properties (the sizes, dtype and device) only read the data.
- **`HumanModel`** (`src/better_human/core/human_model.py`) is built once from a spec with the body's options
  (`density`, `contact_frames`, `stream_joints`); `bh.load(family)` runs the family's loader with its defaults and
  `HumanModel.from_spec` in one step. Dtype and device are chosen there and never change.
- **`Identity`** is a baked subject: `human.bake(betas=...)` takes the identity groups by name, shapes the mesh,
  places the joints, integrates the masses and holds the subject's BetterRobot `Model`.
- **`HumanState`** is one query: `identity.forward(q, streams)` runs forward kinematics once and computes every
  output (joint transforms, skin affines, vertices, frame poses) on first use, then keeps it. A new pose is a new
  state; a state is never changed.

All four records are frozen; `HumanModel`, `Identity` and `HumanState` compare by identity.
The [coupling standard](../contracts/coupling-standard.md) gives the shape and layout of every `HumanState` output,
and [Bodies](../bodies.md) the names you use.

## The rig

`Rig` (`src/better_human/core/rig.py`) is abstract.
A family implements four members: `build_model` (the BetterRobot `Model` with its operational frames), `bake` (joint
placements, the skinning rest mesh, the rest joints and the inverse-bind affines of an identity), `joint_ids` (the
BetterRobot joint of each native joint) and `corrective_features` (what the pose corrective reads). Four hooks have
defaults: `joint_transforms` (by default BetterRobot's joint placements), `stream_joint_basis` and
`placements_from_rest` (for stream groups that move joints) and `without_joints` (for dropping joints). Three rigs
exist, each `@final`:

- `RegressedRig` (`src/better_human/families/regressed_rig.py`, the SMPL family): joints come from a joint
  regressor applied to the shaped vertices; it is the one rig that drops joints.
- `MHRRig` (`src/better_human/families/mhr_rig.py`): each native joint is a chain of revolute joints, one per
  rotation entry of MHR's parameter transform, with mimic joints for parameters that drive several entries;
  proportions set the joint offsets and scales, and the scales fold into the inverse-bind affines.
- `SOMARig` (`src/better_human/families/soma_rig.py`): bake fits the template skeleton to the shaped mesh and
  re-poses both to the template bind, as upstream; its `joint_transforms` turns the twist helpers by the swing-twist
  of their limb segments.

Family behaviour lives in the rig or the loader, never in a branch of `HumanModel`.

## The pose `q`

`q` is a BetterRobot configuration: a free-flyer root `[tx, ty, tz, qx, qy, qz, qw]` followed by the family's joints
(one ball joint per SMPL joint; revolute chains for MHR). The free flyer is world-aligned in every family:
`q[..., :3]` moves the root along the world axes and `q[..., 3:7]` is the root's world rotation from its neutral
orientation. Units are metres, seconds, kilograms and radians, and the world is y-up: every family's loader converts
its native frame, and the baked model's gravity points down `y`. `nq` and `nv` differ, so a pose changes through
`kinematics.integrate(identity.model, q, v)`, never by adding to `q`; read `identity.model.nq`, `nv` and
`q_neutral` instead of hard-coding them.

A batch of poses `q (B..., nq)` broadcasts against the identity's batch `I...`: one identity drives a whole clip, or
a batch of identities each takes a pose. A clip is `(B, T, nq)` with time on axis `-2`; it carries no rate, and the
BetterRobot viewer plays it at its `fps` or at the `times` the clip is added with. BetterHuman never resamples `q`.

## Classic parameters stay at the boundary

The upstream libraries' own inputs (`betas`, `global_orient`, `body_pose`, `transl`, hand PCA, MHR's
`model_parameters`, ...) enter and leave through one pair for every family:

```python
import torch
import better_human as bh
from better_human import families

human = bh.load("smplx", "tests/support/random_weights/smplx_random.npz")
betas, body_pose = torch.zeros(16), torch.full((63,), 0.1)  # this archive's 16 betas; 21 body joints
identity, q, streams = bh.from_classic(human, families.SMPLXClassic(betas=betas, body_pose=body_pose))
classic = bh.to_classic(identity, q, streams=streams)  # a families.SMPLXClassic
print(type(classic).__name__, tuple(q.shape), torch.allclose(classic.body_pose, body_pose))
```
```output
SMPLXClassic (223,) True
```

Each family has one keyword-only `<Family>Classic` dataclass that also carries the identity inputs, so
`from_classic` bakes the identity; an omitted field is zero. The spec's `classic` field, a `ClassicFormat`
(`src/better_human/core/classic.py`), maps the fields; `from_classic` and `to_classic` check every tensor in one
place. The round trip is exact to float precision both ways: the SMPL family keeps axis-angles beyond pi, and MHR's
Euler angles and SOMA's axis-angles come back on the principal branch. Inside BetterHuman there is one pose
convention, BetterRobot's `q`.

## Partial models are loader options

`smplx(hands=False, face=False)` drops the hand and face joints and the expression group, and
`families.drop_joints(spec, names)` removes any terminal joint subtree: its skin weights go to the surviving
ancestor, through the rig's `without_joints` (only `RegressedRig` can; another rig raises `ArgumentError`). A
partial model is never a subclass. A dropped part merges into its survivor's and takes the survivor's density, so
the total volume is kept. A classic field whose joints are dropped must stay `None`, and a hand PCA field cannot lose
single joints (load with `num_pca_comps=None`).

## Mass

The spec's parts segment the mesh, and `mass.PartTopology` closes each part into a solid once per `HumanModel`
([Mass](../mass.md)). `density` resolves to kg/m³ per part at `from_spec`: a table name (the
`"Dempster"` table by default) gives each part the
density of the category its family declares, a float is uniform, `None` builds no masses. On every bake
`mass.part_inertias` integrates each closed part with `better_robot.spatial.inertia_of_mesh`, differentiably, and the
inertias go into `identity.model`, so BetterRobot's dynamics use the subject's mass.

## Where next

- [Bodies](../bodies.md): identity groups, stream groups and how the mesh follows the joints.
- [Mass](../mass.md): how bodies split into parts.
- [Contracts](../contracts/index.md): the rules these layers obey and the records behind them.
