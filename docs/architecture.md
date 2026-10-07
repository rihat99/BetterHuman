# Architecture

BetterHuman turns parametric human weight files (SMPL, SMPL-H, SMPL-X, MANO, FLAME, MHR) into
BetterRobot (BR) models. A family is data. One `Rig` object holds the only family-specific
behaviour. Shape is baked once per subject; pose is a batched, differentiable BR query.

## Layering

```text
better_robot (public API only: Model, ModelBuilder, io.build_model, kinematics, dynamics,
              lie.{so3,se3}, spatial.Inertia, viewer)
      ^
better_human.core       spec, rig, body, pipeline, mass, contact_frames, reduction, classic
      ^
better_human.families   smpl_family, tables, classic, mhr, mhr_rig, mhr_tree, mhr_archive, mhr_config
      ^
better_human.terms      better_human.viz      (both use core + families, never each other's internals)
```

- Each layer imports only from layers below it.
- BetterHuman imports BR through its public surface only. `tests/contract/test_br_imports.py`
  fails on any private BR import (`better_robot.io.builders.*`, `better_robot.io.ir`, ...).

## The four objects

```text
weight file --loader--> FamilySpec --Body(spec)--> Body --bake(**identity)--> Identity --forward(q, stream)--> HumanState
              (families)   (data + rig)     (BR model once)        (shape once)            (per frame, memoised)
```

### FamilySpec (`core/spec.py`)

Plain data: template vertices, faces, top-K skin weights, identity and stream blend groups,
a `Corrective`, `Parts` (segmentation), landmarks and `meta`. Its one behavioural member is
`rig`, which implements the `Rig` protocol (`core/rig.py`):

| Rig method | Job |
|---|---|
| `bake(spec, model, v_shaped, values) -> RigBake` | joint placements, rest joints, rest poses, pelvis offset |
| `stream_joint_basis(spec, name)` | `(J, 3, n)` joint motion of a stream group, or `None` |
| `native_joint_ids(spec, model)` | `(J,)` BR joint ids in native skin-joint order |
| `corrective_features(spec, model, q)` | `(..., P)` features the pose corrective reads |
| `to(dtype, device)` | move the rig |

Two rigs exist:

- `RegressedRig` (`core/rig.py`, SMPL family). Joints come from a joint regressor `(J, V)`
  applied to the shaped vertices. The stream joint basis is `J_regressor @ stream_blendshapes`.
- `AuthoredRig` (`families/mhr_rig.py`, MHR). Joints come from the authored MHR skeleton with
  per-joint scales. It also builds the BR model itself (`build_model`). Its
  `stream_joint_basis` returns `None`: MHR expression moves vertices only.

### Body (`core/body.py`)

```python
Body(spec, *, dtype=None, device=None, mass=True, density=1000.0, contact_frames=None, stream_joints=True)
```

- Builds one BR `Model` (topology) per family at construction. `body.model`, `body.faces`,
  `body.parents`, `body.spec`, `body.dtype`, `body.device`.
- `mass=True` builds the part topology once; inertias are integrated on every bake.
- `contact_frames` is a frame-set JSON path or a mapping `name -> (vertex_id, parent joint | None)`.
  Each entry becomes a BR operational frame, re-placed on every bake.
- `bh.load(family, path=None, **options)` calls the family loader and `Body` in one step.
  `mass`, `density`, `contact_frames` and `stream_joints` go to `Body`; the rest go to the loader.

### Identity (bake once)

`body.bake(**identity)` takes identity groups `(..., n)` by name (`betas` for the SMPL family,
`identity` and `proportions` for MHR). Omitted groups are zero; leading dims broadcast.
It returns an `Identity`:

| field | shape | meaning |
|---|---|---|
| `params` | `{name: (I..., n)}` | identity group values |
| `v_shaped` | `(I..., V, 3)` | shaped vertices, root at the origin |
| `rest_joints` | `(I..., J, 3)` | native-order rest joints |
| `pelvis` | `(I..., 3)` | root offset from the native origin; `transl + pelvis` is the root position |
| `model` | BR `Model` | joint placements, body inertias, contact frames baked in |
| `joint_scales` | `(I..., J)` or `None` | authored rigs only |
| `stream_joint_basis` | `{name: (J, 3, n)}` | empty when `stream_joints=False` |

`identity.nq`, `identity.nv` and `identity.q_neutral` come from `identity.model`.

### HumanState (per frame, memoised)

`identity.forward(q, stream=None)` with `q (B..., nq)`. `B...` broadcasts with `I...`.

| member | shape | note |
|---|---|---|
| `kinematics` | BR `KinematicState` | forward kinematics on `state.model` |
| `joints` | `(..., J, 3)` | native joint order |
| `joint_poses` | `(..., J, 7)` | native joint order |
| `vertices()` | `(..., V, 3)` | computed on first call, then memoised |
| `points(vertex_ids)` | `(..., K, 3)` | skin a vertex subset only |
| `landmarks(name="keypoints")` | `(..., L, 3)` | named landmark sets of the spec |
| `model` | BR `Model` | per-frame placements when a stream moves joints |
| `rest_joints`, `pelvis`, `rest_offset` | | after the stream joint shift |

## Identity groups and stream groups

- Identity groups (`betas`, MHR `identity`/`proportions`) are baked once. They change joint
  placements, inertias and contact frames.
- Stream groups (`expression`) are passed per frame to `forward`. They add vertex offsets
  `rest_offset (..., V, 3)`.

Expression-stream joint shift (regressed rigs, `stream_joints=True`):

```text
motion      = einsum("...n,jcn->...jc", coefficients, stream_joint_basis[name])   # (..., J, 3)
rest_joints = identity.rest_joints + sum over streams of motion
state.model = identity model with joint placements re-derived from the shifted rest joints
```

This matches upstream SMPL-X exactly (expression moves the jaw and eye joints). With
`stream_joints=False` the stream moves vertices only; the error on the SMPL-X goldens is about
1.2 mm (the test asserts it stays between 1e-5 and 2e-3 m).

## Classic converters (at the boundary)

Native parameter sets (`betas`, `global_orient`, `body_pose`, `transl`, hand PCA, ...) are
converted at the boundary, never inside the core:

| family | module | functions |
|---|---|---|
| SMPL, SMPL-H, SMPL-X, MANO, FLAME | `families/classic.py` | `<family>_from_classic(identity, classic) -> (q, stream)`, `<family>_to_classic(identity, q, stream=None)` |
| MHR | `families/mhr.py` | `mhr_from_classic(body, classic) -> (identity, q, stream)`, `mhr_to_classic(identity, q, stream=None)` |

Each family has a `<Family>Classic` dataclass (`SMPLClassic`, `SMPLXClassic`, `MHRClassic`, ...).

## Partial models

`core.drop_joints(spec, names) -> FamilySpec` removes joints from a regressed-rig spec. It
needs a linear corrective (`ValueError` otherwise); `core/reduction.py` slices the corrective.
Loaders expose the common cases: `smplx(hands=False, face=False)` drops the hand and face
joints and the expression group. Converting a pose that uses a dropped joint raises.

## Mass pipeline (`core/mass.py`)

1. The spec's `Parts` segment the mesh; `build_authored_topology` closes each part with
   authored joint caps into one `PartMeshTopology` (built once per `Body`).
2. On every bake the shaped part meshes go through
   `better_robot.spatial.Inertia.from_mesh` (differentiable) with `density` in kg/m³
   (scalar or one value per part).
3. The inertias are written into `identity.model`, so BR dynamics (`inverse_dynamics`,
   `mass_matrix`) use the subject's mass.

## Package layout

```text
src/better_human/
  __init__.py        load, Body, FamilySpec, Identity, HumanState
  core/              spec, rig, body, pipeline (blend shapes, top-K skinning), mass,
                     contact_frames, reduction, classic
  families/          loaders smpl/smplh/smplx/mano/flame (smpl_family), mhr; tables (file layout,
                     joint names); classic converters; MHR rig, tree, archive, config
  terms.py           optimisation terms (markers, landmarks, reprojection, ground, priors,
                     shape fit, root_wrench) for BR solvers
  viz/human_layers.py   human_layers, HumanView, mesh/skinned/skeleton/frames/forces layers
  io/                weight-file readers, convert_chumpy
examples/            01_smpl_shape_pose, 02_smplx_partial, 03_ik_reach, 04_mhr_shape_pose
benchmarks/          run.py, report.py, results/*.json
tests/               core, families, dynamics, io, parity (goldens), contract, viz
```

## Verification

- Parity gates against upstream goldens (`tests/parity/`): SMPL, SMPL-X (29 goldens) and MHR
  (all 7 LODs, against goldens and the TorchScript model). Tolerances are in `docs/contracts.md`.
- Contract test `tests/contract/test_br_imports.py`: public BR API only.
- Examples: each runs headless with `--check`.
