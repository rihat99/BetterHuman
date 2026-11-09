# Architecture

BetterHuman turns parametric human weight files (SMPL, SMPL-H, SMPL-X, MANO, FLAME, MHR, SOMA) into
BetterRobot (BR) models. A family is data. One `Rig` object holds the only family-specific
behaviour. Shape is baked once per subject; pose is a batched, differentiable BR query.

## Layers sit on top of BetterRobot

BetterHuman builds a BetterRobot `Model` per subject and reuses BetterRobot for everything else.

| Shared with BetterRobot | Human-specific (BetterHuman) |
|---|---|
| `Model`, `ModelBuilder`, `io.build_model` | weight-file readers, family tables, the MHR and SOMA rigs |
| `KinematicState`, Jacobians, `dynamics` | blend shapes, pose correctives, top-K skinning |
| `lie.{so3,se3}`, `spatial.inertia_of_mesh` | per-part mass from mesh volume, contact frames |
| `optim` solvers, `Problem`, `TermFn` | `better_human.terms` (markers, reprojection, ground, ...) |
| `viewer` layers and `Viewer` | `better_human.viz` (`human_layers`, `HumanView`, HumanStudio) |
| `exceptions` (`ShapeError`, `UnknownNameError`, ...) | `FormatError`, `AssetNotFoundError`, both under `BetterRobotError` |

`tests/contract/test_br_imports.py` fails on any private BetterRobot import.

## The package DAG is a test

`tests/contract/test_layer_dependencies.py` (rule `H-LAYER-1`, decision record 0011) fails on any import that
points the wrong way between the units of `better_human`, and on any import of a BetterRobot unit that the second
table of the same record does not list for the importer:

```text
better_human (root) ->  core, families, io, mass, terms, viz
studio   ->  better_human (root), core, families, io, mass, viz
viz      ->  core, io
terms    ->  core
families ->  core, io, mass
core     ->  mass
io, mass      (nothing)
```

`studio` arrives with the record's package layout; until then the studio is `viz/studio`, and its imports of the
root, `mass` and `better_robot.lie` are listed with their reasons in `tests/contract/ratchets/edges.json`.

## The four objects

```text
weight file --families.<f>(path, dtype=, device=)--> FamilySpec --HumanModel.from_spec--> HumanModel
            --bake(**identity)--> Identity --forward(q, streams)--> HumanState
```

`bh.load(family, path=None, *, dtype, device, density, contact_frames, stream_joints)` runs the family loader with
its defaults and `HumanModel.from_spec` in one step. Family options (gender, number of betas, LOD, ...) are the
loader's: `HumanModel.from_spec(families.smplx(path, num_betas=16), density=None)`. Dtype and device are chosen once,
at the loader; no BetterHuman object has `to`, and a tensor in another dtype or on another device than the model's
raises (`DtypeMismatchError`, `DeviceMismatchError`).

### FamilySpec (`src/better_human/core/spec.py`)

Frozen, typed data: `template_vertices (V, 3)`, `faces (F, 3)`, `joint_names`, `parents`, `file_joint_ids`, top-K
`skin_weights` and `skin_indices (V, K)`, `identity_bases` and `stream_bases` (group name -> `(V, 3, n)` blend
shapes), a `Corrective` (`LinearCorrective` or `MLPCorrective`, or `None`), `Parts` (segmentation and density
categories), `Landmarks`, `up_axis` and `provenance`. Its one behavioural member is `rig`, an instance of the abstract
`Rig` (`src/better_human/core/rig.py`):

| Rig method | Job |
|---|---|
| `build_model(spec, *, frames)` (abstract) | the BR `Model` in the spec's dtype and device, with operational frames attached |
| `bake(spec, model, *, shaped_vertices, identity) -> RigBake` (abstract) | joint placements, the skinning rest mesh, rest joint translations, inverse-bind affines `(I..., J, 3, 4)`, root offset |
| `joint_ids(spec, model)` (abstract) | `(J,)` BR joint id of each native joint |
| `corrective_features(spec, model, *, q, identity)` (abstract) | `(..., P)` features the pose corrective reads |
| `joint_transforms(spec, model, *, kinematics)` | `(..., J, 3, 4)` world `[R \| p]` of the native joints; by default BR's. Positions stay BR's joint translations; a rig only post-processes rotations BR cannot model (SOMA twist helpers and world-aligned Hips) |
| `stream_joint_basis(spec, name)` | `(J, 3, n)` joint motion of a stream group, or `None` (the default) |
| `placements_from_rest(spec, model, *, rest_joint_translations)` | joint placements for stream-shifted rest joints, or `None` (the default) |

Three rigs exist. `RegressedRig` is `@final`; `MhrRig` and `SomaRig` are not yet (they sit in
`tests/contract/ratchets/abstract_or_final.json` until the families work closes them):

- `RegressedRig` (`src/better_human/families/regressed_rig.py`, SMPL family). Joints come from a joint regressor
  `(J, V)` applied to the shaped vertices; every joint binds at its rest translation (`[I | -rest]`). The stream joint
  basis is `J_regressor @ stream_blendshapes`, so expression moves its joints.
- `MhrRig` (`src/better_human/families/mhr_rig.py`, MHR). Joints come from the authored MHR skeleton: each native
  joint is a chain of revolutes, one per rotation entry of the parameter transform, with mimics for parameters that
  drive several entries. Proportions set the joint offsets and accumulated scales; the scales fold into the
  inverse-bind affines (`[s R | t - (s - 1) R c]`, record 0003). MHR expression moves vertices only.
- `SomaRig` (`src/better_human/families/soma_rig.py`, SOMA). Bake fits the template skeleton to the shaped mesh and
  re-poses mesh and skeleton to the template bind, as upstream. Its `joint_transforms` turns the 32 twist helpers by
  the swing-twist of their limb segments and turns the world-aligned Hips free flyer into the upstream Hips frame.

### HumanModel (`src/better_human/core/human_model.py`)

```python { .no-run }
HumanModel.from_spec(spec, *, density="Dempster", contact_frames=None, stream_joints=True)
```

- Computes every input from the spec, then constructs once; the record is frozen and compares by identity.
- `human.model` is the zero-identity BR `Model` (contact frames attached); `human.spec` holds the data
  (`human.spec.faces`, `human.spec.parents`, ...); `skin_matrix (V, J)`, `joint_ids (J,)`, `part_body_ids (P,)`.
- `density` is a table name (`"Dempster"`, the default), a uniform float or a `()` / `(P,)` tensor in kg/m³, or `None`
  (the baked models carry no inertias). A table needs a density category for every part; a family without one
  raises `ArgumentError` naming the parts (SOMA, MANO and FLAME until their tables ship: pass `None` or a float).
- `contact_frames` is a mapping `name -> (vertex_id, parent joint name | None)`, from a frame-set file through
  `io.read_frame_set(path)` or shipped as `io.shipped_frame_set("smplx_contacts")`. Each entry becomes a BR
  operational frame, re-placed on every bake.

### Identity (bake once)

`human.bake(**identity)` takes identity groups `(..., n)` by name (`betas` for the SMPL family, `identity` and
`proportions` for MHR). Omitted groups are zero; leading dims broadcast; a misspelled group raises
`UnknownNameError`. It returns a frozen `Identity`:

| field | shape | meaning |
|---|---|---|
| `values` | `{name: (I..., n)}` | identity group values |
| `rest_vertices` | `(I..., V, 3)` | the skinning rest mesh, root at the origin |
| `rest_joint_translations` | `(I..., J, 3)` | native joints in the skinning rest frame |
| `inverse_bind_affines` | `(I..., J, 3, 4)` | rest frame into each joint's frame |
| `root_offset` | `(I..., 3)` | root from the native origin; `transl + root_offset` places the root |
| `model` | BR `Model` | joint placements, body inertias, contact frames baked in |

`identity.model.nq`, `identity.model.nv` and `identity.model.q_neutral` size and seed `q`.

### HumanState (per frame, memoised)

`identity.forward(q, streams=None)` with `q (B..., nq)`. `B...` broadcasts with `I...`; a misspelled stream group
raises `UnknownNameError`.

| member | shape | note |
|---|---|---|
| `kinematics` | BR `KinematicState` | forward kinematics on `state.model` |
| `joint_translations` | `(..., J, 3)` | native joint order |
| `joint_transforms` | `(..., J, 3, 4)` | native joint order; the rig's `joint_transforms`, which skinning reads |
| `joint_poses` | `(..., J, 7)` | native joint order; `joint_transforms` as poses |
| `skin_affines` | `(..., J, 3, 4)` | `joint_transforms` composed with the inverse-bind affines |
| `vertices` | `(..., V, 3)` | computed on first access, then memoised |
| `points(vertex_ids)` | `(..., M, 3)` | skin a vertex subset only |
| `landmarks()` | `(..., L, 3)` | the spec's landmarks |
| `frame_poses` | `(..., nframes, 7)` | world poses of the model frames, contact frames included |
| `model` | BR `Model` | per-frame placements when a stream moves joints |
| `rest_joint_translations`, `inverse_bind_affines`, `root_offset`, `stream_offsets` | | after the stream joint shift |

## Identity groups and stream groups

- Identity groups (`betas`, MHR `identity`/`proportions`) are baked once. They change joint
  placements, inertias and contact frames.
- Stream groups (`expression`) are passed per frame to `forward`. They add vertex offsets
  `stream_offsets (..., V, 3)`.

Expression-stream joint shift (regressed rigs, `stream_joints=True`):

```text
motion                  = einsum("...n,jcn->...jc", coefficients, stream_joint_bases[name])   # (..., J, 3)
rest_joint_translations = identity.rest_joint_translations + sum over streams of motion
state.model             = identity model with joint placements re-derived from the shifted rest joints
```

This matches upstream SMPL-X exactly (expression moves the jaw and eye joints). With
`stream_joints=False` the stream moves vertices only; the error on the SMPL-X goldens is about
1.2 mm (the test asserts it stays between 1e-5 and 2e-3 m).

## Skinning (`src/better_human/core/skinning.py`)

One path for every rig: `skin_affines(joint_transforms, inverse_bind_affines)` gives each joint's rest-to-world
affine, and `linear_blend_skinning(rest_vertices, affines, *, weights)` blends them. `blend_shapes` adds identity and
stream offsets; `top_k_skin` keeps the K strongest weights per vertex at load.

## Classic converters (at the boundary)

Native parameter sets (`betas`, `global_orient`, `body_pose`, `transl`, hand PCA, ...) are converted at the boundary,
never inside the core: `families.<family>_from_classic` and `families.<family>_to_classic` for SMPL, SMPL-H, SMPL-X,
MANO, FLAME (`src/better_human/families/classic.py`), MHR (`src/better_human/families/mhr.py`) and SOMA
(`src/better_human/families/soma.py`). Each family has a `<Family>Classic` dataclass (`SMPLClassic`, `SMPLXClassic`,
`MHRClassic`, ...).

## Partial models

`families.drop_joints(spec, names) -> FamilySpec` removes terminal joint subtrees from a regressed-rig spec (their
skin weights go to the surviving ancestor). It needs a linear corrective; `src/better_human/core/reduction.py` slices
the corrective. Loaders expose the common cases: `smplx(hands=False, face=False)` drops the hand and face joints and
the expression group. Converting a pose that uses a dropped joint raises.

## Mass pipeline (`src/better_human/mass/`)

1. The spec's `Parts` segment the mesh; `PartTopology.from_closed_faces` (authored caps) or
   `PartTopology.from_part_ids` (centroid caps) closes each part, once per `HumanModel`.
2. `density` resolves to `()` or `(P,)` kg/m³ at `from_spec`; a table name goes through `mass.part_densities`, which
   gives each part the density of the category its family declares (`Parts.density_categories`). `from_spec` then
   integrates the zero identity's rest mesh once with `check_mesh=True`: a density that is not finite and positive,
   or a degenerate part, raises `ArgumentError` there, so bakes read no tensor values.
3. On every bake the rest mesh is posed at `q_neutral` and `mass.part_inertias` integrates each closed part with
   `better_robot.spatial.inertia_of_mesh` (differentiable) in its part frame.
4. The inertias are written into `identity.model`, so BR dynamics (`inverse_dynamics`, `mass_matrix`) use the
   subject's mass.

The mass package reads no file: the segmentation and the density tables are versioned archives in
`src/better_human/io/assets/`, read by `src/better_human/io/mass_data.py` and handed to `Parts` by the family loaders.

## Where next
- [Segmentation](segmentation.md): how bodies split into parts.
- [Studio](../studio.md): look at a body and edit its parts and frames.
- [Contracts](../contracts/index.md): the rules these layers obey, parity gates and tolerances.
- [Vision and roadmap](../reference/roadmap.md): what BetterHuman is for, what landed, and what comes next.
