# BetterHuman addendum

BetterHuman-specific rules, in BetterRobot's format: ID, level (MUST / SHOULD), rule, rationale, example,
enforcement. The BetterRobot contracts in `BetterRobot/docs/contracts/` apply first (see [index](index.md)).

## H-LAYER-1 Imports follow decision record 0011's two edge tables (MUST)

Among BetterHuman's units: `io` and `mass` import no other unit; `core` imports `mass`; `families` imports `core`,
`io` and `mass`; `terms` imports `core`; `viz` imports `core` and `io` (the studio persists frame sets, a permanent
edge); `studio` imports the package root, `core`, `families`, `io`, `mass` and `viz`; the root imports every unit but
`studio`. Into BetterRobot: `io` only `exceptions` (its errors sit under `BetterRobotError`, BetterRobot 0031); `mass`
`lie` and `spatial`; `core` the root, `kinematics` and `lie`; `families` the root, `io`, `kinematics` and `lie`;
`terms` the root, `dynamics`, `kinematics` and `optim`; `viz` and `studio` `viewer`, `studio` also `lie`. Only
BetterRobot's public surface is imported (`C-DEP-1`, `tests/contract/test_br_imports.py`). `mass` and `studio` arrive
with record 0011's package layout; until then the studio is `viz/studio`, and each import that breaks a table today is
listed with its reason in `tests/contract/ratchets/edges.json`.

Rationale: the body model (`core`) and weight-file reading (`io`) stay usable without any family table or viewer.
Example: `src/better_human/families/smpl_family.py` imports `core` and `io`; `src/better_human/terms.py`
imports only `core`.
Enforcement: `tests/contract/test_layer_dependencies.py`.

## H-API-1 One body API across families (MUST)

A family loader (`smpl`, `smplh`, `smplx`, `mano`, `flame`, `mhr`) returns a `HumanModel`
(`src/better_human/core/human_model.py::HumanModel`). `HumanModel.bake(...)` gives an `Identity`
(`src/better_human/core/human_model.py::Identity`) that owns the baked BetterRobot model; `identity.forward(q, ...)` gives a
`HumanState` (`src/better_human/core/human_model.py::HumanState`). Families declare
*what* they are through a `FamilySpec` (`src/better_human/core/spec.py`); the *how* (LBS, blend shapes, correctives, regressors)
lives once in `core`. Partial models are loader parameters (`hands=False`), never subclasses.
Rationale: one forward pass to test and keep in parity; see principles 6 and 7 below.
Example: `src/better_human/families/smpl_family.py::load_family` serves all five SMPL-family loaders.
Enforcement: review; parity gates below.

## H-API-2 Arguments after the second are keyword-only (SHOULD)

Same rule as BetterRobot `C-API-2` (at most two positional parameters). Current violations are listed by qualified
name in the test's allowlist; the allowlist only shrinks.
Enforcement: `tests/contract/test_keyword_only.py`.

## H-ASSET-1 A shipped asset lives in an `assets/` folder beside the package that reads it; licensed weights are never shipped (MUST)

Small, redistributable data ships in an `assets/` folder beside the package that reads it and is read with
`importlib.resources`: segmentations and densities in `src/better_human/families/assets/`, frame sets in
`src/better_human/io/assets/`. Licensed weights are never downloaded, bundled or cached (principle 10).
Example: `src/better_human/io/assets/smplx_contacts.json`, read by `src/better_human/io/frame_sets.py::shipped_frame_set`.
Enforcement: review.

## H-ASSET-2 Weights resolve from an explicit path, else `io.models_dir()` (MUST)

An explicit `path` argument wins; otherwise the file is `io.weight_path(folder, filename)`, that is
`io.models_dir() / <folder> / <file_pattern>` with the patterns of `src/better_human/families/tables.py` (table under "Weight-file
layout"). `io.models_dir()` is the one reader of `$BETTERHUMAN_MODELS_DIR` and raises with the fix when it is unset
or names no directory; a missing or malformed file raises a clear error naming the path; nothing guesses a default.
Example: `src/better_human/io/load.py::models_dir`, `src/better_human/io/load.py::weight_path`.
Enforcement: `tests/io/test_load.py`; `tests/contract/test_models_dir.py` scans `tests/`, `examples/` and `tools/`
for a `models` path joined onto `__file__` or the working directory, a relative `models` path and
`better_human.__file__`. `benchmarks/` is outside the scan: `benchmarks/run.py` still builds `models/` paths from its
own location.

## H-STUDIO-1 Studio is a thin shell over the BetterRobot viewer (MUST)

The studio adds no viewer machinery: `app.py` drives `better_robot.viewer.Viewer`
(`src/better_human/viz/studio/app.py::HumanStudio`) and `layers.py` subclasses `better_robot.viewer.Layer`
(`src/better_human/viz/studio/layers.py::SegmentationLayer`). Domain logic lives in
`state.py`, `persistence.py`, `posing.py` and `gizmo.py` and is testable without a server; UI wiring lives in
`app.py`, `layers.py`, `segment_editor.py` and `frames_editor.py`. A generic viewer feature goes to BetterRobot.
Rationale: principle 5 (not human-specific means it belongs in BetterRobot).
Enforcement: review; `tests/viz/` exercises the domain modules headless.

## H-STUDIO-2 Studio data layout is `<data>/{segmentations,frames,poses}/<key>/` (MUST)

Studio state is JSON under one data root (default `studio_data/`): vertex segmentations, named frame sets and
poses each in their own directory, keyed by the mesh-topology key.
Example: `src/better_human/viz/studio/persistence.py::StudioPaths` (`segmentation_dir`, `frame_dir`, `pose_dir`).
Enforcement: `tests/viz/`.

## Numeric conventions, layout, gates and baselines

Parity gates are rule `H-PARITY-1` (MUST): never weaken them.

### Units and frames

- Metres, seconds, kilograms, radians. Density in kg/m³.
- y-up world. Families whose native frame differs are converted by their loader. The baked BR
  model carries gravity against `FamilySpec.up_axis` (`model.gravity[:3] == (0, -9.81, 0)` for every
  shipped family), so `root_wrench` and BR dynamics need no root rotation; `terms.ground` defaults to
  `up=(0, 1, 0)` for the same reason.
- Joint order: `HumanState.joints`, `joint_poses`, `rest_joints` use the family's native skin-joint
  order. `q` and BR quantities use BR model order; `rig.native_joint_ids(spec, model)` maps between them.
- `pelvis` is the root offset from the native origin. `transl + pelvis` is the root position.

### State

- `q` = one free-flyer root `(x, y, z, qx, qy, qz, qw)` followed by one spherical
  joint (4 values) per non-root joint. `v` = 6 + 3 per spherical joint.
- The free flyer is world-aligned in every family: `q[:3]` moves the root along the world axes (for the
  SMPL family and SOMA it is the root joint's world position) and `q[3:7]` is the root's world rotation
  relative to its neutral orientation. `q[..., 1] += 1` lifts any body 1 m.
- SMPL: `nq = 99`, `nv = 75`. Read `identity.nq`, `identity.nv`, `identity.q_neutral`; do not hard-code.
- Move on the manifold with BR: `kinematics.integrate(identity.model, q, dq)`.

### Batching

| quantity | shape |
|---|---|
| identity groups to `bake` | `(I..., n)`; omitted groups are zero |
| pose `q` | `(B..., nq)`, usually `(B, nq)` or `(B, T, nq)` |
| stream groups to `forward` | `(B..., n)`, same leading dims as `q` |
| outputs | `(broadcast(I..., B...), ...)`, e.g. `vertices (..., V, 3)` |

`B...` broadcasts with `I...`: one identity can drive a whole clip, or a batch of identities can
take one pose each. Unknown group names raise `ValueError`.

A clip is `(B, T, nq)`: axis `-2` is time and carries no rate of its own. Time follows BR's viewer
model (`BetterRobot/docs/concepts/viewer.md`): the viewer timeline has integer steps played at `fps`;
a layer with `T` samples shows sample `min(step, T - 1)` by default, or, when added with
`Viewer.add(layer, group, times=...)`, `times` is a `(T,)` non-decreasing array of seconds and the
layer shows the latest sample with `times[i] <= step / fps`. A clip recorded at a different rate
therefore passes its `times` to the viewer; BH never resamples `q`.

### Differentiability

- `bake`, `forward`, `vertices`, `points`, `landmarks`, mass (`inertia_of_mesh`) and BR
  kinematics/dynamics are differentiable in the identity values, `q` and streams.
- `HumanState` memoises `joint_transforms`, `skin_affines`, `corrective_feature` and `vertices`; create a new state
  for new inputs, never mutate one.

### Classic conversion

- `<family>_from_classic(identity, classic) -> (q, stream)` and
  `<family>_to_classic(identity, q, stream=None) -> <Family>Classic` in `src/better_human/families/classic.py`.
  MHR: `mhr_from_classic(body, classic) -> (identity, q, stream)` and
  `mhr_to_classic(identity, q, stream=None)` in `src/better_human/families/mhr.py`.
- Hand PCA (SMPL-H, SMPL-X, MANO): `num_pca_comps` must be in `[1, 45]` or `None` (full axis-angle).
  `to_classic` raises `ValueError` when the hand pose leaves the requested PCA subspace.
- Dropped joints: a hand pose given to a body loaded with `hands=False` raises `ValueError`;
  `to_classic` on such a body returns `None` for that hand, so the round trip is closed.
- Round trips are exact up to rotation equivalence: an axis-angle with norm above π comes back
  as the same rotation with norm below π (vertices match to float precision).

### Rig protocol changes

- `Rig.corrective_features(spec, model, q, *, identity=None)` takes the identity group values
  `(I..., n)` that `Body.bake` used, keyword-only, so a shape-dependent corrective (STAR, SUPR) can
  read them; `HumanState` passes `identity=state.identity.params`. **Breaking for third-party
  `Rig` implementations**: add the keyword to the method (ignore it if the corrective is pose-only).
- `Rig.joint_transforms(spec, model, *, kinematics)` is new: `(..., J, 3, 4)` world `[R | p]` of the
  native joints from the BR `KinematicState`. `HumanState.joint_transforms` memoises it, and `joint_poses`
  and skinning derive from it, so a rig can post-process rotations BR cannot model (SOMA twist helpers
  and world-aligned Hips). Positions must stay BR's joint translations: `HumanState.joints` reads those.
  `RigBake.inverse_bind_poses` (copied to `Identity.inverse_bind_poses`) replaces
  `meta["inverse_bind_poses"]` as the skinning bind. **Breaking for third-party `Rig` implementations**:
  add `joint_transforms` (`kinematics.joint_transforms.index_select(-3, self.native_joint_ids(spec, model))`
  keeps the old behaviour) and return authored inverse bind poses in the `RigBake`.
- `RigBake.v_shaped (..., V, 3)` is new and optional: the skinning rest mesh when the rig re-poses the
  shaped mesh. `Body.bake` uses it in place of the blend-shape mesh, so pose correctives are added to
  the re-posed mesh (SOMA re-poses its fitted skeleton to the template bind). `None` keeps the old behaviour.

### Scope

- Families are in-tree only during alpha: a new family lands as a module under
  `better_human/families/`; there is no public `register_family` hook.
- Retargeting (motion transfer between skeletons) is not BH's job; it belongs in a separate package.
- Fitting (bodies to scans, keypoints or video) is not BH's job either; BH supplies the
  differentiable bodies and terms such a fitter uses.

### Weight-file layout

Files live under `io.models_dir() / <folder> / <file_pattern>` (`$BETTERHUMAN_MODELS_DIR`); an explicit `path`
argument wins. Patterns are in `src/better_human/families/tables.py`:

| family | folder | file pattern |
|---|---|---|
| SMPL | `smpl` | `SMPL_{gender}.npz` |
| SMPL-H | `smplh` | `smplh/{gender_lower}/model.npz` |
| SMPL-X | `smplx` | `SMPLX_{gender}.npz` |
| MANO | `mano` | `MANO_{side}.npz` |
| FLAME | `flame` | `{gender}_model.npz` |
| MHR | `MHR/converted` | `mhr_lod{0..6}.npz` (default LOD 1) |
| SOMA | `SOMA` | `soma_mid.npz` (mid LOD only; `tools/convert/soma`) |

### Parity gates (do not weaken)

| gate | test | tolerance | status |
|---|---|---|---|
| SMPL vs upstream goldens | `tests/parity/smplx/test_smpl_parity.py` | see test | pass |
| SMPL-X, 29 goldens, float32 and float64, expression streamed | `tests/parity/smplx/test_smplx_parity.py` | 1e-6 m vertices and joints | worst 5.1e-7 m |
| SMPL-X with `stream_joints=False` | same file | error in (1e-5, 2e-3) m | about 1.2 mm |
| MHR, LOD 0-6, float32 / float64 | `tests/parity/mhr/test_mhr_parity.py` | f32 `(5e-6, 1e-6, 1e-6, 1e-7)`, f64 `(2e-6, 5e-7, 5e-7, 1e-7)` | pass |
| MHR vs TorchScript `mhr_model.pt` | `tests/parity/mhr/test_mhr_torchscript.py` | vertices 2e-6 m, positions 1e-6 m, quaternions 1e-6 | pass |
| SOMA mid vs SOMA-X v0.3.4, 14 goldens one by one and as one identity batch, float32 / float64 | `tests/parity/soma/test_soma_parity.py` | `(6.5e-6, 9e-6, 3.5e-6)` (vertices m; positions m and rotation entries of all 110 skin joints); `reflect_00` vertices 1.1e-5 | worst f32 `(1.46e-6, 1.16e-6, 1.73e-6)`, f32 `reflect_00` `(5.36e-6, 3.58e-6, 1.13e-6)`, f32 batch `(3.13e-6, 4.35e-6, 1.73e-6)`, f64 `(2.89e-6, 4.23e-6, 1.60e-6)` |
| SOMA bake vs the same goldens' `rest_shape`, `bind_transforms` | `tests/parity/soma/test_soma_parity.py::test_bake_matches_upstream_rest_shape_and_bind` | `(6e-6, 1e-5, 1.1e-6)` (rest mesh m, bind positions m, bind rotation entries; all 110 joints); `reflect_00` rest mesh 1.2e-5 | worst f32 `(9.5e-7, 1.31e-6, 5.4e-7)`, f32 `reflect_00` `(5.90e-6, 3.87e-6, 5.4e-7)`, f32 batch `(2.86e-6, 4.89e-6, 5.4e-7)`, f64 `(2.92e-6, 4.66e-6, 5.0e-7)` |

Parity tests are marked `licensed`: the full tier deselects them, and a run that selects them fails when the
models directory or an asset is missing (`tests/conftest.py`).
SOMA gates are twice the worst error measured over the 14 goldens, one by one and with the 13 correctives-on
goldens as one identity batch, in float32 and float64, rounded up; one gate serves both dtypes (the bake test runs
the goldens one by one; its batch column is measured). The oracle is float32 and its joint regression alone rounds
by up to 5.2e-6 m, so float64 and batched float32 sit further from it than single float32, which sums in the
oracle's order; no SOMA position gate is below that rounding. `reflect_00`, a 5-sigma identity whose skeleton fit
takes the Kabsch reflection branch at 13 covariances (11 finger, hand and toe joints, then both hands again), run
alone, sets its own vertex and rest mesh gates (`PARITY_ATOLS_BY_CASE`, `BAKE_ATOLS_BY_CASE`) and not the general
ones: single float32 errs 5.4e-6 and 5.9e-6 m there (at the left pinky) because float32 joint regression rounds
further on so large an identity (3.0e-6 and 3.2e-6 m with only the regression in float64; float64 SVDs do not help),
and upstream's own float32 run is 7.5e-6 m (vertices) from its float64 run on that case.

### Measured baselines

`benchmarks/run.py`, float32, median ms. Ratio = ours / upstream (below 1 = ours faster).
Upstream = vendored `smplx` forward for SMPL and SMPL-X, TorchScript `mhr_model.pt` for MHR.
CPU: AMD EPYC 9654, 8 threads. CUDA: NVIDIA RTX PRO 6000 Blackwell.

| family | op | CPU B=1 | CPU B=1024 | CUDA B=1 | CUDA B=1024 |
|---|---|---:|---:|---:|---:|
| SMPL | vertices (ratio) | 1.21 (0.69) | 296 (0.95) | 1.19 (0.80) | 8.05 (0.99) |
| SMPL-X | vertices (ratio) | 3.02 (0.42) | 464 (0.74) | 1.25 (0.47) | 12.3 (0.82) |
| MHR | vertices (ratio) | 22.4 (0.59) | 1310 (0.25) | 2.50 (1.15) | 26.8 (1.02) |
| SMPL | bake with mass (B=1) | 11.2 | | 20.8 | |
| SMPL-X | bake with mass (B=1) | 23.5 | | 45.4 | |
| MHR | bake with mass (B=1) | 20.3 | | 21.8 | |
| SMPL | fk / rnea | 0.26 / 0.31 | 6.87 / 15.8 | 0.60 / 0.73 | 0.74 / 0.81 |
| SMPL-X | fk / rnea | 0.35 / 0.37 | 12.0 / 25.6 | 0.79 / 0.72 | 0.83 / 0.82 |
| MHR | fk / rnea | 0.94 / 0.55 | 41.2 / 45.2 | 1.61 / 0.77 | 2.08 / 1.55 |

Caveats:

- Inputs differ slightly. MHR "default" vertices include the expression stream; SMPL-X "default"
  passes no stream.
- SMPL bake without mass is 0.41 ms on CPU: mass integration dominates `bake`.
- Batch 8192 vertex cells (ours and upstream) were skipped where the estimate exceeded 8 GB
  (for example SMPL 10.8 GB, SMPL-X 16.5 GB, MHR TorchScript 29.0 GB). fk and rnea ran at 8192.
- Full tables: `benchmarks/results/p4_bodies_{cpu,cuda}.json`.

## Design principles

They mirror BetterRobot's rules; when the two disagree, BetterRobot's numerical contracts win.

### Over-engineering is the enemy

1. **Minimum code that solves the problem.** No speculative flexibility, no
   config knobs nobody asked for, no error handling for impossible states.
2. **An abstraction needs ≥ 2 in-tree callers.** The parametric-body base
   class earns its existence by SMPL and SMPL-X both using it from day one;
   nothing else gets a base class until a second subclass exists.
3. **No forwarding layers, no raise-only placeholders, no "for later" hooks.**
   Unimplemented ideas live in `plan/` or the roadmap, not in the API.
4. **Line count is a design signal.** If a diff grows a module past what the
   change plainly needs, structure was added that the problem doesn't have.

### Human-specific rules

5. **If it is not human-specific, it does not belong here.** Kinematics,
   dynamics, Lie math, solvers, viewers come from BetterRobot. The moment a
   would-be BetterHuman utility is model-agnostic robotics, it is a
   BetterRobot contribution instead.
6. **One model, one file; shared math in shared modules.** A family loader
   returns a `FamilySpec` that declares *what* it is (topology, parameter spaces, weight-file
   schema, classic-convention quirks); the *how* (LBS, blend shapes,
   correctives, regressors) lives once in common modules. Copy-pasted
   forward passes across model classes — the smplx-package disease — are
   forbidden.
7. **Partial models are parameters, never subclasses.** `smplx(...,
   hands=False)` style loader options, implemented by one generic
   reduction mechanism in the base machinery. No `SMPLXmini`-style class
   forks, ever.
8. **Classic conventions are quarantined at the boundary.** Inside
   BetterHuman there is exactly one state convention: BetterRobot-style `q`.
   Each model's native format (axis-angle vectors, offset root translations,
   PCA hand spaces) enters or leaves only through its `from_classic` /
   `to_classic` pair, which must be exact inverses and covered by round-trip
   tests. A private pose-feature policy may transiently reconstruct the
   rotation representation required by authored correctives; it never becomes
   parallel public state.
9. **Parity with the original implementation is the safety net.** Every
   model's forward pass is verified numerically against the original
   author-released code (run in its own pinned venv) before the model is
   considered supported. Never weaken a parity test to land a change.
10. **Weights are the user's.** No downloading, no bundling, no silent
    caching of licensed model files. Explicit path argument, documented
    directory convention, clear error messages when files are missing or
    malformed.

### Numerical and performance honesty

11. **Never silently fall back** to a slow path; warn once with the reason.
12. **No host synchronization in hot loops**; preserve dtype/device; never
    silently cast, move, or squeeze. Batched by default, trailing event
    axes, arbitrary leading batch axes — identical to BetterRobot contracts.
13. **GPU is a first-class target.** Device-conditional behavior is validated
    on CPU and on pinned CUDA hardware: such tests are marked `cuda`, and a run
    that selects them without a device fails; nothing is reported as a pass.

### Process

14. **Alpha: breaking changes are fine, compatibility shims are not.** No
    aliases, no deprecation paths.
15. **Docs move with the code in the same change.**
16. **Deletion needs evidence; results reports are honest** — exact counts,
    deviations named as findings.
17. **One behavior, one test; test behavior, not trivia.** float32
    gradchecks, following the workspace testing rules.
