# BetterHuman addendum

BetterHuman-specific rules, in BetterRobot's format: ID, level (MUST / SHOULD), rule, rationale, example,
enforcement. The BetterRobot contracts in `BetterRobot/docs/contracts/` apply first (see [index](index.md)).

## H-LAYER-1 Imports follow the BetterHuman package DAG (MUST)

`core` and `io` import nothing from `better_human`. `families` imports `core` and `io`. `terms` imports `core`.
`viz` (including `viz/studio`) imports anything below it and the package root. Nothing below `viz` imports `viz`.
Every package may import BetterRobot's public surface (`C-DEP-1`, `tests/contract/test_br_imports.py`).

Rationale: the body model (`core`) and weight-file reading (`io`) stay usable without any family table or viewer.
Example: `src/better_human/families/smpl_family.py` imports `core` and `io`; `src/better_human/terms.py`
imports only `core`.
Enforcement: `tests/contract/test_layer_dependencies.py`.

## H-API-1 One body API across families (MUST)

A family loader (`smpl`, `smplh`, `smplx`, `mano`, `flame`, `mhr`) returns a `Body`
(`src/better_human/core/body.py:23`). `Body.bake(...)` gives an `Identity` (`core/body.py:173`) that owns the
baked BetterRobot model; `identity.forward(q, ...)` gives a `HumanState` (`core/body.py:229`). Families declare
*what* they are through a `FamilySpec` (`core/spec.py`); the *how* (LBS, blend shapes, correctives, regressors)
lives once in `core`. Partial models are loader parameters (`hands=False`), never subclasses.
Rationale: one forward pass to test and keep in parity; see principles 6 and 7 below.
Example: `src/better_human/families/smpl_family.py:59` (`load_family`) serves all five SMPL-family loaders.
Enforcement: review; parity gates below.

## H-API-2 Arguments after the second are keyword-only (SHOULD)

Same rule as BetterRobot `C-API-2` (at most two positional parameters). Current violations are listed file:line in the test's
allowlist; the allowlist only shrinks.
Enforcement: `tests/contract/test_keyword_only.py`.

## H-ASSET-1 Shipped assets live in `families/assets/`; licensed weights are never shipped (MUST)

Small, redistributable data (segmentations, densities) ships in `src/better_human/families/assets/` and is read
with `importlib.resources`. Licensed weights are never downloaded, bundled or cached (principle 10).
Example: `families/assets/smpl_vert_segmentation.json`, read by `viz/studio/persistence.py:14`.
Enforcement: review.

## H-ASSET-2 Weights resolve from an explicit path, else `$BETTERHUMAN_MODELS_DIR` (MUST)

An explicit `path` argument wins; otherwise the file is `$BETTERHUMAN_MODELS_DIR/<folder>/<file_pattern>` with
the patterns of `families/tables.py` (table under "Weight-file layout"). A missing or malformed file raises a
clear error naming the path; there is no silent fallback.
Example: `src/better_human/io/load.py:26` (`MODELS_ENV`), `io/load.py:34` (resolution).
Enforcement: `tests/io/`.

## H-STUDIO-1 Studio is a thin shell over the BetterRobot viewer (MUST)

The studio adds no viewer machinery: `app.py` drives `better_robot.viewer.Viewer` (`viz/studio/app.py:12`) and
`layers.py` subclasses `better_robot.viewer.Layer` (`viz/studio/layers.py:10`). Domain logic lives in
`state.py`, `persistence.py`, `posing.py` and `gizmo.py` and is testable without a server; UI wiring lives in
`app.py`, `layers.py`, `segment_editor.py` and `frames_editor.py`. A generic viewer feature goes to BetterRobot.
Rationale: principle 5 (not human-specific means it belongs in BetterRobot).
Enforcement: review; `tests/viz/` exercises the domain modules headless.

## H-STUDIO-2 Studio data layout is `<data>/{segmentations,frames,poses}/<key>/` (MUST)

Studio state is JSON under one data root (default `studio_data/`): vertex segmentations, named frame sets and
poses each in their own directory, keyed by the mesh-topology key.
Example: `src/better_human/viz/studio/persistence.py:33` (`segmentations`), `:37` (`frames`), `:41` (`poses`).
Enforcement: `tests/viz/`.

## Numeric conventions, layout, gates and baselines

Folded from the former `docs/contracts.md`. Parity gates are rule `H-PARITY-1` (MUST): never weaken them.

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
- SMPL: `nq = 99`, `nv = 75`. Read `identity.nq`, `identity.nv`, `identity.q_neutral`; do not hard-code.
- Move on the manifold with BR: `identity.model.integrate(q, dq)`.

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

- `bake`, `forward`, `vertices`, `points`, `landmarks`, mass (`Inertia.from_mesh`) and BR
  kinematics/dynamics are differentiable in the identity values, `q` and streams.
- `HumanState` memoises `vertices`, `frame_poses` and `joints`; create a new state for new inputs, never mutate one.

### Classic conversion

- `<family>_from_classic(identity, classic) -> (q, stream)` and
  `<family>_to_classic(identity, q, stream=None) -> <Family>Classic` in `families/classic.py`.
  MHR: `mhr_from_classic(body, classic) -> (identity, q, stream)` and
  `mhr_to_classic(identity, q, stream=None)` in `families/mhr.py`.
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

### Scope

- Families are in-tree only during alpha: a new family lands as a module under
  `better_human/families/`; there is no public `register_family` hook.
- Retargeting (motion transfer between skeletons) is not BH's job; it belongs in a separate package.
- Fitting (bodies to scans, keypoints or video) is not BH's job either; BH supplies the
  differentiable bodies and terms such a fitter uses.

### Weight-file layout

Files live under `$BETTERHUMAN_MODELS_DIR/<folder>/<file_pattern>`; an explicit `path` argument
wins. Patterns are in `families/tables.py`:

| family | folder | file pattern |
|---|---|---|
| SMPL | `smpl` | `SMPL_{gender}.npz` |
| SMPL-H | `smplh` | `smplh/{gender_lower}/model.npz` |
| SMPL-X | `smplx` | `SMPLX_{gender}.npz` |
| MANO | `mano` | `MANO_{side}.npz` |
| FLAME | `flame` | `{gender}_model.npz` |
| MHR | `MHR/converted` | `mhr_lod{0..6}.npz` (default LOD 1) |

### Parity gates (do not weaken)

| gate | test | tolerance | status |
|---|---|---|---|
| SMPL vs upstream goldens | `tests/parity/test_smpl_parity.py` | see test | pass |
| SMPL-X, 29 goldens, float32 and float64, expression streamed | `tests/parity/test_smplx_parity.py` | 1e-6 m vertices and joints | worst 5.1e-7 m |
| SMPL-X with `stream_joints=False` | same file | error in (1e-5, 2e-3) m | about 1.2 mm |
| MHR, LOD 0-6, float32 / float64 | `tests/parity/test_mhr_parity.py` | f32 `(5e-6, 1e-6, 1e-6, 1e-7)`, f64 `(2e-6, 5e-7, 5e-7, 1e-7)` | pass |
| MHR vs TorchScript `mhr_model.pt` | `tests/parity/test_mhr_torchscript.py` | vertices 2e-6 m, positions 1e-6 m, quaternions 1e-6 | pass |

Parity tests skip only when the licensed asset is absent.

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

Folded from the former `docs/design_principles.md`. They mirror BetterRobot's rules; when the two disagree,
BetterRobot's numerical contracts win.

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
    on CPU and on pinned CUDA hardware when it is available; an unavailable
    device produces a visible skip and is never reported as a pass.

### Process

14. **Alpha: breaking changes are fine, compatibility shims are not.** No
    aliases, no deprecation paths.
15. **Docs move with the code in the same change.**
16. **Deletion needs evidence; results reports are honest** — exact counts,
    deviations named as findings.
17. **One behavior, one test; test behavior, not trivia.** float32
    gradchecks, following the workspace testing rules.
