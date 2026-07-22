# Contracts

Normative rules every BetterHuman module and every future model must obey.
BetterRobot's contracts apply wholesale; this page adds the human-specific
ones. Breaking a contract is a bug even if tests pass.

## Inherited from BetterRobot (restated, not redefined)

- Pose: `[tx, ty, tz, qx, qy, qz, qw]`, quaternion scalar-last. Twist:
  `[linear(3), angular(3)]`.
- `nq ≠ nv` in general; never `q + v` — use `Model.integrate` /
  `Model.difference`.
- Public tensors: trailing event axes, arbitrary leading batch axes,
  PyTorch broadcasting.
- Preserve dtype and device; never silently cast, move, or squeeze; no host
  synchronization in hot loops; never silently fall back to a slow path
  (warn once).

## Units and frames

- SI everywhere: meters, kilograms, seconds, radians. Raw MHR assets and
  oracle outputs use centimeters; its offline converter stores meters and the
  runtime loader validates that normalized unit. There are no unit flags on
  the public API.
- Shape-derived mass uses 1000 kg/m³ by default. A density override is either
  scalar, a tensor in the model-declared physical-part order with one positive
  kg/m³ value per part, or a named density table resolved to that tensor. MHR
  declares 19 reference segmentation groups. The BetterRobot universe remains
  massless. Packed COM and inertia values are expressed in each part's
  rest-joint frame.
- Segmentation has two sources but one closed topology. A model with an authored
  closed-3D asset uses it (volume-preserving joint-cap parts, the SMPL-family
  default); a model without one falls back to argmax skinning assignment with
  centroid-capped parts. Both yield a single `PartMeshTopology` fed to the
  differentiable `Inertia.from_mesh`; neither is a per-query cost.
- World up-axis, handedness, and rest orientation follow the model's
  canonical rest pose expressed in BetterRobot's world frame; each model's
  `BodyStructure` records its native conventions (unit, up, forward) and the
  loader normalizes once at load (SOMA's `NATIVE_*` pattern).

## State

- The only pose representation in BetterHuman is BetterRobot `q` on the
  body's kinematic tree. Axis-angle, Euler angles, rotation matrices, and
  4×4 stacks may exist only as model-private intermediates inside classic
  converters, weight-file loaders, and pose-feature policies; they are never
  parallel public pose state.
- Shape lives on the model side. `with_shape(**named_groups)` is the only
  way shape enters; pose-query methods (`fk`, `vertices`, Jacobians,
  dynamics) must not accept shape parameters. Parameter groups are named
  and typed (`betas`, `expression`, …); a model declares its groups and
  their widths in `BodyStructure`.
- `ParametricBody` instances are functionally immutable: `with_shape`
  returns a new body sharing static structure; nothing mutates in place.

## Batching

- Shape groups accept arbitrary leading batch dims: `betas (B, n_betas)`
  produces batch-bearing `BodyValues` products (`v_shaped (B, V, 3)`) and BR
  `ModelValues` tables (`joint_placements (B, njoints, 7)`), broadcast
  against `q` per BetterRobot's execution-batch rules.
- Trajectories: per-body values combined with `(B, T, nq)` configurations
  require the singleton time axis on value tables (`(B, 1, njoints, 7)`) —
  batch broadcast is right-aligned and excludes event axes.
- Vertex outputs are `(..., V, 3)` with the same leading batch semantics.

## Differentiability

- The full chain `shape groups → BodyValues → ModelValues → FK/dynamics →
  vertices/residuals` is autograd-clean. No `.item()`, no in-place ops that
  break the graph, no detached caches on the differentiable path. Cached
  shape products are tensors in the graph of the `with_shape` call that
  produced them.
- float32 is the test standard, including gradchecks (workspace rule).
  Oracles may run float64; parity tolerances are calibrated per dtype and
  recorded next to the test.

## Classic conversion

- `from_classic` and `to_classic` are exact inverses over canonical numerical
  state, to documented tolerance and with round-trip tests. A model whose
  classic container includes shape state may return `(shaped_body, q)` from
  `from_classic`; its inverse is `shaped_body.to_classic(q)`. Reference aliases
  such as MHR's absent expression and an explicit zero expression may
  canonicalize to one representation.
- Conversions may depend on baked shape (the SMPL pelvis offset `J₀(β)`
  does); they read it from the body, never take it as an extra argument.
- Classic parameter containers are per-model typed dataclasses. No
  positional mega-tensors, no quaternion scalar-order mixing: everything
  crossing the boundary is converted to scalar-last immediately.

## Weight files

- One self-describing `.npz` per model variant or LOD; the loader validates field names,
  shapes, and dtypes against the schema before constructing anything, and
  errors name the offending field, the file, and the fix. Validation is
  **selected-subset semantics**: every selected schema field must be present
  and correct; explicitly excluded optional payloads are neither required nor
  inspected; unknown extra fields are ignored. Vendor files carry
  UV/landmark/dict cruft (the MPI SMPL-X `.npz` does) that must not block
  loading. No `print`; use `logging`.
- Resolution order: explicit `model_path` argument, else
  `$BETTERHUMAN_MODELS_DIR/<family>/<canonical-name>`, else raise with both
  options spelled out. The library never downloads weights and the
  repository never contains gated weights.
- The legacy SMPL-family converter emits the joint regressor dense; MHR has an
  authored skeleton and no joint regressor. Loaders expose every selected
  numeric field as a dense tensor. Unicode metadata remains a NumPy array
  because PyTorch has no string tensor dtype; loaders never mutate the source
  file.

## Parity (the safety net)

- Every supported model reproduces its original implementation's vertices
  and joints on the golden grid. A true float64 oracle compared with our
  float64 path agrees to ≤ 1e-6 m. When an upstream oracle serializes only
  float32 output (MHR through `pymomentum`), both BetterHuman dtypes are
  compared with that same float32 grid. Position, quaternion, and scale
  tolerances are measured and pinned by dtype; mixed f64-vs-f32 results are
  not presented as float64-oracle parity. Quaternion comparisons are
  sign-agnostic.
- Golden grids must include each model's known traps (SMPL-X hand mean
  pose, jaw+eyes-before-hands pose ordering, expression-as-shape-columns,
  hand-PCA expansion).
- Parity, round-trip, and gradcheck tests are never weakened, skipped-in,
  or tolerance-bumped to land a change. Tests that need local weights skip
  with an explicit reason when weights are absent — skipping is visible,
  not silent.

## Measured baselines

Recorded when the corresponding safety net landed. Correctness pins must not
regress. Performance rows explicitly marked observational are reproducible
reference points to remeasure and report, not pass/fail targets.

| Quantity | Baseline | Note |
|---|---|---|
| SMPL/SMPL-X float64 parity vs `smplx` oracle | ≤ 1e-6 m (worst measured ~1.04e-7) | sacred; never weakened |
| SMPL/SMPL-X float32 parity | pinned at 1e-6 (worst ~7e-7) | keep pinned |
| MHR float32 parity vs float32 `pymomentum` oracle | V ≤ 5e-6 m; P ≤ 1e-6 m; Q ≤ 1e-6; S ≤ 1e-7 | applies independently to every LOD |
| MHR float64 parity vs float32 `pymomentum` oracle | V ≤ 2e-6 m; P ≤ 5e-7 m; Q ≤ 5e-7; S ≤ 1e-7 | mixed-precision oracle; not float64-oracle parity |
| MHR LOD1 TorchScript vs float32 golden | V ≤ 2e-6 m; P ≤ 1e-6 m; Q ≤ 1e-6; S ≤ 1e-7 | independent secondary oracle; worst measured V 1.3113021850585938e-6 m; measured P/Q/S bit-exact |
| Partial-model property (SMPL-X reduced vs full-with-identity DOFs) | ≤ 5e-7 m | keep |
| Neutral SMPL mass — argmax segmentation (density 1000) | 73.43 kg parts / 74.76 kg whole, 1.78 % gap | the argmax fallback's inherent gap |
| Neutral SMPL volume — authored closed-3D segmentation | 24-part sum within +0.0055 % of whole mesh | authored path must hold this |
| Aggregate vs whole-mesh COM separation | 0.02 mm (authored), 2.47 mm (argmax) | |
| SMPL/SMPL-X CRBA min eigenvalue (SPD check) | > 0 | keep |
| Test gate | 230 passed, 19 warnings in 116.88s on pinned RTX 6000 Ada hardware | all six CUDA-guarded cases executed; zero skipped |

### Compact CUDA performance

The current observational baseline comes from one official invocation of
[`benchmark_bodies.py`](../benchmarks/benchmark_bodies.py): float32, batch sizes
1 and 256, five warmups, and 30 explicitly synchronized timed calls per case.
The tables report medians. The host carried an NVIDIA RTX 6000 Ada Generation,
Torch 2.13.0+cu126, CUDA runtime 12.6, driver 560.35.03, and Warp 1.15.0.
Shared-host scheduling and concurrent load may affect these values.

BetterRobot was revision
`8c4e619ab1262863b13678698e53f3f0ac9d7eed`; its eager matrix FK landed in
`8fc19a2`. The BetterHuman base was `c8da7ec`, with a dirty worktree containing
the compact benchmark implementation. Only eager Torch and fused Warp were
measured; the compiled FK lane requires a separate cold-start and cache
protocol and has no RNEA counterpart.

The reference column is a full forward from upstream `smplx` or the MHR LOD1
TorchScript release. BetterHuman bakes shape outside timing and streams poses.
Neither reference has a shape-amortized API, so each ratio is a user-workload
comparison rather than a matched-FLOP comparison. These rows are observational,
not pass/fail targets.

#### Fixed-shape vertex streaming

| Body | Batch | Reference full (ms) | BH Torch (ms) | × ref | BH Warp (ms) | × ref |
|---|---:|---:|---:|---:|---:|---:|
| SMPL | 1 | 1.552809 | 3.928650 | 2.530028 | 2.199730 | 1.416613 |
| SMPL | 256 | 2.318609 | 4.107289 | 1.771445 | 2.938329 | 1.267281 |
| SMPL-X | 1 | 1.707323 | 2.722384 | 1.594534 | 1.316091 | 0.770850 |
| SMPL-X | 256 | 4.319260 | 5.333991 | 1.234932 | 3.847292 | 0.890729 |
| MHR | 1 | 3.039193 | 7.717042 | 2.539175 | 2.327013 | 0.765668 |
| MHR | 256 | 12.105253 | 13.400580 | 1.107005 | 8.654711 | 0.714955 |

The pre-timing vertex gate passed for both BetterHuman lanes. Worst maximum
absolute error was `5.96046448e-7 m` for the SMPL family and
`1.43051147e-6 m` for MHR.

#### Absolute FK and RNEA

| Body | Joints | nv | Batch | FK Torch (ms) | FK Warp (ms) | RNEA Torch (ms) | RNEA Warp (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| SMPL | 25 | 75 | 1 | 2.514741 | 0.625391 | 11.645354 | 1.053602 |
| SMPL | 25 | 75 | 256 | 1.589354 | 0.453229 | 6.956790 | 0.806229 |
| SMPL-X | 56 | 168 | 1 | 1.871313 | 0.362131 | 12.951559 | 0.630839 |
| SMPL-X | 56 | 168 | 256 | 1.956333 | 0.617371 | 13.316848 | 1.181127 |
| MHR | 203 | 131 | 1 | 6.387257 | 0.898219 | 88.525171 | 1.613611 |
| MHR | 203 | 131 | 256 | 6.693205 | 1.820299 | 53.731947 | 4.096781 |

RNEA uses deterministic nonzero configurations, velocities, and accelerations,
shape-specific masses, and reused `Data` workspaces. It has no original or
upstream comparison. Sign-aligned Torch/Warp FK passed `rtol=2e-5` and
`atol=2e-6`; RNEA passed `rtol=3e-4` and `atol=2e-4`. Any Warp fallback warning
was a hard failure. MHR's absolute RNEA timing does not establish support for
solve-based dynamics with its rank-deficient coordinates.

For MHR, V denotes vertices, P native joint positions, Q sign-agnostic native
joint quaternions, and S native joint scale. V and P are measured in meters;
Q and S are dimensionless. Full-grid maxima are:

| Dtype | LOD | V | P | Q | S |
|---|---:|---:|---:|---:|---:|
| float32 | 0 | 2.02656e-6 | 3.57628e-7 | 2.98023e-7 | 0 |
| float32 | 1 | 2.80142e-6 | 7.59959e-7 | 6.55651e-7 | 0 |
| float32 | 2 | 1.96695e-6 | 3.57628e-7 | 2.98023e-7 | 0 |
| float32 | 3 | 2.26498e-6 | 7.15256e-7 | 4.76837e-7 | 0 |
| float32 | 4 | 2.02656e-6 | 3.57628e-7 | 3.57628e-7 | 0 |
| float32 | 5 | 1.90735e-6 | 4.76837e-7 | 2.98023e-7 | 0 |
| float32 | 6 | 1.78814e-6 | 2.68221e-7 | 3.57628e-7 | 0 |
| float64 | 0 | 6.79632e-7 | 1.83320e-7 | 1.31398e-7 | 2.48013e-8 |
| float64 | 1 | 8.48870e-7 | 2.21570e-7 | 1.92913e-7 | 4.66294e-8 |
| float64 | 2 | 5.58113e-7 | 1.31547e-7 | 1.02206e-7 | 5.79166e-8 |
| float64 | 3 | 5.61791e-7 | 9.95071e-8 | 1.08197e-7 | 2.90327e-8 |
| float64 | 4 | 5.53013e-7 | 2.06190e-7 | 1.33385e-7 | 2.55663e-8 |
| float64 | 5 | 4.46301e-7 | 1.22673e-7 | 1.22210e-7 | 2.29539e-8 |
| float64 | 6 | 5.93538e-7 | 1.20236e-7 | 1.82994e-7 | 2.92798e-8 |

MHR currently does not meet the CRBA baseline. Its physical rotation transform
has rank 119/125 because the overcomplete spine controls contain six
structural null directions; consequently, any physically derived MHR CRBA has
rank at most 125/131. The current 19-group inertia placement measures rank
50/131 at neutral LOD1, with 72 zero diagonal entries and
`λmin = -7.7016e-15` from numerical roundoff. PSD-not-SPD is structural under
this overcomplete configuration; solve-based MHR dynamics remain unsupported
pending an independent-coordinate redesign.
