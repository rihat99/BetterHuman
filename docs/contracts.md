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
| SMPL CPU batch-256 performance vs upstream `smplx` | pose-only 1.206803×; shape+pose 1.344826× | 30-call shared-host medians; observational, no target |
| SMPL-X CPU batch-256 performance vs upstream `smplx` | pose-only 1.088869×; shape+pose 1.349441× | 30-call shared-host medians; observational, no target |
| SMPL CUDA Warp pose-only performance vs upstream `smplx` | batch 16: 1.462866×; batch 256: 1.380076× | 30-call medians; observational, no target |
| SMPL-X CUDA Warp pose-only performance vs upstream `smplx` | batch 16: 0.799960×; batch 256: 0.919583× | 30-call medians; observational, no target |
| SMPL CUDA shape+pose performance vs upstream `smplx` | batch 16: 11.063593×; batch 256: 7.651324× | fair native-CUDA path; Torch-FK-bound; observational, no target |
| SMPL-X CUDA shape+pose performance vs upstream `smplx` | batch 16: 13.558983×; batch 256: 4.604730× | fair native-CUDA path; Torch-FK-bound; observational, no target |
| MHR LOD1 CPU batch-256 performance vs TorchScript | pose-only 0.232685×/0.164993×; shape+pose 0.352023×/0.287297× (correctives on/off) | 30-call medians; shared-host timing |
| MHR LOD1 CUDA Warp pose-only performance vs TorchScript | batch 16: 0.899320×/1.056314×; batch 256: 0.744054×/0.736357× (correctives on/off) | both batches meet `≤2×` after matrix-blend LBS |
| MHR LOD1 CUDA shape+pose performance vs TorchScript | batch 16: 37.085810×/61.101840×; batch 256: 12.738454×/16.263777× (correctives on/off) | fair path remains on BetterRobot Torch FK and misses target |
| Partial-model property (SMPL-X reduced vs full-with-identity DOFs) | ≤ 5e-7 m | keep |
| Neutral SMPL mass — argmax segmentation (density 1000) | 73.43 kg parts / 74.76 kg whole, 1.78 % gap | the argmax fallback's inherent gap |
| Neutral SMPL volume — authored closed-3D segmentation | 24-part sum within +0.0055 % of whole mesh | authored path must hold this |
| Aggregate vs whole-mesh COM separation | 0.02 mm (authored), 2.47 mm (argmax) | |
| SMPL/SMPL-X CRBA min eigenvalue (SPD check) | > 0 | keep |
| Test gate | 226 passed, 19 warnings on pinned RTX 6000 Ada hardware | all five CUDA cases executed; unavailable hardware skips stay visible elsewhere |

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
