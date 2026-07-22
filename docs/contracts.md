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

For performance rows, Workload A recomputes shape and pose from supplied
parameters on every call; its reference ratios compare equivalent computations.
Workload B bakes shape once and streams poses; its ratio compares per-frame user
cost because neither upstream reference exposes a shape-amortized API.
SMPL-family slash-separated values are SMPL / SMPL-X; MHR values are
correctives on / off. `F+B / F` is total forward-plus-backward divided by
separately measured forward time, not a backward-only median. Official Warp
rows are forward-only; combining `--use-warp --grad` uses an explicitly labeled
Warp-forward/Torch-recompute-VJP hybrid. All performance values are float32
batch-256 medians of 30 calls after five warmups and are subject to shared-host
scheduling and concurrent-load noise.

| Quantity | Baseline | Note |
|---|---|---|
| SMPL/SMPL-X float64 parity vs `smplx` oracle | ≤ 1e-6 m (worst measured ~1.04e-7) | sacred; never weakened |
| SMPL/SMPL-X float32 parity | pinned at 1e-6 (worst ~7e-7) | keep pinned |
| MHR float32 parity vs float32 `pymomentum` oracle | V ≤ 5e-6 m; P ≤ 1e-6 m; Q ≤ 1e-6; S ≤ 1e-7 | applies independently to every LOD |
| MHR float64 parity vs float32 `pymomentum` oracle | V ≤ 2e-6 m; P ≤ 5e-7 m; Q ≤ 5e-7; S ≤ 1e-7 | mixed-precision oracle; not float64-oracle parity |
| MHR LOD1 TorchScript vs float32 golden | V ≤ 2e-6 m; P ≤ 1e-6 m; Q ≤ 1e-6; S ≤ 1e-7 | independent secondary oracle; worst measured V 1.3113021850585938e-6 m; measured P/Q/S bit-exact |
| SMPL family CPU forward | A: 1.309162× / 1.418376×; B: 1.033997× / 1.315579× | observational, no target |
| SMPL family native-CUDA forward | A: 7.917632× / 6.613593×; B: 7.633592× / 6.462927× | synchronized; observational, no target |
| SMPL family Warp-requested forward | A on Torch FK: 7.016172× / 9.535991×; B on Warp: 1.338540× / 0.950417× | synchronized; observational, no target |
| SMPL family CPU gradient, A | 106.526944 ms (3.183103 F+B/F, 2.057300× upstream) / 221.165732 ms (3.017203, 1.968564×) | observational, no target |
| SMPL family CPU gradient, B | 66.829707 ms (4.457463 F+B/F, 1.363085× upstream) / 213.446547 ms (3.574656, 1.606579×) | observational, no target |
| SMPL family native-CUDA gradient, A | 86.299966 ms (3.852947 F+B/F, 12.397440× upstream) / 213.747098 ms (3.041208, 10.571821×) | synchronized; observational, no target |
| SMPL family native-CUDA gradient, B | 70.088864 ms (3.352008 F+B/F, 8.476885× upstream) / 185.992442 ms (2.720660, 9.811030×) | synchronized; observational, no target |
| MHR LOD1 CPU forward | A: 206.352065 / 249.103795 ms (0.226302× / 0.316514×); B: 136.900657 / 148.225082 ms (0.150136× / 0.188336×) | observational, no target |
| MHR LOD1 native-CUDA forward | A: 169.116997 / 161.538835 ms (14.942053× / 17.634505×); B: 84.796631 / 164.078545 ms (7.492066× / 17.911754×) | synchronized; observational, no target |
| MHR LOD1 Warp-requested forward | A on Torch FK: 164.781857 / 163.516999 ms (12.686466× / 15.811168×); B on Warp: 8.292859 / 6.657831 ms (0.638463× / 0.643775×) | synchronized; observational, no target |
| MHR LOD1 CPU gradient, A | 656.606512 / 722.676883 ms; 2.573378 / 2.192284 F+B/F | absolute; observational, no target |
| MHR LOD1 CPU gradient, B | 490.848457 / 510.086399 ms; 3.094853 / 2.763079 F+B/F | absolute; observational, no target |
| MHR LOD1 native-CUDA gradient, A | 617.513200 / 603.712264 ms; 1.641031 / 2.763036 F+B/F | synchronized absolute; observational, no target |
| MHR LOD1 native-CUDA gradient, B | 363.791832 / 398.367353 ms; 2.008058 / 3.698338 F+B/F | synchronized absolute; observational, no target |
| Partial-model property (SMPL-X reduced vs full-with-identity DOFs) | ≤ 5e-7 m | keep |
| Neutral SMPL mass — argmax segmentation (density 1000) | 73.43 kg parts / 74.76 kg whole, 1.78 % gap | the argmax fallback's inherent gap |
| Neutral SMPL volume — authored closed-3D segmentation | 24-part sum within +0.0055 % of whole mesh | authored path must hold this |
| Aggregate vs whole-mesh COM separation | 0.02 mm (authored), 2.47 mm (argmax) | |
| SMPL/SMPL-X CRBA min eigenvalue (SPD check) | > 0 | keep |
| Test gate | 230 passed, 19 warnings in 158.23s on pinned RTX 6000 Ada hardware | all six CUDA-guarded cases executed; zero skipped |

The complete observational tables, agreement rows, environment, and commands
are recorded in [Order 02 results](../plan/for_agents/02_results.md).

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
