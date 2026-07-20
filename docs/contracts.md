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

- SI everywhere: meters, kilograms, seconds, radians. Models with other
  native units (MHR is centimeters) convert inside their loader/converters;
  no unit flags on the public API.
- World up-axis, handedness, and rest orientation follow the model's
  canonical rest pose expressed in BetterRobot's world frame; each model's
  `BodyStructure` records its native conventions (unit, up, forward) and the
  loader normalizes once at load (SOMA's `NATIVE_*` pattern).

## State

- The only pose representation in BetterHuman is BetterRobot `q` on the
  body's kinematic tree. Axis-angle, Euler angles, rotation matrices, and
  4×4 stacks may exist **only** inside `from_classic` / `to_classic` and
  weight-file loaders.
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

- `from_classic` and `to_classic` are exact inverses:
  `to_classic(from_classic(c)) == c` and
  `from_classic(to_classic(q)) == q` to documented tolerance, for every
  model, covered by round-trip tests.
- Conversions may depend on baked shape (the SMPL pelvis offset `J₀(β)`
  does); they read it from the body, never take it as an extra argument.
- Classic parameter containers are per-model typed dataclasses. No
  positional mega-tensors, no quaternion scalar-order mixing: everything
  crossing the boundary is converted to scalar-last immediately.

## Weight files

- One self-describing `.npz` per model; the loader validates field names,
  shapes, and dtypes against the schema before constructing anything, and
  errors name the offending field, the file, and the fix. Validation is
  **required-subset semantics**: every schema field must be present and
  correct; unknown extra fields are ignored (logged at debug) — vendor
  files carry UV/landmark/dict cruft (the MPI SMPL-X `.npz` does) that must
  not block loading. No `print`; use `logging`.
- Resolution order: explicit `model_path` argument, else
  `$BETTERHUMAN_MODELS_DIR/<family>/<canonical-name>`, else raise with both
  options spelled out. The library never downloads weights and the
  repository never contains gated weights.
- Loaders never densify sparse regressors and never mutate the source file.

## Parity (the safety net)

- Every supported model reproduces its original implementation's vertices
  and joints on the golden grid: float64 oracle vs our float64 pass agrees
  to ≤ 1e-6 (m); the float32 tolerance is measured and pinned when the
  goldens land, not guessed.
- Golden grids must include each model's known traps (SMPL-X hand mean
  pose, jaw+eyes-before-hands pose ordering, expression-as-shape-columns,
  hand-PCA expansion).
- Parity, round-trip, and gradcheck tests are never weakened, skipped-in,
  or tolerance-bumped to land a change. Tests that need local weights skip
  with an explicit reason when weights are absent — skipping is visible,
  not silent.
