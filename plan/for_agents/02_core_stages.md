# Order 02 — io schema + core pipeline stages (SMPL-neutral)

> **Implementation log (2026-07-21, `dev`):** Complete. The 66-field MHR
> schema validates the real LOD6 archive while skipped corrective payloads are
> checked from headers only. Generic 6D/MLP/full-bind/scaled-LBS stages and two
> body policy seams landed; all SMPL/SMPL-X parity maxima are exactly unchanged.
> Upstream MHR has no MLP biases, so the generic bias remains optional.

Adds the generic machinery MHR needs to `io/` and `core/` **without changing
SMPL/SMPL-X behavior in any way**. Independent of order 01 (needs only the
reference formulas and existing tests); may run in parallel with it.

## Ground rules (binding)

- Read `docs/design_principles.md`, `docs/architecture.md`, `docs/contracts.md`,
  `references/design/MHR.md` before coding.
- **No BetterRobot modifications.** Report BR gaps in the results file.
- **SMPL/SMPL-X parity and full suite must be byte-unchanged** — the measured
  baselines in `docs/contracts.md` §Measured baselines bind this order.
- The safety net only grows; alpha, no back-compat shims.
- Write `plan/for_agents/02_results.md`: delivered, deviations, exact test
  counts, LOC deltas per file.

## 1. io — schema + loading

- `io/schema.py`: `MHR_SCHEMA` validating the per-LOD npz emitted by order 01
  (names, shapes, dtypes up front — same style as `SMPL_SCHEMA`). Keep it
  explicit; do not resurrect a symbolic-dimension DSL.
- `io/load.py`: resolution `model_path` arg →
  `$BETTERHUMAN_MODELS_DIR/MHR/converted/mhr_lod{n}.npz` → actionable error.

If order 01 has not landed yet, code against the schema spec in
`plan/for_agents/01_convert_goldens.md` and validate with a small synthetic
npz fixture; the real-file check happens in order 03.

## 2. core/pipeline.py — three new stages

Written once, generic, unit-tested against the reference formulas
(`references/parametric_body/MHR/mhr/{mhr,io,utils}.py`):

- **6D pose feature**: given per-joint Euler channel values `(rx, ry, rz)`
  (or equivalently local rotation matrices), produce the first two columns of
  `R = Rz(rz)·Ry(ry)·Rx(rx)`, flattened, **minus the identity feature** —
  exactly the reference `batch6DFromXYZ` with identity subtraction
  (`mhr.py:63-67`). Test: port the reference trig formula directly and compare
  on random angles to machine precision. No Euler angles should be needed by
  callers that already have rotation matrices/quaternions — provide the
  matrix-based entry point.
- **MLP corrective**: masked-dense first layer (bias, ReLU) + dense output
  layer `[3000, V·3]`. Numerically identical to the reference `SparseLinear`
  path — but do **not** copy the reference's rebuild-dense-weight-every-forward
  hack (`utils.py:115-140`); bake the mask into the dense weight once at
  construction. Functional: returns offsets, never mutates its input (the
  reference's in-place `rest_pose +=` aliasing is a known wart).
- **Generalized skinning**:
  - `skinning_transforms` variant accepting full 7D rest poses (non-identity
    bind **rotations** — MHR's pre-rotation chains make bind orientations
    non-trivial), i.e. `T_skin = world(q) ∘ inverse(bind_pose)`;
  - `lbs` accepting an optional per-joint scalar scale `s_j` applied as a
    similarity transform per influence:
    `p' = Σ_j w_j · (t_j + s_j · R_j · (R_bind_j⁻¹ · (p − t_bind_j)))`.
  - At `s = 1` and identity bind rotations these must reduce **exactly** to
    the current SMPL path (add a test asserting tensor equality, not
    closeness, on the SMPL fixture).

## 3. Seam promotion — minimal

The pose-corrective and shape-bake policies now have two in-tree callers
(linear/SMPL vs MLP/MHR; regressed-joints vs authored-skeleton). Promote
**only what the second caller forces**, per the pluggability policy in
`docs/architecture.md`: small named functions or a policy field on the body —
not a protocol framework, not a plugin registry. Notes:

- `ParametricBody.vertices_from_data` currently hard-codes the SMPL feature
  (all-spherical quats → `(R−I)` → linear posedirs). It needs a per-model
  corrective policy; keep `ParametricBody` one class (architecture:
  "one class, configured, never a family tree").
- `with_shape` currently hard-codes regressed joints + pelvis centering.
  MHR's bake (order 03) is different math; structure the split so order 03
  can supply its bake without copying pipeline code. Prefer the smallest
  refactor that keeps SMPL's path literally identical.
- **No BetterHuman-side ParameterTransform machinery**: BR's
  `reduced_coordinates.expand_configuration(structure, q)` already maps public
  q → full per-joint values; MHR will reuse it (order 03). Do not duplicate.

## Acceptance

- SMPL/SMPL-X full suite green with **unchanged parity numbers** (compare
  against `docs/contracts.md` baselines; run the parity tests before and
  after and diff the reported maxima).
- New stages covered by unit tests: reference-formula equivalence (6D feature,
  MLP), exact-reduction test (generalized skinning at scale=1/identity binds),
  float32 gradchecks through each new stage, batching with arbitrary leading
  axes, dtype/device preservation.
- ruff clean; LOC report per file in the results.
