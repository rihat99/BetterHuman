# Order 02 results — SMPL-neutral MHR core stages

Completed on branch `dev` on 2026-07-21. This order adds the shared loader and
tensor machinery needed by MHR and promotes only the two model-policy seams
forced by the next body implementation. BetterRobot was not modified.

## Delivered

- An explicit 66-field `MHR_SCHEMA` covering every converted archive member,
  including exact shapes, storage dtypes, the shared vertex dimension, and the
  derived `V*3` corrective width.
- `load_model_npz(..., exclude_fields=...)` with required-field checks, full
  shape/dtype validation, Unicode metadata preservation, and lazy exclusion.
  Excluded members have their NPY headers validated without reading payloads,
  which keeps the 2.58 GiB LOD0 corrective layer lazy.
- A verified canonical resolver call for
  `$BETTERHUMAN_MODELS_DIR/MHR/converted/mhr_lod{n}.npz`; the existing generic
  resolver already implemented the required precedence and error behavior.
- Exact MHR 6D pose features from either XYZ Euler channels or local rotation
  matrices, with identity subtraction and arbitrary leading batch axes.
- One-time sparse-to-dense first-layer baking and a functional, non-mutating
  ReLU MLP corrective stage. The output weight remains `(3000, V*3)`, avoiding
  a multi-gigabyte transpose at LOD0.
- Full 7D bind-pose skinning transforms and optional scale-aware LBS about bind
  joint positions. Scale-one/identity-bind execution is tensor-equal to the
  legacy SMPL path, including on the real SMPL fixture.
- Two function policies on `ParametricBody`: model-specific shape baking and
  pose correctives. The prior SMPL operations were moved verbatim to
  `smpl_family.py` and bound for both SMPL and SMPL-X.
- Synthetic schema/loader fixtures, formula tests, arbitrary-leading-axis
  tests, exact reduction tests, float32 gradchecks, and explicit dtype/device
  assertions. The Weight files reference contract now states that Unicode
  metadata remains NumPy because PyTorch has no string tensor dtype.

## Real archive checks

`models/MHR/converted/mhr_lod6.npz` passed the final schema and loader:

- full load: 66 fields;
- corrective-excluded load: 58 fields, with all eight skipped headers still
  shape/dtype-validated and no corrective payload indexed;
- template vertices: `(595, 3)` `torch.float32`;
- joint names: `(127,)` NumPy `<U20`;
- corrective output: `(3000, 1785)` `torch.float32`;
- baked first layer: `(3000, 750)`, exactly 53,136 nonzeros, with nonzero mask
  tensor-equal to the stored mask.

## Parity and verification

The exact maxima before and after seam promotion are identical:

| Model / dtype | Vertex max before = after | Joint max before = after |
|---|---:|---:|
| SMPL float64 | `1.0412031070661243e-07` | `5.7804501407687781e-08` |
| SMPL float32 | `7.0780515670776367e-07` | `5.2899122238159180e-07` |
| SMPL-X float64 | `5.5905974738479358e-08` | `8.6539269084084935e-09` |
| SMPL-X float32 | `6.5565109252929688e-07` | `5.0663948059082031e-07` |

Final checks:

- SMPL/SMPL-X parity files: **84 passed**, 0 failed.
- Focused new-stage, I/O, and real-SMPL reduction gate: **36 passed**, 0
  failed, 9 expected float32-gradcheck warnings.
- Full suite: **184 passed, 4 skipped, 0 failed**, 15 warnings. The skips are
  existing unavailable-CUDA cases; warnings are expected float32 gradcheck
  notices plus two existing SciPy sparse deprecations.
- `ruff check .` and `git diff --check` pass.

Primary commands:

```bash
.venv/bin/pytest tests/core/test_pipeline.py tests/io/test_load.py \
  tests/io/test_schema.py \
  tests/bodies/test_smpl.py::test_generalized_skinning_reduces_exactly_on_smpl_fixture -q
.venv/bin/pytest tests/parity/test_smpl_parity.py \
  tests/parity/test_smplx_parity.py -q
.venv/bin/pytest tests/ -q
.venv/bin/ruff check .
git diff --check
```

## Deviations and findings

1. **The reference MHR MLP has no biases.** Both upstream linear layers are
   constructed with `bias=False`, despite the plan describing a biased first
   layer. The generic function accepts an optional first-layer bias for formula
   completeness; MHR will pass `None` in Order 03.
2. **Unicode archive fields cannot become dense PyTorch tensors.** Sixteen MHR
   metadata fields remain NumPy Unicode arrays; all numeric and Boolean fields
   become dense tensors without conversion. `docs/contracts.md` was narrowed
   accordingly.
3. **Lazy exclusion still validates the complete schema.** Presence alone was
   insufficient for the binding contract, while indexing LOD0 correctives
   would defeat `use_correctives=False`. The loader therefore reads only NPY
   headers for excluded fields and validates their shapes/dtypes with the full
   schema. A guarded test proves no skipped payload is indexed.
4. **No MHR-specific resolver wrapper was added.** The existing
   `resolve_model_path(family, filename, model_path)` already produces the
   exact required precedence when called with `("MHR",
   "converted/mhr_lod{n}.npz", model_path)`. A test pins that behavior; another
   wrapper would be duplicate code.
5. **Scaled LBS needs bind positions as well as scales.** The requested
   similarity transform cannot rotate/scale about the authored bind pivot from
   `T_skin` alone, so `lbs` accepts `bind_joint_positions` whenever
   `joint_scales` is supplied. Its scale-one correction is exactly zero and
   leaves legacy SMPL arithmetic unchanged.
6. **One derived-width sentinel was necessary.** `V*3` is handled as one
   explicit special case, parallel to the existing `(J-1)*9` sentinel; no
   general symbolic-dimension language or parser was introduced.

## BetterRobot carry-forwards

- BetterRobot SE(3) poses do not carry accumulated MHR scale. BetterHuman must
  retain the scale side channel introduced here and pass it into LBS.
- MHR skin-joint order is not the current SMPL assumption
  `joint_pose_world[..., 1:, :]`. Order 03 must bind an explicit native skin
  joint to BetterRobot joint mapping; `native_joint_indices` must not be
  repurposed because SMPL-X already uses it for reduced-model provenance.
- Converting the authoritative float32 4x4 FBX bind matrices to 7D poses may
  add small quaternion roundoff. Order 03/04 must measure this against the
  goldens rather than substituting recomputed zero-FK binds.

## LOC deltas

Counts are additions/deletions against `HEAD` for implementation and tests.
The `docs/contracts.md` row counts only this order's hunk; the pre-existing
measured-baselines append is excluded. Task-plan tracking files are also
excluded.

| File | Added | Deleted |
|---|---:|---:|
| `docs/contracts.md` | 3 | 2 |
| `src/better_human/bodies/smpl.py` | 2 | 3 |
| `src/better_human/bodies/smpl_family.py` | 85 | 3 |
| `src/better_human/bodies/smplx.py` | 2 | 3 |
| `src/better_human/core/body.py` | 17 | 58 |
| `src/better_human/core/pipeline.py` | 111 | 9 |
| `src/better_human/io/__init__.py` | 3 | 1 |
| `src/better_human/io/load.py` | 65 | 7 |
| `src/better_human/io/schema.py` | 90 | 1 |
| `tests/bodies/test_smpl.py` | 23 | 0 |
| `tests/core/test_pipeline.py` | 296 | 2 |
| `tests/io/conftest.py` | 81 | 0 |
| `tests/io/test_load.py` | 92 | 1 |
| `tests/io/test_schema.py` | 36 | 1 |
| **Total** | **906** | **91** |
