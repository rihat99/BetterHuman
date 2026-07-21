# Order 04 results — over-engineering cleanup (deletion order)

Delivered on `dev`, BetterHuman only. BetterRobot untouched. No numerics
changed: the shipped joint regressors are already dense, so removing the CSR
path cannot alter FK/parity — proven byte-for-byte below.

## Numerics unchanged (the prime directive)

Deterministic SMPL float64 parity probe over all 11 goldens, run before and
after every cut (identical to all printed digits):

| Quantity | Before | After |
|---|---|---|
| `J_regressor.layout` | `torch.strided` (dense) | `torch.strided` (dense) |
| worst vertex error vs `smplx` oracle | `1.041203e-07` | `1.041203e-07` |
| worst joint error vs `smplx` oracle | `5.780450e-08` | `5.780450e-08` |
| `rest_joints.sum()` (f64 checksum) | `7.103380377229e-01` | `7.103380377229e-01` |

The shipped SMPL asset already loads dense (`torch.strided`), confirming the CSR
apparatus was dead weight for every shipped weight file.

## Test gate (unchanged tolerances)

- **CPU** (`CUDA_VISIBLE_DEVICES=""`): baseline `109 passed, 6 skipped` →
  after `105 passed, 6 skipped`. The 4 fewer are exactly the 4 removed
  CSR-only load tests (see below); a dense-load coverage test was added in
  their place, so the real safety net did not shrink.
- **CUDA** (GPU 1, RTX 6000 Ada): `109 passed, 2 skipped` (the 2 skips are the
  absent SMPL-X parity goldens; all CUDA-conditional cases pass).
- Converter round-trip (`test_real_smpl_pickle_converts_losslessly_with_stub`)
  passes with the now-**dense** regressor: the converted `J_regressor` is
  byte-identical to `raw["J_regressor"].todense()`, and loads back
  `torch.strided`.
- `ruff check src tests` → **All checks passed!**
- No parity / round-trip / float32-gradcheck / mass / dynamics test was
  weakened; tolerances are untouched.

## LOC deltas (measured `wc -l`, working tree at session start vs now)

Net **`src` −363 LOC** (2932 → 2569). Baseline was the post-order-03 working
tree, so these deltas isolate order-04's cuts. (`git diff --stat` vs `HEAD` is
misleading here — `HEAD` predates the uncommitted orders 01–03.)

| File | Before | After | Δ | Cut |
|---|---|---|---|---|
| `io/schema.py` | 448 | 201 | **−247** | CSR + symbolic-DSL collapse + SMPL-H/MANO |
| `core/reduction.py` | 194 | 163 | −31 | CSR sparse-COO slice branch |
| `io/load.py` | 115 | 86 | −29 | `_load_sparse_regressor` + CSR archive keys |
| `core/mass.py` | 283 | 264 | −19 | density dedup + `part_meshes` wrapper |
| `io/convert_chumpy.py` | 189 | 174 | −15 | dense emit + `schema_for_model` |
| `core/pipeline.py` | 107 | 96 | −11 | CSR `sparse.mm` branch + `so3` hoist |
| `core/__init__.py` | 14 | 9 | −5 | drop mass re-exports |
| `core/body.py` | 216 | 214 | −2 | drop dead `num_faces` field |
| `io/__init__.py` | 14 | 12 | −2 | drop SMPL-H/MANO exports |
| `bodies/smpl.py` | 186 | 185 | −1 | drop `num_faces=` kwarg |
| `bodies/smplx.py` | 391 | 390 | −1 | drop `num_faces=` kwarg |
| **total src** | **2932** | **2569** | **−363** | |

## The six cuts (all executed)

1. **CSR joint-regressor apparatus deleted (5 files).** Converter emits the
   regressor dense (`value.todense()`); `schema.py` lost `CSR_DATA/INDICES/
   INDPTR/SHAPE/KEYS`, `dense_or_csr`, `_validate_csr`, `_expect_component`, and
   every CSR branch in `validate()`; `load.py` lost `_load_sparse_regressor` and
   the CSR archive-key handling (all fields load dense); `reduction.py`'s
   `_slice_regressor_rows` collapsed to an inline
   `joint_regressor.index_select(0, survivor_tensor)`; `pipeline.py`'s
   `regress_joints` keeps only the dense `einsum`. `contracts.md` "loaders never
   densify" line updated to "converter emits the regressor dense; loaders load
   every field as a dense tensor".
2. **Symbolic-dimension DSL collapsed.** Deleted the `Dimension`
   class (`scale`/`offset`/`describe`), `minimum_dimensions`, the resolved-shape
   error machinery, and the pretty-printer. Fields are now `(shape, dtype)` with
   `int | str` axes bound on first sighting and **one** inline special case,
   `POSE_FEATURES → (J-1)*9`. Shape+dtype checking and actionable
   field/file/fix errors preserved (schema tests still green).
3. **Validation de-duplicated (density).** `body_inertias_from_shape` now trusts
   the density already validated by `bodies/smpl_family.py:density_tensor`
   (single `density.broadcast_to((num_parts,))`; dropped the second
   isinstance/shape/Real re-validation and the unused `from numbers import
   Real`). The wrong-shape density error still fires — from `density_tensor` at
   the `SMPL.__init__` boundary (test green).
4. **SMPL-H / MANO converter schemas dropped.** Removed `SMPLH_SCHEMA`,
   `MANO_SCHEMA`, the `SCHEMAS` dict and `schema_for_model`; the converter uses
   `SMPL_SCHEMA` directly and `--model` choices narrow to `("smpl",)`.
   `io/__init__` exports narrowed accordingly.
5. **`core` surface narrowed + test-only wrapper dropped.** `core/__init__` no
   longer re-exports the mass symbols (real importers use `core.mass` directly);
   deleted `part_meshes` and inlined its two calls
   (`meshes_from_topology(v, build_part_topology(f, ids, n))`) in its only
   caller, `test_mass.py`.
6. **Minor.** Dead rank guard gone (the new `_bind_axes` rank-checks in pass 1,
   so no guard is needed); `so3` import hoisted to module scope in `pipeline.py`;
   dead `BodyStructure.num_faces` dropped (written in `smpl.py`/`smplx.py`,
   never read). `load()` registry left as-is (owner-blessed).

## Test edits (safety net only grows)

- `test_load.py`: removed the 4 CSR-only tests (`test_sparse_regressor_loads…`,
  `test_noncanonical_csr…` ×2 params, `test_mixed_csr_index_dtypes…`,
  `test_nonnumeric_csr_shape…`) and the `_write_sparse_fixture` helper; added
  `test_dense_regressor_loads_without_dtype_conversion` and kept the
  object-field load-error test on a dense fixture.
- `test_real_weights.py`: converter round-trip now asserts a **dense**
  byte-identical regressor and `torch.strided` on load (was CSR).
- `test_reduction.py`, `test_pipeline.py`: dropped the CSR-layout assertions;
  the dense weight-transfer / renormalization / einsum-parity assertions stay.
- `test_schema.py`: the multi-wrong-rank test de-CSR'd (breaks `weights` and
  `kintree_table` ranks with a valid dense `J_regressor`); still asserts a clean
  field-named rank error with no unbound-dimension leak.
- `test_mass.py`: `part_meshes` call inlined.

## Deviations / findings (honest)

- **`reduce_joint_data` validation kept (finding, not a miss).** Cut #3's second
  bullet listed `segment_vertices`, `build_part_topology`, `reduce_joint_data`
  as helpers to trim. In the current tree: `segment_vertices` has no validation
  to remove; `build_part_topology` has only a single minimal `if not surface`
  guard; `reduce_joint_data`'s extensive input validation is a **tested public
  contract** (`test_reduction_rejects_malformed_joint_data` asserts its
  `joint_regressor` / `skinning_weights` / `posedirs` / vertex-count /
  positive-finite-sums messages) and it is a reusable generic helper exercised
  standalone. Removing it would delete a passing test — forbidden by "the safety
  net only grows". So only the clearly-redundant density double-validation was
  removed. Net −363 already meets the ~300 target without weakening a test.
- **Cross-order files touched, per handoff.** The density dedup edits `mass.py`
  (order 03's file) and `num_faces` removal edits `body.py` + `bodies/smpl.py` +
  `bodies/smplx.py` (order 02's files). Both are explicitly authorized by the
  order-04 handoff (cut #3 "order 03 left this dedup to you"; cut #6 num_faces
  "drop it if it has no consumer" — it has none). Orders 01–03 have already
  landed in the working tree, so this is not a live coordination conflict.
