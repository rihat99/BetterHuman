# Order 02 results

Completed on `dev` on 2026-07-20 after Order 01.

## Delivered

- Declarative symbolic schemas over `V`, `J`, `S`, and `F` for SMPL and
  SMPL-X, plus the SMPL-H and MANO schemas required by the converter CLI.
- Required-subset validation with exact model dtypes, symbolic pose dimension
  `(J - 1) * 9`, actionable file/field/remedy errors, and debug logging for
  ignored vendor extras.
- Dense-or-CSR `J_regressor` storage. CSR validation covers component shapes,
  dtypes, matching index widths, offsets, ranges, and sorted unique columns;
  loading produces a sparse CSR tensor without densification.
- Explicit-path-first resolution with `$BETTERHUMAN_MODELS_DIR` fallback.
- A Chumpy-free converter CLI with NumPy/Python compatibility aliases,
  minimal `chumpy.ch.Ch` and `chumpy.reordering.Select` pickle stubs, CSR
  serialization, schema validation before writing, and overwrite protection.
- Direct `scipy>=1.11` dependency, because SciPy sparse types are read by the
  converter rather than being an incidental transitive dependency.

## Files exercised

All available local files required by the order were read without modification:

- `models/smpl/SMPL_FEMALE.npz`
- `models/smpl/SMPL_MALE.npz`
- `models/smpl/SMPL_NEUTRAL.npz`
- `models/smplx/SMPLX_FEMALE.npz`
- `models/smplx/SMPLX_MALE.npz`
- `models/smplx/SMPLX_NEUTRAL.npz`
- `models/smpl/models/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl`

The three SMPL NPZs contain float64 model arrays, 300 shape columns, int32
kinematic tables, and uint32 faces. The SMPL-X files match the reference table:
float32 template/correctives/regressor/weights/hands, float64 shapedirs with
400 columns, int64 kinematic tables, and uint32 faces. Their object dictionaries,
UV tables, and landmark tables were ignored without enabling pickle loading.

## Chumpy shim and conversion findings

- The neutral SMPL v1.1.0 pickle unpickled successfully on Python 3.13.11 and
  NumPy 2.5.1 without Chumpy installed.
- Every required dense array was byte-identical in C order after conversion.
- The original SciPy CSC regressor converted losslessly to canonical CSR
  components and reloaded as `torch.sparse_csr` with shape `(24, 6890)`.
- The exercised CLI output was schema-valid and measured 85,496,790 bytes.
- The historical pickle references `scipy.sparse.csc.csc_matrix`; SciPy emits
  its expected deprecation warning while unpickling. PyTorch also emits its
  upstream one-time notice that CSR tensor support is beta. Neither warning is
  a fallback or data change.

## Verification

- `uv sync`: passed; the lock remains at 76 resolved packages.
- `uv run python -m better_human.io.convert_chumpy ... --model smpl`: passed
  against the neutral v1.1.0 pickle.
- `uv run pytest tests/ -q -rs`: **20 passed, 0 skipped, 3 warnings in 15.03s**.
- `uv run ruff check src/ tests/`: all checks passed.
- Independent contract review found no remaining schema, loader, CSR, path, or
  converter blocker after malformed-input regressions were added.
- The combined `models/` and `references/` path/size/mtime manifest remained
  `1e12ec801fada88a15763f372770d3e065e4d1015a680a3f0e50917cb52e2258`.
- BetterRobot was not modified.

## Deviations and findings

- **Plan deviations:** none.
- SMPL kinematic tables accept the original uint32 dtype and the int32 dtype in
  the existing local converted NPZs; all other required SMPL/SMPL-X dtypes are
  exact.
- SMPL-H and MANO schemas were implemented in this order because the required
  CLI advertises those model choices and must validate its own output. No body
  math or later-order API was introduced.
