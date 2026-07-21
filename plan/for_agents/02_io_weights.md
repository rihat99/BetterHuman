# Order 02 — Weight-file io: schemas, validated loader, chumpy converter

> **Agent log — 2026-07-20:** Completed on `dev`; all 6 local SMPL/SMPL-X
> NPZs and the neutral SMPL v1.1.0 pickle passed, with 20 tests and no skips.
> Findings are limited to upstream CSR-beta and legacy SciPy-pickle warnings;
> no plan deviations. See [`02_results.md`](02_results.md).

Read first: `docs/design_principles.md`, `docs/architecture.md` (§Weights),
`docs/contracts.md` (§Weight files), `references/design/smplx.md` §4 (field
tables — the schema source of truth), `references/design/SOMA-X.md` §5.4
(pickle-stub shim), `plan/README.md` ground rules. Never modify `models/`
— read-only test input.

## Context

Original SMPL-family weights come as chumpy-era `.pkl` (SMPL, MANO, FLAME)
or `.npz` (SMPL-X, SMPL-H from MPI). BetterHuman consumes exactly one
format: a clean, self-describing `.npz` per model with a validated schema.
This order builds the io layer and the converter; no body math yet.

## Deliverables

1. **`src/better_human/io/schema.py`** — declarative per-model field
   schemas (name → expected shape pattern/dtype) for SMPL and SMPL-X,
   derived from the verified field tables in `references/design/smplx.md`
   §4 (e.g. `posedirs` last dim `= (J−1)·9`; SMPL-X `shapedirs` packs
   300 shape + 100 expression columns). Shape patterns are symbolic over
   `V`/`J` so variants validate without magic numbers scattered in code.
   `validate(dict, schema)` raises one actionable error naming field,
   expectation, and file. Validation is **required-subset semantics**
   (`docs/contracts.md` §Weight files): schema fields must be present and
   correct; unknown extras are ignored and logged at debug — the raw MPI
   SMPL-X `.npz` carries object-dict fields (`joint2num`, `part2num`), UV
   arrays (`ft`, `vt`), and landmark tables that must not block loading.
2. **`src/better_human/io/load.py`** —
   - `resolve_model_path(family, filename, model_path=None)`: explicit arg
     wins; else `$BETTERHUMAN_MODELS_DIR/<family>/<filename>`; else raise
     with both remedies spelled out. Document the conventional layout in
     the module docstring (it matches the existing local `models/` tree —
     inspect it).
   - `load_model_npz(path, schema) -> dict[str, torch.Tensor/np.ndarray]`
     with up-front validation; sparse regressors stay sparse (store CSR
     components in the npz; reconstruct `torch.sparse_csr_tensor` or keep
     index/value tensors — pick one, document it). No dtype surprises:
     arrays load as stored; conversion to the body's dtype happens at body
     construction, not here.
3. **`src/better_human/io/convert_chumpy.py`** — CLI
   (`uv run python -m better_human.io.convert_chumpy <in.pkl> <out.npz>
   --model smpl|smplh|mano`) converting original `.pkl`s to our `.npz`:
   - pickle-stub shim: install fake `chumpy` (+ the NumPy-2-removed
     aliases) into `sys.modules` at unpickle time so arrays materialize
     without chumpy installed — port the ~80-line pattern from
     `references/parametric_body/SOMA-X` (`_smpl_family_loader.py:18-95`);
     reimplement, don't copy verbatim (different license — Apache, fine,
     but our code style applies).
   - scipy-sparse `J_regressor` → CSR components; validate output against
     the schema before writing; refuse to overwrite without `--force`.
   - If the shim fails on a real local file, stop and report — the pinned
     museum-NumPy fallback venv is a documented follow-up, not this order.
4. **Tests** (`tests/io/`):
   - schema validation: correct file passes; mutated dict (missing field,
     wrong shape) fails with the actionable message.
   - path resolution: arg precedence, env-var fallback (monkeypatched),
     error path.
   - against real local weights (skip with explicit reason when absent):
     load `models/smplx/*.npz` and `models/smpl/*` files, validate; run
     the converter on an original SMPL `.pkl` if present and byte-compare
     array contents against loading the same pkl via the shim directly.

## Acceptance

- Real local SMPL and SMPL-X weights load and validate on this host (do
  not commit any weight data).
- Converter produces schema-valid `.npz` from a real chumpy `.pkl` on this
  host; results file records which files were exercised.
- Full gate `uv run pytest tests/ -q` green; ruff clean; skip reasons
  visible in `-rs` output.

## Results file

`plan/for_agents/02_results.md`: schemas implemented, files exercised,
shim findings (exact pkl vintages that worked), deviations.
