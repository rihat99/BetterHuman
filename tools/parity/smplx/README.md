# Regenerate the SMPL and SMPL-X parity goldens

The goldens are outputs of licensed weights, so they live in the models directory, not in git (decision record
0015): `$BETTERHUMAN_MODELS_DIR/goldens/smpl_neutral/` (13 cases) and `$BETTERHUMAN_MODELS_DIR/goldens/smplx_neutral/`
(29 cases). The release tier runs `tests/parity/smplx/` against them. Regenerate them only when the oracle or the
weights change, into a new directory `$OUT`, and compare every array with the current files before copying them over:
a regenerated number can hide a regression.

1. Materialize the pinned oracle environment (`smplx==0.1.28`, locked in `uv.lock`):

   ```bash
   uv sync --project tools/parity/smplx --locked
   ```

2. Dump both grids from the converted neutral weights:

   ```bash
   uv run --project tools/parity/smplx python tools/parity/smplx/dump_smpl.py \
     "$BETTERHUMAN_MODELS_DIR/smpl/SMPL_NEUTRAL.npz" "$OUT/smpl_neutral"
   uv run --project tools/parity/smplx python tools/parity/smplx/dump_smplx.py \
     "$BETTERHUMAN_MODELS_DIR/smplx/SMPLX_NEUTRAL.npz" "$OUT/smplx_neutral"
   ```

3. Once the new files are accepted and copied into `$BETTERHUMAN_MODELS_DIR/goldens/`, run the gates from the
   repository root: `uv run pytest tests/parity/smplx -q -m licensed`.

Every archive carries `provenance` (decision record 0014), a JSON string with `oracle` (the installed `smplx`
version), `generator` (the dump script's repository path), `asset_sha256`, `dtype` (`float64`, the oracle's precision)
and `seed` (the grid seed). `asset_sha256` is the sha256 of the `sha256sum` listing of the weights file, for example
SMPL-X:

```bash
cd "$BETTERHUMAN_MODELS_DIR/smplx" && sha256sum SMPLX_NEUTRAL.npz | sha256sum
```
