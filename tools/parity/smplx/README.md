# Regenerate the SMPL, SMPL+H and SMPL-X parity goldens

The goldens are outputs of licensed weights, so they live in the models directory, not in git (decision record
0015): `$BETTERHUMAN_MODELS_DIR/goldens/smpl_neutral/` (13 cases) and `$BETTERHUMAN_MODELS_DIR/goldens/smplx_neutral/`
(29 cases). The release tier runs `tests/parity/smplx/` against them. Regenerate them only when the oracle or the
weights change, into a new directory `$OUT`, and compare every array with the current files before copying them over:
a regenerated number can hide a regression. A re-dump of the licensed goldens is not bit for bit: the inputs and
provenance come back identical, but vertices and joints can differ from the stored goldens in the last few bits
(up to 6 ulp, 7.8e-16 m), as another CPU or BLAS path computes them. Compare with a tolerance far below the gates,
not for equality.

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

A golden written by the dump script hashes the files that dump read; a golden stamped after its dump hashes the
files present when it was stamped (decision record 0025, proposed).

## Random-weight archives and goldens

The full tier runs the same parity on random weights (decision record 0016): one tracked archive per family,
`tests/support/random_weights/{smpl,smplh,smplx}_random.npz`, in the family's upstream layout at its native vertex
count, and the oracle's outputs on it in `tests/parity/smplx/random/{smpl,smplh,smplx}/`, which
`tests/parity/smplx/test_random_weights.py` gates at the licensed tolerance. `make_random_archive.py` draws every
array from a fixed seed (no number comes from licensed weights) and writes the same bytes on every run. Regenerate
the archives only when their layout changes, then the goldens from them, and review both as gate data:

```bash
for family in smpl smplh smplx; do
  uv run --project tools/parity/smplx python tools/parity/smplx/make_random_archive.py $family \
    tests/support/random_weights/${family}_random.npz
  uv run --project tools/parity/smplx python tools/parity/smplx/dump_$family.py \
    tests/support/random_weights/${family}_random.npz tests/parity/smplx/random/$family
done
```

A random-weight golden's `asset_sha256` is the hash of its archive, which `tests/contract/test_golden_provenance.py`
checks in every tier.
