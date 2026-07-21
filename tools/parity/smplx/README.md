# Regenerate SMPL-X parity goldens

Generate the licensed, local-only SMPL-X oracle grid with the pinned upstream
`smplx==0.1.28` environment.

1. Place the converted neutral weights at
   `models/smplx/SMPLX_NEUTRAL.npz` or choose another explicit path.
2. Materialize the locked oracle environment:

   ```bash
   uv sync --project tools/parity/smplx --locked
   ```

3. Dump the 29-case grid:

   ```bash
   uv run --project tools/parity/smplx python \
     tools/parity/smplx/dump_smplx.py \
     models/smplx/SMPLX_NEUTRAL.npz \
     tests/parity/goldens/smplx_neutral
   ```

4. Run `uv run pytest tests/parity/test_smplx_parity.py -q` from the repository
   root. The 29-case grid is committed under `tests/parity/goldens/smplx_neutral/`
   (like the SMPL grid), so the parity test runs by default; regenerate with the
   steps above only when the oracle or the model changes.
