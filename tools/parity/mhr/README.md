# Regenerate MHR converted weights and parity goldens

Use this pinned, CPU-only tools environment to convert the local MHR release
assets and run the upstream `pymomentum` oracle. BetterHuman never imports
`pymomentum` at runtime. Pixi is not installed on the target box, so this uses
the documented conda fallback. `trimesh` is the one pip dependency because the
requested 4.9.0 release was not available from conda-forge when verified.

1. Create the environment from the repository root:

   ```bash
   conda env create --file tools/parity/mhr/environment.yml
   ```

2. Verify that the smallest rig loads:

   ```bash
   conda run --name betterhuman-mhr-parity python \
     tools/convert/mhr/convert_mhr.py \
     --assets-dir "$BETTERHUMAN_MODELS_DIR/MHR/assets" \
     --output-dir "$BETTERHUMAN_MODELS_DIR/MHR/converted" \
     --lod 6
   ```

3. Convert all seven LODs:

   ```bash
   conda run --name betterhuman-mhr-parity python \
     tools/convert/mhr/convert_mhr.py \
     --assets-dir "$BETTERHUMAN_MODELS_DIR/MHR/assets" \
     --output-dir "$BETTERHUMAN_MODELS_DIR/MHR/converted"
   ```

4. Dump the deterministic oracle grid (`--reference-dir`, default
   `references/parametric_body/MHR`, is the read-only checkout of the reference
   `mhr` package):

   ```bash
   conda run --name betterhuman-mhr-parity python \
     tools/parity/mhr/dump_mhr.py \
     --assets-dir "$BETTERHUMAN_MODELS_DIR/MHR/assets" \
     --converted-dir "$BETTERHUMAN_MODELS_DIR/MHR/converted" \
     --output-dir "$OUT"
   ```

5. Once the new files in `$OUT/mhr_lod{0..6}/` are accepted and copied into
   `$BETTERHUMAN_MODELS_DIR/goldens/`, run the gates from the repository root:
   `uv run pytest tests/parity/mhr -q -m licensed`.

The source assets remain unchanged. Converted archives and the float32 golden
grid are outputs of the licensed assets, so they live in the models directory,
not in git (decision record 0015): `$BETTERHUMAN_MODELS_DIR/goldens/mhr_lod{0..6}/`
holds 48 archives, 30 LOD1 cases and three cases for each other LOD. Do not
regenerate it for a BetterHuman implementation change. Regenerate only when the
upstream MHR release, converter semantics, or intentionally reviewed oracle grid
changes, into a new directory `$OUT`; inspect every binary delta and remeasure and
document parity before accepting the refresh.

Every archive carries `provenance` (decision record 0014), a JSON string with
`oracle` (the installed `pymomentum` version), `generator` (the dump script's
repository path), `asset_sha256`, `dtype` (`float32`) and `seed` (the case seed).
`asset_sha256` is the sha256 of the `sha256sum` listing of the four files the
oracle reads for the LOD, for example LOD1:

```bash
cd "$BETTERHUMAN_MODELS_DIR/MHR/assets" && sha256sum compact_v6_1.model \
  corrective_activation.npz corrective_blendshapes_lod1.npz lod1.fbx | sha256sum
```

For the goldens stamped on 2026-11-06, `asset_sha256` hashes the asset files
present at stamp time, not at dump time (decision record 0014, note); a golden
written by the dump script hashes the files that dump read.
