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
     --assets-dir models/MHR/assets \
     --output-dir models/MHR/converted \
     --lod 6
   ```

3. Convert all seven LODs:

   ```bash
   conda run --name betterhuman-mhr-parity python \
     tools/convert/mhr/convert_mhr.py \
     --assets-dir models/MHR/assets \
     --output-dir models/MHR/converted
   ```

4. Dump the deterministic oracle grid:

   ```bash
   conda run --name betterhuman-mhr-parity python \
     tools/parity/mhr/dump_mhr.py \
     --assets-dir models/MHR/assets \
     --output-dir tests/parity/goldens
   ```

The source assets remain unchanged. Converted archives are local-only under
the gitignored `models/MHR/converted/` directory. The float32 golden grid under
`tests/parity/goldens/mhr_lod{0..6}/` is committed test data: 48 archives,
comprising 30 LOD1 cases and three cases for each other LOD. Do not regenerate
it for a BetterHuman implementation change. Regenerate only when the upstream
MHR release, converter semantics, or intentionally reviewed oracle grid
changes; inspect every binary delta and remeasure and document parity before
accepting the refresh.
