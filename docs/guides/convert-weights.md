# Convert weights

Some families are released in a form the loaders do not read: SMPL as a Chumpy pickle, MHR and SOMA as their
authors' asset folders. This guide turns each into the file [Load a family](load-a-family.md) expects. SMPL-H,
SMPL-X, MANO and FLAME are released as `.npz` files and need no conversion; put them in the layout of that guide.
Every converter runs from the repository root, reads the files you downloaded and writes under
`$BETTERHUMAN_MODELS_DIR`; nothing it writes is committed.

## SMPL from a Chumpy pickle

`tools/convert/smpl/convert_chumpy.py` reads a legacy pickle without installing Chumpy, and writes the `.npz` file
the SMPL loader reads; the loader checks the written file. `uv run` installs the script's own two dependencies:

```bash
mkdir -p "$BETTERHUMAN_MODELS_DIR/smpl"
uv run tools/convert/smpl/convert_chumpy.py \
  /path/to/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl \
  "$BETTERHUMAN_MODELS_DIR/smpl/SMPL_NEUTRAL.npz" \
  --model smpl
```

Run it once per gender, naming the output `SMPL_MALE.npz` or `SMPL_FEMALE.npz`.

## MHR from its release assets

`tools/convert/mhr/convert_mhr.py` turns the MHR release (kept in `$BETTERHUMAN_MODELS_DIR/MHR/assets/`) into one
archive per level of detail, `mhr_lod0.npz` to `mhr_lod6.npz`, in the format
[`better_human.mhr`](../reference/formats/mhr.md). It needs `pymomentum`, which the library never imports; the conda
environment of `tools/parity/mhr/environment.yml` has it:

```bash
conda env create --file tools/parity/mhr/environment.yml
conda run --name betterhuman-mhr-parity python tools/convert/mhr/convert_mhr.py \
  --assets-dir "$BETTERHUMAN_MODELS_DIR/MHR/assets" \
  --output-dir "$BETTERHUMAN_MODELS_DIR/MHR/converted"
```

`--lod 6` converts one level only, a quick check that the assets load. The upstream MHR repository states
Apache-2.0, but a downloaded asset folder may carry no `LICENSE` file: check the terms of the exact release before
you share its files.

## SOMA from SOMA-X v0.3.4

`tools/convert/soma/convert_soma.py` turns the SOMA-X release (Hugging Face `nvidia/soma-x`, tag v0.3.4, Apache-2.0;
kept in `$BETTERHUMAN_MODELS_DIR/SOMA/assets/`) into `$BETTERHUMAN_MODELS_DIR/SOMA/soma_mid.npz`, in the format
[`better_human.soma`](../reference/formats/soma.md). It needs the reference environment with `py-soma-x` 0.3.4,
pinned in `tools/parity/soma/requirements.txt`; `tools/convert/soma/README.md` lists what it reads and checks.

```bash
CUDA_VISIBLE_DEVICES="" /path/to/soma_env/.venv/bin/python tools/convert/soma/convert_soma.py \
  --assets-dir "$BETTERHUMAN_MODELS_DIR/SOMA/assets" --output-dir "$BETTERHUMAN_MODELS_DIR/SOMA"
```

## Check the result

Load the converted family once: `bh.load("smpl")`, `bh.load("mhr")` or `bh.load("soma")` reads the new file, and a
file with a wrong layout or format version raises naming the file and the fix.
