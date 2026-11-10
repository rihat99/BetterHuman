# Regenerate the SOMA parity goldens

`dump_soma.py` runs upstream SOMA-X (`SOMALayer`, `py-soma-x` 0.3.4, mid LOD, float32, metres) on 14 fixed cases and
writes one archive per case; the module docstring lists the arrays. Like the other goldens computed from body-model
weights, they live in the models directory, not in git (decision record 0015):
`$BETTERHUMAN_MODELS_DIR/goldens/soma_mid/`. The release tier runs `tests/parity/soma/` against them. Regenerate
them only when the oracle or the assets change, into a new directory `$OUT`, and compare every array with the current
files before copying them over: a regenerated number can hide a regression.

It needs the reference environment pinned in `tools/parity/soma/requirements.txt` (`py-soma-x` 0.3.4 with `soma`,
`pxr`, `warp`, `torch`; the converter, `tools/convert/soma/README.md`, uses the same one) and the SOMA-X release
assets (Hugging Face `nvidia/soma-x`, tag v0.3.4, commit `56a3d957a549c66425b500fddf7f3183bc6fbdff`), kept in
`$BETTERHUMAN_MODELS_DIR/SOMA/assets/` with the release's `SOMA-X-HF-MANIFEST.json` and `LICENSE`. Create the
environment outside the repository, then run from the repository root; warp enumerates GPUs unless CUDA is hidden:

```bash
uv venv --python 3.12 /path/to/soma_env/.venv
uv pip sync --python /path/to/soma_env/.venv/bin/python tools/parity/soma/requirements.txt
CUDA_VISIBLE_DEVICES="" /path/to/soma_env/.venv/bin/python tools/parity/soma/dump_soma.py \
    --assets-dir "$BETTERHUMAN_MODELS_DIR/SOMA/assets" --output-dir "$OUT/soma_mid"
uv run pytest tests/parity/soma -q -m licensed   # once the new files are in $BETTERHUMAN_MODELS_DIR/goldens/soma_mid
```

Every archive carries `provenance` (decision record 0014), a JSON string with `oracle` (the installed `py-soma-x`
version), `generator` (the dump script's repository path), `asset_sha256`, `dtype` (`float32`) and `seed` (the case
seed). `asset_sha256` is the sha256 of the `sha256sum` listing of the four asset files `SOMALayer` reads:

```bash
cd "$BETTERHUMAN_MODELS_DIR/SOMA/assets" && sha256sum SOMA_neutral.npz SOMA_procedural_transforms.json \
    SOMA_template_rig.usda correctives_model.pt | sha256sum
```

A golden written by the dump script hashes the files that dump read; a golden stamped after its dump hashes the
files present when it was stamped (decision record 0025, proposed).
