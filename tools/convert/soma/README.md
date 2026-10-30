# SOMA-X converter

`convert_soma.py` turns the SOMA-X release assets (Hugging Face `nvidia/soma-x`, tag v0.3.4, Apache 2.0)
into one self-contained archive, `models/SOMA/soma_mid.npz` (mid LOD, 18056 vertices, metres, y-up, z-forward).

It reads `SOMA_neutral.npz` (PCA shape space, triangles), `SOMA_template_rig.usda` (110-joint skin rig,
bind mesh, skin weights), `SOMA_procedural_transforms.json` (32 twist helpers on 8 limb segments) and
`correctives_model.pt` (pose-corrective MLP). It builds upstream `SOMALayer` once to reuse its exact joint
regressor, folded public skin weights, public joint order and corrective masks. If
`SOMA-X-HF-MANIFEST.json` is present, the file hashes are checked against it.

It needs the reference environment with `py-soma-x` 0.3.4 (`soma`, `pxr`, `warp`, `torch`); the
BetterHuman runtime never imports these. Run from the repository root:

```bash
CUDA_VISIBLE_DEVICES="" /path/to/soma_env/.venv/bin/python tools/convert/soma/convert_soma.py \
    --assets-dir /path/to/soma_assets --output-dir models/SOMA
```

Output: `models/SOMA/soma_mid.npz` (about 86 MiB, not committed). Its key schema is the contract with the SOMA
loader; the converter logs its checks (faces, bind vs T-pose, twist parents, regressor error) while it runs.
