# Render the segmentation images

Use this guide after an intentional change to segmentation source data,
topology construction, or the documentation renderer. Ordinary body-model
implementation changes should not rewrite the images.

From the repository root, make the converted model weights available through
the standard directory layout and run:

```bash
uv run python docs/render_segmentation.py \
  --models-dir "$BETTERHUMAN_MODELS_DIR"
```

The script uses headless matplotlib, fixed colors, and a fixed camera. It
regenerates each available SMPL, SMPL-X, and MHR image under
`docs/segmentation/` and skips models whose local weights are absent. Inspect
every changed PNG; an unchanged input and environment should produce the same
bytes.
