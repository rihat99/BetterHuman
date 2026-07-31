# HumanStudio

HumanStudio is an interactive [viser](https://viser.studio) app for inspecting
and authoring parametric-body data: pose joints with in-viewport gizmos, sweep
shape and density, paint vertex segmentations, place named frames, and read out
per-part and total mass. It talks to viser directly and never modifies
BetterRobot.

## Install and launch

viser ships in the optional `studio` extra:

```bash
uv sync --extra studio
```

Launch from the command line (weights resolve through the standard loader; pass
`--models-dir` or set `$BETTERHUMAN_MODELS_DIR`, otherwise the in-repo `models/`
directory is used):

```bash
uv run python -m better_human.viz.studio --model smpl --port 8080
uv run python -m better_human.viz.studio --model smplx --gender female
uv run python -m better_human.viz.studio --model mhr --lod 1
```

Then open the printed `http://localhost:<port>` URL. Equivalent programmatic
entry points:

```python
from better_human.viz.studio import HumanStudio, launch

launch(model="smpl", port=8080)          # construct and block
studio = HumanStudio(model="mhr", lod=1)  # construct without blocking
studio.run()
```

`better_human` itself never imports viser: the studio lives under
`better_human.viz.studio` and only pulls viser in when the app is constructed.

## Feature tour

The sidebar has four tabs and a status line that reports load time, mass, and
errors (load failures land here instead of stopping the server).

- **Body** — model / gender / LOD dropdowns and a *Load model* button; one
  folder per shape group with a slider per coefficient (range ±3, step 0.05)
  plus per-group *Reset* and *Random*; a density control (a
  `body_densities.json` table dropdown for SMPL, a number input otherwise) with
  *Apply density* to reload; the total-mass readout; and pose *Save* / *Load* /
  *Reset* with a saved-pose dropdown. *Deselect joint* detaches the gizmo.
- **Layers** — visibility checkboxes for the mesh, skeleton, frames, and parts
  layers; a *Color by segmentation* toggle; and sliders for mesh opacity,
  joint-sphere radius, frame radius, and parts explode. Defaults: mesh and
  skeleton on, frames and parts off.
- **Segment** — choose the active segmentation (argmax, a built-in
  `*_vert_segmentation.json`, or a saved set), then in edit mode paint vertices
  onto the active part with a sphere brush or rectangle marquee
  (assign/unassign). *Build bodies* closes the current part labels into solid
  meshes, feeds the parts layer, and prints a per-part and total mass table at
  the loaded body's density.
- **Frames** — choose a frame set (config default or saved), and in edit mode
  click the mesh to add a frame at the nearest vertex (auto-named `frame_N`).
  Rename, delete, save, or *Export as config*.

**Posing.** Click a red joint sphere to attach the single gizmo at that joint's
world pose. Non-root joints get a rotation-only gizmo; the root joint gets a
full 6-DOF gizmo. Dragging rewrites the configuration through the safe local
`q` update and the mesh re-skins. Clicking a fixed MHR joint (one with no
degree of freedom) reports that in the status line instead of showing a gizmo.

## Authored data (`studio_data/`)

Saves land under a `studio_data/` root (override with `--data-dir`), keyed by
mesh topology (`smpl_neutral`, `smplx_female`, `mhr_lod1`, ...):

```
studio_data/
  segmentations/<key>/<name>.json   # {part_name: [vertex_id, ...]}
  frames/<key>/<name>.json          # {frame_name: vertex_id}
  poses/<key>/<name>.json           # {"model_key": ..., "q": [floats]}
```

Segmentation files use the same schema as the shipped
`*_vert_segmentation.json` assets, so they interchange. A pose file records the
topology key and configuration width; loading checks both and refuses a
mismatch.

## Export to config

The Frames tab's *Export as config* button renders the active frame set as the
paste-ready `frame_names` / `frame_vertex_ids` tuples used by a
`BodyConfig` (`bodies/smpl_family/config_smpl.py` and friends). Copy the block
into a config to promote authored frames into a model's built-in landmarks.

## Known limitations

- **Browser interactions are not covered by tests.** The automated tests drive
  the session and app API headlessly; gizmo dragging, brush/marquee painting,
  and click-to-add-frame are exercised through their handlers but not through a
  real browser pointer.
- **Pose changes lag.** Every pose or shape edit recomputes blendshapes, FK,
  and skinning on CPU, so the mesh updates with a visible delay. This is
  expected and acceptable.
- **MHR has many shape sliders.** MHR exposes identity (45), expression (72),
  and proportions (73) coefficients, so its shape folders are large; they are
  collapsed by default.
- **Parts posing is a rigid approximation.** Each closed part is moved by its
  dominant native joint's skinning transform, which matches the skinned surface
  only where a part is rigidly weighted to one joint.
- **User-segmentation mass uses a single density.** *Build bodies* integrates
  every part at the loaded body's mean density, since per-part densities are not
  defined for arbitrary authored segmentations.
