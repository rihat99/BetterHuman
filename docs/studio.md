# HumanStudio

HumanStudio is a browser tool for inspecting and authoring body-model data: pose joints with
in-viewport gizmos, sweep shape and density, paint vertex segmentations, place named contact frames,
and read out part masses. It runs on CPU.

It is a client of the BetterRobot viewer (`better_robot.viewer.Viewer`, y-up light theme). All scene
geometry is BetterRobot `Layer`s added with `viewer.add`: the BetterHuman mesh / frames layers
(`better_human.viz.human_layers`), a skeleton `SkeletonLayer`, and two studio layers in
`viz/studio/layers.py` (`SegmentationLayer`, `PartsLayer`). The *Timeline* folder in the sidebar
belongs to the viewer; the studio adds a tab group (Body, Display, Segment, Frames). Layer visibility has one control: the viewer's
**Layers** folder, one checkbox per layer (`human/mesh`, `human/skeleton`, `human/segmentation`,
`human/parts`, `human/edit`). Hidden segmentation/parts layers are rebuilt when shown, not on every edit.

## Launch

```bash
uv run python -m better_human.viz.studio --family smpl --gender neutral --port 8080
uv run python -m better_human.viz.studio --family mhr --lod 1
```

Flags: `--family` (smpl, smplh, smplx, mhr, ...), `--gender`, `--lod` (MHR), `--port`, `--host`
(default `127.0.0.1`), `--models-dir` (sets `BETTERHUMAN_MODELS_DIR`), `--data-dir` (default: the
repo's `studio_data`, whatever the cwd). From Python:

```python
from better_human.viz.studio import HumanStudio, launch
studio = HumanStudio(family="smplx", gender="neutral", port=8080)  # data_dir defaults to <repo>/studio_data
studio.run()            # or: launch(family="smpl")
```

`HumanStudio(..., server=...)` accepts an existing viser server (the tests pass a fake one).

## Feature tour

- **Body** tab: shape sliders per identity group (first 10 coefficients, range +-5) with *Reset* and
  *Random* per group; density control (SMPL / SMPL-H: a dropdown with the uniform launch density and
  the `body_densities.json` tables Naive, Chandler, Dempster, Clauser, Custom; other families: a
  number in kg/m^3); *Apply density* reloads the body keeping pose and frames; *Deselect joint*;
  *Reset pose*; pose *Save* (by name) and *Load* from a saved-pose dropdown that refreshes after each
  save; total-mass readout from the baked model.
- **Posing**: click a joint sphere (or the mesh, which selects the vertex's skinning joint) to attach
  a gizmo at the joint's world pose. Non-root joints get a rotation-only gizmo, the root a 6-DoF one;
  a joint without degrees of freedom reports so in the status line. The selected sphere is
  highlighted and the mesh is dimmed (35 % of the opacity slider value) until deselected.
- **Display** tab: mesh opacity, joint radius and the
  part *Explode* slider.
- **Segment** tab: pick or add a part, paint with a sphere brush (click the mesh), or drag a
  rectangle marquee; save / load named segmentations or the built-in `*_vert_segmentation.json`.
  *Build bodies* integrates each part's mass on the rest shape (`identity.v_shaped`) and fills a
  table. With the body's own segmentation it reuses the body's mass topology, so the total equals
  the total-mass readout; an edited segmentation closes parts with `build_part_topology`.
- **Frames** tab: in add mode a mesh click places a frame at the hit vertex; a click on an existing
  frame (pose mode) selects it and syncs the editor; rename, set the parent joint (or default
  skinning joint), delete, save / load, and *Export* a `contact_frames={...}` snippet for
  `bh.load(family, contact_frames=...)`.

## Data layout

```
<data_dir>/segmentations/<model_key>/<name>.json
<data_dir>/frames/<model_key>/<name>.json
<data_dir>/poses/<model_key>/<name>.json
```

`model_key` identifies the mesh topology (`persistence.model_key`), so files are shared between
genders of one family and refused for a different topology.

## Known limitations

- Browser interactions (real clicks, drags, marquee) are not tested; the tests drive the handlers
  with synthetic events on a fake server.
- Every pose or shape edit recomputes skinning on CPU; expect lag on SMPL-X and MHR.
- MHR exposes many identity sliders (only the first 10 per group are shown); MHR posing is untested.
- *Explode* moves rigid open part meshes, posed by their dominant joint, not closed part meshes.
- *Build bodies* uses one (mean) density for an edited segmentation; per-part table densities apply
  only to the body's own parts.
- Opacity and radius changes rebuild the whole layer group; the joint highlight writes the
  skeleton layer's colour array in place.
- Clicking a selected frame again does not clear its highlight (the old studio toggled it).
