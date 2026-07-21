# Order 05 results — docs + rendered segmentation comparison

Delivered on `dev`, BetterHuman only. BetterRobot untouched. No `src/` code
changed; this order adds a docs render helper, rendered images, an explanation
page, doc-truth fixes, and one packaging entry.

## Rendered segmentation images (the owner's ask)

`docs/render_segmentation.py` (headless matplotlib Agg, fixed camera
`elev=8, azim=-90`, colors from a seed-0 permutation of `gist_rainbow`) renders,
for each model × each available segmentation type, a part-colored surface and an
exploded body-split view. Re-runs are **byte-identical** (verified by md5 across
two runs). Weights resolve through the normal loader / `$BETTERHUMAN_MODELS_DIR`;
absent weights are skipped.

Eight PNGs under `docs/segmentation/`:

| File | Model | Type | View |
|---|---|---|---|
| `smpl_authored_surface.png` | SMPL | authored | part-colored surface (named per-vertex parts) |
| `smpl_authored_split.png` | SMPL | authored | exploded **closed** joint-capped solids |
| `smpl_argmax_surface.png` | SMPL | argmax | part-colored surface (strongest skinning joint) |
| `smpl_argmax_split.png` | SMPL | argmax | exploded centroid-capped solids |
| `smplx_authored_surface.png` | SMPL-X | authored | part-colored surface (named per-vertex parts) |
| `smplx_authored_split.png` | SMPL-X | authored | exploded **open** surface shells (no closed-3D asset) |
| `smplx_argmax_surface.png` | SMPL-X | argmax | part-colored surface (55 parts incl. per-finger) |
| `smplx_argmax_split.png` | SMPL-X | argmax | exploded centroid-capped solids |

Rendering sources per combination:
- SMPL authored surface / SMPL-X authored surface: `*_vert_segmentation.json`
  per-vertex named parts (24 for SMPL; 27 for SMPL-X, 6 verts unlabeled → gray).
- argmax surfaces: `structure.part_ids` (`segment_vertices`).
- SMPL authored split: `structure.part_topology` (the authored closed-3D topology
  SMPL loads by default) via `meshes_from_topology` — joint-capped solids.
- argmax splits: `build_part_topology(faces, part_ids, n)` → `meshes_from_topology`
  — centroid-capped solids.
- SMPL-X authored split: the intact mesh split into open per-part shells by the
  vertex labeling (SMPL-X ships no authored closed-3D asset).

## Explanation page

`docs/segmentation.md` (flat markdown, Diátaxis *explanation* style — no code
steps): the two segmentation sources → one closed `PartMeshTopology`; why the
authored joint-cap partition is volume-preserving (shared cap vertices telescope
to the whole-body volume; the 3 open seam parts — spine3, left_collar,
right_collar — are integrated about the part mean so the aggregate is exact);
when each is used (authored = SMPL-family default; argmax = generic fallback,
SMPL-X). Embeds all 8 images in per-model 2×2 tables and cites the measured
Baselines numbers (authored +0.0055 % / 74.76 kg / 0.02 mm vs argmax
+1.78 % / 73.43 kg / 2.47 mm). Closes with the regenerate command.

## Design-doc truth fixes

`docs/architecture.md`:
- Pluggability seam list: "sparse per-coordinate joint regressor" → "a dense
  joint regressor" (order-04 CSR removal).
- Concrete-model paragraph: added `BodyConfig` + per-model `config_*.py` data
  modules, `smpl_family.py` shared binding, SMPL topology re-homed out of
  BetterRobot, and the restored `body.landmark_positions(vertices)` accessor.
- "Mass and dynamics": rewritten from argmax-only to the **two-source** model
  (authored closed-3D default via `orient_faces_by_component`, replacing
  `trimesh.fix_normals`, tolerating the 3 open seam parts; argmax fallback with
  centroid caps). Replaced the stale numbers/claims — dropped the false "STAR
  segmentation … argmax fallback is used" sentence (the authored asset is now
  restored and used for SMPL) and the pre-order-03 10.19→41.50 ms figure; kept
  the `compute_mass=False` opt-out. Cites authored (74.76 kg, +0.0055 %, 0.02 mm)
  and argmax (73.43 kg, +1.78 %, 2.47 mm).
- "Package layout": added `config_body.py`, `config_smpl.py`, `config_smplx.py`,
  `smpl_family.py`, and `bodies/assets/`.

`docs/contracts.md`:
- Density override wording now includes "or a named density table resolved to
  that tensor" (order-03 str option).
- Added a normative bullet: two segmentation sources, one closed
  `PartMeshTopology` fed to `Inertia.from_mesh`; authored volume-preserving
  default vs argmax fallback; neither a per-query cost.
- (The order-04 "converter emits the regressor dense; loaders load every field
  as a dense tensor" line was already correct — left as-is.)

## Packaging fix

`pyproject.toml`: added `artifacts = ["*.npy", "*.json"]` under
`[tool.hatch.build.targets.wheel]`, and `matplotlib>=3.8` to the `dev`
dependency group (the render helper).

**Finding:** the assets were *already* shipping in the built wheel before this
change — hatchling with `packages = ["src/better_human"]` includes every file
under the package tree, not just `.py`, and the assets are not gitignored. So the
handoff's premise ("a built wheel currently would NOT include them") did not hold
on this tree. The `artifacts` entry is kept anyway: it makes the intent explicit
and pins inclusion even if a future VCS-ignore rule matches `*.npy`/`*.json`
(plausible in a repo that already gitignores `models/`). Verified via `uv build
--wheel`: all four data files (`smpl_3d_segmentation.npy`,
`smpl_vert_segmentation.json`, `smplx_vert_segmentation.json`,
`body_densities.json`) are present in the wheel both before and after.

## Tests / lint

- **CPU** (`CUDA_VISIBLE_DEVICES=""`): **105 passed, 6 skipped** — identical to
  the order-04 CPU baseline (no `src`/`tests` changes).
- **CUDA** (GPU 2, RTX 6000 Ada): **109 passed, 2 skipped** (2 skips = absent
  SMPL-X parity goldens) — identical to the order-04 CUDA baseline.
- `ruff check docs/render_segmentation.py` → clean (the only touched `.py`).
  `ruff check src tests` still clean.

## Deviations / findings (honest)

1. **Roadmap migration not done.** The plan's `### 05` mentions "migrate the old
   plan's roadmap into docs," but the authoritative order-05 handoff and the
   stated definition of done enumerate only images + `segmentation.md` +
   `architecture`/`contracts` truth + packaging. The roadmap already lives in
   `plan/README.md` "Beyond this plan"; I did not duplicate it into `docs/` to
   avoid a divergent second copy and to stay within the handoff scope. Trivial
   follow-up if the owner wants a `docs/roadmap.md`.
2. **Packaging premise did not hold** (see Packaging fix) — assets already
   shipped; the added config is a robustness pin, not a repair.
3. **Render helper lives in `docs/`** (`docs/render_segmentation.py`), not
   `examples/`, since its output is `docs/segmentation/` and it is a docs
   generator rather than an interactive example. `matplotlib` is a `dev`-group
   dependency (not a runtime/library dependency).
4. **Cross-image colors do not correspond** for SMPL authored (surface colors by
   named per-vertex part; split colors by native joint index). Each image is
   independently a valid "distinct color per part" view; no correspondence is
   claimed in the page. SMPL-X authored surface/split *do* share the vertex
   labeling and its colors.
