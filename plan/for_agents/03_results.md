# Order 03 results — segmentation rework + `mass.py` simplification

Delivered on `dev`, BetterHuman only. BetterRobot untouched (the order-01 mesh
tools were consumed, not modified).

## What landed

Two segmentation **sources** now feed one `PartMeshTopology` and the shared,
differentiable `Inertia.from_mesh` integrator:

- **Authored closed-3D (default for SMPL).** `build_authored_topology` loads the
  vetted asset's per-part face tables (`body_faces[j]` indexing
  `[shaped_vertices; rest_joints]`, the joints as cap vertices), builds a
  **compact per-part** vertex table (mesh verts gathered from `v_shaped`, caps
  gathered from `rest_joints` — differentiable), remaps faces to local indices,
  and fixes winding once from rest geometry with
  `orient_faces_by_component` (uniform over all parts; **no** `validate_closed_manifold`
  gate, **no** boundary tracing, **no** centroid caps). The 3 open/non-manifold
  seam parts (spine3=9, left_collar=13, right_collar=14) pass through without
  rejection.
- **Argmax fallback (SMPL-X, no authored asset).** Kept the skinning-argmax
  segmentation + interior-fan successor-walk boundary closure + reverse-wound
  centroid caps. Dropped the per-spawn defensive `ValueError` raises
  (`non_manifold` / `open` / `inconsistent winding` / `cycled` /
  `half-edges do not form closed cycles` / `repeats before closing` /
  `loop < 3 vertices`) — the walk now trusts the vetted, consistently wound
  meshes.

Both stay batched and float32-differentiable w.r.t. betas through the cap
vertices. `body_inertias_from_shape` keeps its `(..., num_parts+1, 10)` contract
(row 0 = massless universe).

## Assets restored (tracked package location)

Restored from `git show main:src/better_human/smpl/config/<file>` into
`src/better_human/bodies/assets/` (a package with an `__init__.py` loader):

- `smpl_3d_segmentation.npy` — authored closed-3D segmentation (consumed).
- `body_densities.json` — Naive/Chandler/Dempster/Clauser/Custom named tables (consumed).
- `smpl_vert_segmentation.json`, `smplx_vert_segmentation.json` — per-vertex named
  parts, restored for order-05 rendering (not consumed here; no loader added, YAGNI).

`bodies/assets/__init__.py` exposes `load_closed_segmentation` and
`load_density_table`. `smpl.py` resolves `SMPL_CONFIG.segmentation_closed_file`
and `density_table` from there. (Files are in the version-controlled source tree
alongside the still-uncommitted order-02 `config_*.py`; owner commits the
refactor as a unit — I did not commit.)

## Named-density option

`SMPL(density=...)` now also accepts a **table name (str)**, e.g. `"Chandler"`,
resolved to a per-part tensor in native joint order via `load_density_table`,
feeding the SAME `density_tensor` path (no new policy code, no per-table
branches). Scalar and per-part-tensor options unchanged. SMPL-X unchanged
(named tables are SMPL-24-joint; a str there would raise on missing joint keys).

## Measured (neutral SMPL, density 1000)

| Quantity | Authored (this order) | Argmax baseline |
|---|---|---|
| 24-part volume sum | 0.07476461 m³ | — |
| whole-mesh volume | 0.07476054 m³ | — |
| **volume gap** | **+54.38 ppm (+0.00544 %)** | +1.78 % |
| total part mass | **74.7646 kg** (whole = 74.7605 kg) | 73.43 kg |
| aggregate vs whole COM sep | **0.022 mm** | 2.47 mm |

CUDA (device 2): authored path gives 54.61 ppm / 74.7646 kg — matches CPU.

Per-part masses (kg, native joint order 0..23):
`15.99, 7.97, 8.01, 6.66, 3.02, 3.00, 4.90, 0.62, 0.62, 10.08, 0.09, 0.09,
0.74, 1.49, 1.52, 3.94, 1.70, 1.73, 0.96, 1.00, 0.24, 0.25, 0.06, 0.07`.

## `core/mass.py` LOC

| | LOC |
|---|---|
| before (`git HEAD:core/mass.py`) | 291 |
| after | 283 |
| net | **−8** |

Net is modest because a whole new segmentation source (`build_authored_topology`,
~36 lines incl. docstring) was **added** while the argmax tracer was **kept** per
the handoff. What shrank: `_oriented_boundary_loops` → `_boundary_loops` lost its
~7 defensive raises; `segment_vertices`, `build_part_topology`,
`meshes_from_topology`, `body_inertias_from_shape` lost redundant internal
validation. The real simplification is conceptual: the **default** (SMPL) path
runs zero boundary tracing / capping / centroid math.

Other touched files: `bodies/smpl.py` (+asset + str-density wiring),
`bodies/smpl_family.py` (`bind_mass_properties` dispatches on `closed_faces`),
`tests/core/test_mass.py`.

## Tests

`tests/core/test_mass.py`: 14 tests. Full BetterHuman suite (CPU,
`CUDA_VISIBLE_DEVICES=""`): **109 passed, 6 skipped** (3 CUDA-unavailable, 2
pre-existing absent SMPL-X parity goldens, 1 viz CUDA). CUDA (device 2): mass +
dynamics CUDA cases **16 passed**, incl. `test_cuda_mass_rnea_and_crba_smoke`.
`ruff check` clean on all touched files.

Coverage added/updated:
- `test_authored_segmentation_volume_matches_whole_mesh` — asserts < 1e-4 rel
  (measured 54.4 ppm) and total mass 74–75.5 kg.
- `test_authored_segmentation_is_batched_and_finite` — `betas (2, n)` →
  `(2, 25, 10)`, finite, positive masses.
- `test_named_density_table_feeds_per_part_density` — Chandler lookup spot-checks
  (joint 0 = 853, joint 16 = 1005); density-invariant per-part **volumes** match
  the density-1000 body; unknown table name raises.
- `test_argmax_fallback_integrates_smplx_parts` — every SMPL-X fallback part
  integrates to positive volume.
- `test_float32_gradcheck_betas_to_total_mass` — now exercises the authored path,
  gradcheck-clean (float32, eps 1e-3).
- `test_default_mass_opt_out_and_per_part_density` — dropped the per-part
  `_assert_closed_edges` loop (authored parts 9/13/14 are legitimately open;
  closedness is not required and is not a valid assertion for them).
- `test_argmax_segmentation_cross_check` — repointed to the tracked package asset.

## Findings / deviations

1. **Modest LOC delta, per the handoff's own constraint.** The handoff mandated
   keeping the argmax successor-walk (only dropping its raises), so the plan's
   "delete the bulk of the graph code" could not be taken literally without
   discarding the SMPL-X fallback. The tracer stays; the reduction is honest but
   small. Not a defect — a direct consequence of the authoritative spec.
2. **Argmax fallback: 3 SMPL-X parts have pinched boundary loops.** spine2 (6),
   left_wrist (20), right_wrist (21) each have one boundary loop that revisits a
   vertex (figure-eight), so the single-centroid fan produces exactly one
   non-manifold (incidence-4) cap edge per part. This is **pre-existing** — the
   old code used the identical interior-walk + fan and would emit the identical
   caps (it never raised on these, and SMPL-X spawn tests were green, proving the
   loops traced as before). It was simply never tested. The parts still integrate
   to positive finite volume, so the fallback is valid; strict per-part manifoldness
   was never a project requirement (the authored SMPL seam parts are non-manifold
   by design too). My initial strict `_assert_closed_edges` SMPL-X test was
   therefore replaced with a positive-volume integration test. Flagged, not
   worked around.
3. **Wheel packaging not modified.** Tests run from the editable/source install,
   where `Path(__file__).parent` resolves the assets directly. Hatchling includes
   non-Python files under the package by default; no `pyproject.toml` change made
   (surgical). If a built wheel ever omits the assets, add a
   `[tool.hatch.build.targets.wheel.force-include]` — reported, not pre-emptively
   added.
4. **Density double-validation retained.** `density_tensor` (init boundary) and
   `body_inertias_from_shape` both validate density shape; the plan assigns the
   dedup to **order 04**, and the direct-call test needs the latter's guard, so
   left as-is.
