# Order 01 results — BetterRobot mesh face-orientation + closedness tools

Completed on `dev` (BetterRobot working tree) on 2026-07-21. This order is
authorized to edit BetterRobot mesh tools only; no BetterHuman source changed.
Includes the adversarial-review follow-up (non-orientable bug fix + a non-strict
best-effort orienter for the real authored segmentation).

## Delivered

A torch-only, GPU-safe companion to `Inertia.from_mesh` in a new module
`src/better_robot/spatial/mesh.py` (282 LOC). Public API:

- `validate_closed_manifold(faces) -> None` — raises unless every undirected
  edge has incidence exactly 2. Boundary edges (incidence 1 → open surface) and
  non-manifold edges (incidence > 2) raise `ValueError` naming up to five
  offending edges (non-manifold edges also report their incidence). Multiple
  disjoint closed components are accepted. Edge-incidence uses `torch.unique` on
  the input device (CPU or CUDA).
- `orient_faces_consistently(faces) -> torch.Tensor` — **strict** path.
  Validates closed-manifoldness, then walks the face-adjacency graph (edges
  shared by exactly two faces) reversing any neighbour that would traverse the
  shared edge in the same direction, so every face in a connected component
  agrees on winding. Integer face table only (never vertex values); returns a
  table on the input device/dtype. The single global winding sign is folded by
  `Inertia.from_mesh`. **Now raises `ValueError("surface is non-orientable…")`**
  when the walk meets a contradiction (see bug fix below).
- `orient_faces_by_component(vertices, faces) -> torch.Tensor` — **non-strict,
  best-effort** path (new, from the follow-up). Builds adjacency over
  incidence-2 edges only (boundary and non-manifold edges are ignored, never
  raised on), orients each connected component consistently, then flips each
  component whose **mean-centered signed tetrahedron volume** is negative so every
  component faces outward. Centering matches `Inertia.from_mesh` exactly (same
  `vertices.mean`), so the subsequent integration needs no global sign fold and
  component volumes add constructively. Integer table out → the orientation is a
  static rest-geometry precompute and downstream integration stays
  differentiable. On a clean closed manifold it produces the same integrated
  result as the strict orienter.

Export edit `src/better_robot/spatial/__init__.py` (+13 / −1): re-exports the
three functions and adds them to `__all__`.

## BetterRobot changes (explicit list)

1. Added `src/better_robot/spatial/mesh.py` (282 LOC) — three public functions +
   private helpers `_check_face_table`, `_undirected_edges`,
   `_build_face_adjacency`, `_orient_components`, `_apply_reversal`.
2. Edited `src/better_robot/spatial/__init__.py` (three new public exports).
3. Added `tests/spatial/test_mesh.py` (323 LOC).

Nothing else in BetterRobot was touched. No `models/` or `references/` change.

## Bug fix (adversarial review item 1)

The strict orienter previously assumed a validated closed 2-manifold is
orientable and its DFS "never conflicts" — **false**. The minimal RP²
triangulation (antipodal quotient of the icosahedron: 10 faces / 6 verts, every
edge incidence-2) passes `validate_closed_manifold` yet is non-orientable, and
the old DFS silently returned a still-inconsistent table. Fixed: the walk now
checks already-signed neighbours and raises
`ValueError: surface is non-orientable: faces 8 and 5 impose contradictory
winding across their shared edge`. The false inline comment was corrected. Added
a test (`test_non_orientable_surface_is_rejected_by_the_strict_orienter`)
asserting RP² validates as closed but is rejected by the strict orienter.

## Measured numbers

**Strict mixed-winding parity (box, faces {0,3,5,8,11} reversed, density 2.5),
CPU and CUDA:** reference vol/mass = 24.0 / 60.0; un-repaired mixed = 4.0 / 10.0
(wrong); reoriented = 24.0 / 60.0; `max|reoriented − reference|` over the packed
10-vector = **0.000e+00** (bit-exact), COM and inertia diffs = 0.0.

**Non-strict on an irregular (symmetry-broken) perturbed icosahedron** (8/20
faces reversed → `orient_faces_by_component` → `from_mesh`): a non-symmetric mesh
so summation order can't hide a partial inconsistency behind cancellation.
`max|by_component − reference|` = **0.000e+00** on both CPU and CUDA.

**Real authored SMPL segmentation** (`git show
main:src/better_human/smpl/config/smpl_3d_segmentation.npy`, `body_faces` = 24
parts over `[v_template(6890); rest_joints J(24)]`, whole-mesh via
`models/smpl/SMPL_NEUTRAL.npz`), run as a standalone probe (see deviation 1):

- Non-strict parts (validate raises: boundary + incidence-3 T-junctions at the
  shoulder-girdle seam): **exactly {9 spine3, 13 left_collar, 14 right_collar}** —
  matches the review's finding. All 24 parts pass through
  `orient_faces_by_component` without raising.
- Whole-mesh volume = **0.074760534**.
- Sum of 24 part volumes, **per-part local integration** (each part sliced to its
  own vertices, the order-03-faithful pipeline) = **0.074764610** →
  **+54.5 ppm (+0.0055 %)**, reproducing the review's +55 ppm and far under the
  1.78 % argmax baseline.
- Sum with **full `[v_template; J]` vertices** passed to every part (whole-body
  centroid centering) = 0.074780470 → +266.7 ppm (+0.0267 %). The 5× difference
  is entirely the open-seam parts {9,13,14}: an open surface's signed tetra
  volume is origin-dependent, and the tool centers by whatever `vertices` it is
  given (matching `from_mesh`). The tool is correct; order 03 should integrate
  each part with its own vertices to get the 54.5 ppm figure. Both are correct
  data points; reporting both.

## Verification

- `tests/spatial/test_mesh.py`: **22 test cases (15 functions, device- and
  input-parametrized), 22 passed, 0 skipped** (`cpu` and `cuda` both ran).
- Full suite `CUDA_VISIBLE_DEVICES=2 uv run pytest tests/ -q`:
  **1584 passed, 0 skipped, 0 failed**, 93 warnings, 130.14 s (was 1575 before
  this order; +9 new cases).
- CUDA: **passed**. Pinned via `CUDA_VISIBLE_DEVICES=2` → NVIDIA RTX 6000 Ada
  Generation; the warp CUDA suite ran (not skipped), confirming the device was
  live. Torch 2.x / Python 3.11.15.
- `uv run ruff check` and `ruff format --check` on the three changed files:
  **clean**.

## Deviations / findings

1. **The real-asset regression test was run as a standalone probe, not committed
   to BR's suite.** The review suggested a skip-if-absent BR test that loads
   `models/smpl/SMPL_NEUTRAL.npz` and the segmentation via `git show main:…` from
   the BetterHuman repo. Committing that into BetterRobot's suite would couple BR
   to BetterHuman's SMPL asset layout + git — directly against the plan's
   BR-purity principle (ground rule 4; order 02 is actively removing all
   SMPL/human specifics from BR). I therefore ran the asset validation as a probe
   (numbers above, reproducing the review's +55 ppm) and instead committed
   equivalent **synthetic** coverage in BR:
   - `test_by_component_orients_disjoint_shells_outward_and_conserves_volume`:
     two disjoint boxes (one with reversed winding) → outward-oriented → mass ==
     density·(vol_A+vol_B) == 48, plus the per-component opposite-half-edge
     invariant. Exercises multi-component + volume conservation.
   - `test_by_component_degrades_gracefully_on_boundary_and_non_manifold_input`:
     open (dropped cap) and incidence-3 (duplicated face) meshes → returns
     without raising, where the strict path raises. Exercises boundary +
     T-junction tolerance.
   - `test_by_component_repairs_mixed_winding_on_an_irregular_mesh` (CPU+CUDA):
     the non-symmetric-mesh case the review asked for.
   **Recommendation:** put the committed 24-part asset regression test in
   **order 03** (BetterHuman), where the asset and the per-part pipeline live.
2. **Repo-wide `ruff check src tests` is not clean at baseline** — 113
   pre-existing errors across 39 unrelated files (mostly `PLC0415` in
   `tests/test_pinocchio/*`), **none in files this order touched**. Not fixed
   (scope discipline). My three files are ruff- and format-clean.
3. **Test dtype is float32** per the workspace `write-tests` skill + handoff (the
   neighbouring `test_inertia_from_mesh.py` uses float64). No gradcheck applies:
   reorientation is a pure integer-table permutation with no gradient path;
   `from_mesh` differentiability is unaffected because only integer indices are
   reordered.
4. **`orient_faces_by_component` takes single-mesh `(V, 3)` vertices, not a
   batch.** The orientation is a static rest-geometry precompute (one integer
   table). It uses vertices only to sign each component's volume; per-batch signs
   would be ambiguous, so a batch axis is intentionally rejected.
5. **Strict and non-strict paths are both kept**, as instructed — the argmax
   fallback (order 03) produces clean manifolds and uses the strict path; the
   authored-segmentation path uses the non-strict path.
