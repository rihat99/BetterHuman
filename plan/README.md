# Plan — BetterHuman refinement: simplify, restore, re-home

State at planning time (2026-07-21, branch `dev`): the six-order rewrite
landed and **works** — SMPL and SMPL-X load, pass float64 parity ≤1e-6 against
the `smplx` oracle, round-trip through `from_classic`/`to_classic`, bake shape
into BetterRobot `ModelValues`, integrate per-part inertias, and render a live
skinned mesh. An owner review + an independent over-engineering audit then
found three classes of problem this plan fixes:

1. **Over-abstraction.** ~340–400 LOC (~13–15% of `src`) is removable with no
   loss of capability or correctness. The single biggest item: a
   "never densify the joint regressor" contract that fans a CSR apparatus
   across five files to avoid materializing a ≤2.3 MB matrix that **every
   shipped asset already stores dense**.
2. **Two silent regressions vs. the old code.**
   - **Landmarks were dropped.** Both models ship `landmarks={}`; the old
     per-body config carried `frame_names` + `frame_vertex_ids` (OpenPose-style
     keypoint vertex tables) and `joint_links` (skeleton edges). `BodyStructure`
     reserves the field but nothing populates it.
   - **The custom volume-preserving segmentation was thrown away** on an
     incomplete rationale. The rejected asset is a set of **pre-closed per-part
     face tables** (joint-as-cap-vertex, total split volume = whole-body
     volume; borders intentionally share vertices; some faces intentionally
     flipped, historically fixed with `trimesh.fix_normals`). It maps directly
     onto `PartMeshTopology` with **zero boundary tracing**. In its place the
     code recomputes closed parts at runtime (argmax + half-edge boundary
     tracing + centroid caps), carrying a measured **1.78 % volume gap** and
     ~120 lines of pure-Python graph code.
3. **BetterRobot contamination.** SMPL/human-specific data lives in
   `better_robot.io.builders.smpl_like` (joint names, parents, human
   proportions), and BetterHuman's `SMPL` even imports `JOINT_NAMES` from
   BetterRobot. BR must have zero SMPL/human specifics; the topology belongs in
   BetterHuman config.

No backward compatibility. Alpha, breaking changes are fine, everything on
`dev`. Detailed measured records of the original six orders remain in git
history (`git show <sha>:plan/for_agents/0N_results.md`); the acceptance
baselines this plan must not regress are captured in **Baselines** below.

## Settled decisions (owner-confirmed 2026-07-21)

These are binding for the orders below; they are not open questions.

1. **Drop the "never densify `J_regressor`" contract.** Every shipped asset is
   dense (1.32 MB f64 SMPL, 2.30 MB f32 SMPL-X); CSR appears only because the
   converter chooses to emit it, and the SMPL-X reduction sparse branch is dead
   (the converter never emits `smplx`). Deleting the CSR path removes ~150 LOC
   across `io/schema.py`, `io/load.py`, `io/convert_chumpy.py`,
   `core/reduction.py`, `core/pipeline.py`. **This reverses `contracts.md`
   "loaders never densify"** — order 05 updates that contract accordingly.
2. **Config format = a `BodyConfig` dataclass + one small pure-data Python
   module per model** (`bodies/config_smpl.py`, `bodies/config_smplx.py`),
   holding joint names, joint-index groups, landmark names + vertex ids,
   `joint_links`, segmentation-asset filenames, density-table name, axis
   conventions, and default parameter widths. Chosen over JSON/TOML:
   type-checked, no parser/schema layer to maintain (parsing would re-introduce
   the machinery we are cutting), data cleanly separated from logic.
3. **Two segmentation sources, one closed topology.** An **authored closed-3D
   segmentation** (volume-preserving, default for SMPL-family where the asset
   exists) and the **argmax fallback** (generic, for models without an asset)
   both produce one `PartMeshTopology`. The argmax path is kept but its
   per-spawn defensive assertions are trimmed.
4. **BetterRobot keeps only the generic builder.** `build_kinematic_tree_*`
   stays (it is clean and human-free). The SMPL topology data leaves BR; BR's
   own ~13 test/bench files + `examples/04_*` are rewired to a generic,
   neutrally-named branching-tree fixture in BR test-support, not a shipped
   "smpl_like" API.
5. **Drop the speculative SMPL-H / MANO converter schemas** (no body consumes
   them; roadmap items belong in `plan/`, not the API). They are re-added with
   the SMPL-H / MANO bodies when those land.

## The orders

| Order | What | Acceptance in one line |
|---|---|---|
| `for_agents/01_br_mesh.md` | (BR) mesh face-orientation + closedness tools (strict validate + non-strict per-component orient), companion to `Inertia.from_mesh` | mixed-winding closed part integrates to correct volume; authored SMPL parts orient via the non-strict tool without rejection; strict path rejects open/non-manifold/non-orientable with actionable error; BR suite green on CPU + CUDA |
| `for_agents/02_config.md` | `BodyConfig` per model + **re-home SMPL topology out of BR into config**; refactor `bodies/*`; rewire BR's ~13 tests + `examples/04_*` to a generic fixture; restore landmarks | BR shipped API has no SMPL/human symbols and its suite is green on the generic fixture; `bodies/*` import no topology from BR; `smplx.py` shrinks to converter logic; `landmarks` non-empty and exercised by a real consumer; parity/round-trip green |
| `for_agents/03_segmentation.md` | two segmentation sources → one closed `PartMeshTopology`; restore custom asset via BR winding-fix; simplify `mass.py` | custom-seg SMPL total mass matches whole-mesh volume to tight tol (closes the 1.78 % gap); argmax path still tested; both differentiable + batched |
| `for_agents/04_simplify.md` | delete CSR apparatus; collapse `schema.py` symbolic DSL; de-duplicate validation; narrow `core` surface; drop SMPL-H/MANO schemas | net `src` LOC down ~300–400; every parity/round-trip/gradcheck/mass test still green; ruff clean |
| `for_agents/05_docs_render.md` | render per-model × per-segmentation comparison images into `docs/`; update `architecture`/`contracts`; migrate roadmap | images exist for SMPL & SMPL-X × {argmax, custom}; docs true to code |

Dependencies: **01** (BR mesh tools) is a pure addition to BetterRobot and
depends on nothing — it is the tool order 03 leans on. **02** re-homes the SMPL
topology from BetterRobot into `BodyConfig`; the removal from BR and the arrival
in BetterHuman config land together so **both repos stay green** (this is why the
`smpl_like` extraction lives here, not in 01 — deleting it from BR before
BetterHuman has its own topology would leave BetterHuman red). **03** depends on
01 (mesh orientation) and 02 (config holds the segmentation-asset references).
**04** is independent deletion that may run any time after 01; scheduled after 03
to avoid churn on shared files (`mass.py` belongs to 03; 04 owns
`io`/`schema`/`load`/`reduction`/`pipeline`). **05 last.** Each order ends with
the full test gate green — across **both** repos — and docs true.

### 01 — BetterRobot: mesh face-orientation + closedness tools

`Inertia.from_mesh` folds *global* winding but not *mixed* per-face winding
within one surface (order 05 found it accepts a box with one reversed face). Add
torch-only, GPU-safe tools in `spatial/mesh.py` (companion to
`Inertia.from_mesh`):

- **Strict:** `validate_closed_manifold(faces)` (raises on boundary /
  non-manifold / non-orientable input, naming offending edges) and
  `orient_faces_consistently(faces)` (consistent winding for a validated closed
  orientable manifold). Order 03's argmax fallback produces clean manifolds and
  uses these.
- **Non-strict:** `orient_faces_by_component(vertices, faces)` — orient per
  connected component over incidence-2 edges only (ignoring boundary /
  non-manifold edges), then flip each component to a positive mean-centered
  signed volume so `from_mesh`'s global fold is a no-op. **Required because the
  authored SMPL segmentation is not strictly manifold** (order 03). Best-effort,
  never raises; output is a static integer face table, so downstream integration
  stays differentiable.

Reorientation is a **static topological** precompute baked into the face table —
the vertex integration stays differentiable. These replace the historical
`trimesh.fix_normals`. First of the two authorized BR edits; the `smpl_like`
extraction lands in order 02.
*Acceptance:* a mixed-winding closed mesh integrates to the correct positive
volume after reorientation; the authored SMPL parts (incl. the 3 open/non-manifold
seam parts) pass through `orient_faces_by_component` without raising; open /
non-manifold / non-orientable input to the strict path raises naming offending
edges; BR full suite green on CPU + CUDA; change listed in the results file.
*Status (2026-07-21):* strict tools implemented + independently verified (13
tests green, mixed-winding integration exact CPU+CUDA); non-strict tool and the
non-orientable-detection fix in progress.

### 02 — Body config + landmark restoration

Introduce a `BodyConfig` (frozen dataclass) holding per-body **metadata**:
`joint_names`, joint-index groups (SMPL-X `body`/`face`/`left_hand`/`right_hand`
ranges that drive the reduction flags), landmark `frame_names` +
`frame_vertex_ids`, `joint_links` (skeleton edges), segmentation-asset
filenames, density-table name, axis conventions (`unit`/`up`/`forward`),
`native_pose_format`, and default parameter widths. One pure-data module per
model. Config references untracked weight/segmentation assets by filename; the
loader resolves them from the models dir.

Refactor `bodies/smpl.py` and `bodies/smplx.py` to read from config — this
removes the 57-line inline `SMPLX_JOINT_NAMES` tuple, the `BODY_JOINTS` /
`FACE_JOINTS` / `HAND_JOINTS` constants, and the BR `JOINT_NAMES` import,
shrinking `smplx.py` toward just its converter logic. Populate
`BodyStructure.landmarks` and the convention fields from config, and give
`landmarks` a real consumer (viz landmark-point overlay) so the field is live,
not documentation. This is the constructive answer to the audit's "dead
`BodyStructure` fields" finding: **populate + consume**, do not delete (the
convention fields are `contracts.md`-mandated; landmarks are a restored
feature). Restore the tracked landmark/`joint_links` metadata from git history.

**Re-home the SMPL topology out of BetterRobot (the second authorized BR edit).**
Move the SMPL/human topology (`JOINT_NAMES`, `PARENTS`, human proportion offsets,
`make_smpl_like_*`) **out** of `better_robot.io.builders.smpl_like`; BR keeps the
generic `build_kinematic_tree_body` / `build_kinematic_tree_model`. The real SMPL
topology arrives in BetterHuman's `BodyConfig` in the same order, so both repos
stay green. BR's own suite uses `make_smpl_like_*` as its representative
free-flyer + spherical fixture across ~13 test/bench files (two import
`JOINT_NAMES`/`PARENTS` directly) + `examples/04_smpl_like_body.py`; rewire them
to a generic, neutrally-named branching-tree fixture (e.g. `j0..jN`, arbitrary
offsets) in BR **test-support**, not the shipped API. Every BR change is listed
in the order's results file.
*Acceptance:* `bodies/smpl.py`/`smplx.py` import no topology from BetterRobot;
`smplx.py` materially shorter; `landmarks` non-empty for both models and
exercised by a real consumer (a `landmark_positions()` accessor on the body,
plus an optional viz overlay); no SMPL/human-specific symbol remains in BR's
shipped API
(`grep -riE "smpl|pelvis|betas|human" src/better_robot` finds only unrelated
comments); BR full suite + example green on the generic fixture; all BetterHuman
parity, round-trip, and gradcheck tests still green.

### 03 — Segmentation rework + `mass.py` simplification

Define one closed `PartMeshTopology` produced by either source:

- **Authored closed-3D segmentation** (default for SMPL-family): load the
  vetted asset's per-part face tables (`body_faces[j]` indexing
  `[shaped_verts; rest_joints]`, the joint as cap vertex), orient each part with
  BR's non-strict `orient_faces_by_component`, and integrate via
  `Inertia.from_mesh` — **no boundary tracing, no centroid caps**. Verified
  reality (probe on the real asset): 21/24 parts are strict closed manifolds;
  **3 parts — spine3, left_collar, right_collar — are genuinely open +
  non-manifold at the shoulder-girdle seam**, and spine3 splits into 3
  components. Do **not** gate authored parts on `validate_closed_manifold` (it
  would reject 17.5 % of body volume); use the non-strict per-component path
  uniformly on all parts. Because those 3 parts are open, per-part signed volume
  is origin-dependent (~0.1–1 %/m); `Inertia.from_mesh` integrates at the part
  mean, so **assert the part-mean-centered aggregate, not each open part's
  absolute volume**. Measured aggregate: 24-part volume sum within **+0.0055 %**
  of the whole mesh.
- **Argmax fallback** (generic, models without an asset): keep the
  skinning-argmax + successor-walk boundary closure, but drop the per-spawn
  manifold/winding/cycle assertions (they can only fire when onboarding a new
  model on a vetted mesh; gate behind a one-time onboarding check if kept).

Both feed the differentiable, batched `Inertia.from_mesh`. This deletes the
bulk of `mass.py`'s pure-Python graph code and closes the volume gap on the
custom path. Also restore the per-part density tables (`body_densities.json`:
Naive/Chandler/Dempster/Clauser/Custom) as a named-table density option, feeding
the same tensor density path (no new policy code).
*Acceptance:* custom-segmentation neutral SMPL total mass equals whole-mesh
integration to a tight tolerance (materially better than the 1.78 % argmax gap);
argmax path retained and tested; both paths float32-gradcheck clean and batched;
`mass.py` LOC down.

### 04 — Over-engineering cleanup (the audit's cuts)

Owns `io/`, `core/schema`-adjacent, `core/reduction.py`, `core/pipeline.py`,
`bodies/registry.py`, `core/__init__.py`. Does **not** touch `mass.py` (order 03)
or the config/body files (order 02).

- **Delete the CSR apparatus** (decision 1): converter emits a dense regressor;
  remove `dense_or_csr`, the `CSR_*` constants, `_validate_csr`,
  `_expect_component`, `_load_sparse_regressor`, the sparse branch of
  `reduce_joint_data`, and the sparse branch of `regress_joints`. (~150 LOC)
- **Collapse the symbolic-dimension DSL** in `io/schema.py` to explicit
  `(dims, dtype)` specs with named-axis binding and one inline special case for
  the single derived dim `(J-1)*9`; drop `Dimension.describe`/`scale`/`offset`
  and `minimum_dimensions`. Keep shape+dtype checking (rule #10). (~50 LOC)
- **De-duplicate validation:** validate at the true public boundaries
  (`with_shape`, `SMPL/SMPLX.__init__`, the loader) only; internal helpers trust
  their validated inputs. Single density validation (drop the second copy).
  Narrow `core/__init__` exports to the real public surface; drop the test-only
  `part_meshes` wrapper. (~75 LOC)
- **Drop SMPL-H / MANO converter schemas** and the `schema_for_model`/`SCHEMAS`
  indirection (decision 5); the converter uses `SMPL_SCHEMA` directly. (~30 LOC)
- **Minor:** remove the dead ternary in `_validate_csr`'s shape branch and the
  dead rank guard in `_bind_plain_dimensions`; move the local `so3` import in
  `pipeline.py` to module scope; leave the `ParametricBody._replace` /
  `object.__setattr__` pattern as-is unless a cleaner functional-subclass shape
  falls out for free (it is the tax of thin subclassing, not a bug).

*Acceptance:* net `src` LOC down ~300–400; parity, round-trip, gradcheck, and
mass tests all still green at unchanged tolerances; ruff clean. **The safety net
only grows** — no parity/round-trip/gradcheck test is weakened to land a cut.

### 05 — Docs + rendered segmentation comparison

Render, for each model (SMPL, SMPL-X) and each segmentation type (argmax,
custom), an image of the colored body-part assignment and a split/exploded view
of the parts, saved under a `docs/` subfolder (e.g. `docs/segmentation/`), with a
short page explaining the two types so the difference is visually obvious.
Update `architecture.md` and `contracts.md` for the two-source segmentation, the
config system, the restored landmarks, and the dropped CSR contract. Migrate the
old plan's roadmap into docs.
*Acceptance:* four images (2 models × 2 seg types) plus the split views exist and
are referenced from docs; every doc page and docstring the plan falsifies is
updated; docs build clean.

## Baselines (the rewrite must not regress these)

| Quantity | Baseline | Note |
|---|---|---|
| SMPL/SMPL-X float64 parity vs `smplx` oracle | ≤ 1e-6 m (worst measured ~1e-7) | sacred; never weakened |
| float32 parity | pinned at 1e-6 (worst ~7e-7) | keep pinned |
| Partial-model property (SMPL-X reduced vs full-with-identity DOFs) | ≤ 5e-7 m | keep |
| Neutral SMPL mass — argmax (density 1000) | 73.43 kg parts / 74.76 kg whole, **1.78 % gap** | the argmax fallback's gap |
| Neutral SMPL volume — custom seg (measured in verification) | 24-part sum within **+0.0055 %** of whole mesh | order 03's target; the custom path must hit this |
| Aggregate vs whole-mesh COM separation | 2.47 mm (argmax) | expected to improve on custom |
| CRBA min eigenvalue (SPD check) | > 0 | keep |
| Test gate | full suite green; CUDA cases pass on a pinned GPU | skips stay visible |

## Ground rules

1. **Read `docs/design_principles.md`, `docs/architecture.md`,
   `docs/contracts.md` before writing code.** They are binding; update the page
   an order falsifies in the same order.
2. **No backward compatibility.** Alpha, breaking changes fine, all on `dev`.
   No shims, no aliases, no migration paths.
3. **The safety net only grows.** Parity goldens, converter round-trips, float32
   gradchecks, and mass conservation are never weakened to land a change.
4. **BetterRobot is edited only in the two authorized areas** — mesh tools (01a)
   and the `smpl_like` extraction (01b). Any other BR gap is reported in the
   results file, not worked around by duplicating BR functionality here. Every
   BR change is listed in the results file.
5. **Never touch `models/` (weights) or `references/`** except reading. Restore
   tracked metadata/segmentation assets from git history into the package, not
   from the license-gated `models/` tree.
6. **Honest results files.** Each order writes `for_agents/0N_results.md`:
   delivered, deviations-as-findings, exact test counts, measured numbers as
   measured (never rounded to "green"). LOC deltas reported per file.
7. **Subagents run on Opus.** No feature branches, no worktrees unless asked.
8. **CUDA is agent-runnable here** (RTX 6000 Ada; pin via
   `CUDA_VISIBLE_DEVICES`). Device-conditional code is validated on CPU and CUDA
   or explicitly reported as not.

## Beyond this plan (roadmap, not scheduled)

MANO / SMPL-H / FLAME (same family: schema + config + tables); STAR / SUPR
(quaternion pose features, β-dependent correctives — first real test of the
corrective seam); MHR (authored skeleton, parameter transform, MLP correctives —
promotes the `core` seams to protocols); Anny (CC0 default, phenotype shape
space); keypoint/landmark output heads (127/144-joint OpenPose conventions);
human residuals on the BR optimizer (swing/twist tables, pose priors);
human→robot retargeting (`better_human.retarget`); cross-model conversion;
differentiable measurements (`better_human.measure`).
