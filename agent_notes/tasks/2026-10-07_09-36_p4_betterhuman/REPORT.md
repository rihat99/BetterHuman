# REPORT — P4 BetterHuman on the new BetterRobot (2026-10-07)

Task folder: `/home/rikhat.akizhanov/better/dev/BetterHuman/agent_notes/tasks/2026-10-07_09-36_p4_betterhuman/`
(plan `PLAN.md`, API contract `M1_DESIGN.md`, subagent notes in `subagents/`). Everything is committed on the
`dev` branches: BetterHuman `fcb425d..HEAD`, BetterRobot `12d1287..fc49fc2`. Nothing was pushed or merged.

## Outcome in one paragraph

BetterHuman was rewritten as a thin layer on BetterRobot's public API. A body family is now data (`FamilySpec`)
plus one behaviour (`Rig`); a subject is baked once (`Body.bake -> Identity`) and posed per frame
(`Identity.forward -> HumanState`), with expression streamed per frame and exact joints. The SMPL-X vertex gate
holds at 5.1e-7 m (gate 1e-6 in the tests, 6e-7 in the plan), MHR holds on all 7 LODs, and the new core is
faster than upstream smplx on CPU (0.42–0.95× its time) and about equal on GPU. Human cost terms run on
`optim2`, the viewer is BetterHuman layers on the BetterRobot `Viewer`, and the studio (posing, segment
painting, contact frames, persistence) runs on that same viewer. The suite is green except one pre-existing
1-ulp test (listed below).

## What changed, by milestone

| milestone | result | where |
|---|---|---|
| M0 facts | BH did not import against BR dev (old `RenderContext`, `Data`, private builders). Listed in `M1_DESIGN.md`. | `M1_DESIGN.md` |
| M1 core | `FamilySpec`, `Corrective`, `Parts`, `Rig`/`RegressedRig`/`AuthoredRig`, `Body`, `Identity`, `HumanState`, `drop_joints`, top-K skinning; loaders smpl/smplh/smplx/mano/flame/mhr as tables + short functions; classic converters per family. | `src/better_human/core/`, `families/` |
| M2 public API only | `tests/contract/test_br_imports.py` forbids private BR imports. BR gained four small public pieces instead: `Model.expand_configuration`, camera `up` axis, `PoseGizmo(**options)`, `SkeletonLayer.colors`, `ModelBuilder(gravity=)`. | BR `12d1287`, `dc0dae3`, `5f4d393`, `fc49fc2` |
| M3 terms | `shaped`, `human_state`, `joints`, `points`, `landmarks`, `reprojection`/`project`, `ground`, `contact`, `pose_prior`, `shape_prior`, `contact_wrenches`, `root_wrench` (root rows of inverse dynamics with contact forces, `window=3`). | `src/better_human/terms.py`, `tests/test_terms.py` |
| M4 viewer | `human_layers(state, mesh="exact"/"skinned", skeleton, frames, forces)` and `HumanView.update(state)` on the BR `Viewer`; examples 01–04 rewritten; `artifacts/smplx_swing.gif` recorded offline. | `src/better_human/viz/human_layers.py`, `examples/` |
| M5 studio | `HumanStudio` on the BR `Viewer`: BR layers + two studio layers (`SegmentationLayer`, `PartsLayer`), `PoseGizmo` posing, BR picking tools, same editors and `studio_data/` formats. See the studio section. | `src/better_human/viz/studio/`, `docs/studio.md` |
| M6 benchmarks | `benchmarks/run.py` (CLI) + `report.py`; CPU and CUDA JSON. | `benchmarks/results/p4_bodies_{cpu,cuda}.json` |
| M7 docs/review | `docs/architecture.md`, `docs/contracts.md`, `README.md`, `CLAUDE.md`, `docs/studio.md`; clean-context review found 2 bugs + 3 contract gaps, all fixed (commit `5cfa5de`). | `docs/`, this report |

## Numbers (the metrics agreed in the plan)

Parity (float32 and float64, CPU):

| family | gate | measured |
|---|---|---|
| SMPL-X vs upstream smplx, 29 goldens, expression streamed | 1e-6 m (plan: 6e-7) | worst 5.1e-7 m |
| SMPL-X `stream_joints=False` (approximate joints) | documented, not gated | 1.2e-3 m joints |
| MHR vs TorchScript, LOD 0–6 | `PARITY_ATOLS` 2e-6/1e-6/1e-6/1e-7 | float32 vertices ≤ 1.13e-6, float64 ≤ 8.5e-7 |

Speed, median wall time in ms, `ours / upstream` in brackets (below 1 = ours is faster). CPU: AMD EPYC 9654,
8 threads, float32. CUDA: one GPU, measured only because `gpu_free.py` reported it FREE. Full tables:
`subagents/m6_bench_cpu.md`, `subagents/m6_bench_cuda.md`.

| family, op | CPU B=1 | CPU B=1024 | CUDA B=1 | CUDA B=1024 |
|---|---|---|---|---|
| SMPL vertices | 1.21 (0.69) | 296 (0.95) | 1.19 (0.80) | 8.1 (0.99) |
| SMPL vertices fwd+bwd | 3.25 (0.73) | 658 (1.04) | 6.3 (0.81) | 22.4 (1.01) |
| SMPL-X vertices, no expression | 3.02 (0.42) | 464 (0.74) | 1.25 (0.47) | 12.3 (0.82) |
| SMPL-X vertices, expression streamed, exact joints | 3.54 (0.50) | 500 (0.80) | 2.06 (0.78) | 13.4 (0.90) |
| SMPL-X fwd+bwd, expression streamed | 7.85 (0.53) | 996 (0.86) | 9.1 (0.85) | 35.5 (0.96) |
| MHR LOD1 vertices vs TorchScript | 22.4 (0.59) | 1310 (0.25) | 2.5 (1.15) | 26.8 (1.02) |
| MHR fwd+bwd (no baseline) | 48.3 | 2848 | 13.5 | 71.9 |

Other rows: bake at B=1 on CPU is SMPL 0.41 ms (11.2 with mass), SMPL-X 1.22 (23.5), MHR 6.2 (20.3); FK /
RNEA at B=1024 on CPU: SMPL 6.9 / 15.8 ms, SMPL-X 12.0 / 25.6, MHR 41.2 / 45.2. Every vertex cell at B=8192 was
skipped (estimated 11–39 GB against the 8 GB limit); FK and RNEA ran at 8192. Caveats: the upstream SMPL-X call
always includes expression, ours "no expression" passes none; the MHR TorchScript model blends identity on every
call while ours bakes it once. Viewer: 0.9 ms per frame push; GIF of 30 frames rendered offline in 1.5 s.
Optimisation: SMPL-X marker fit ~90 ms per LM iteration at B=4; `root_wrench` Jacobian check 1.6e-7; free-fall
root residual < 1e-6 (test).

## Review findings and what was done

Clean-context review (mars_aux, scripts under the session scratchpad `review_m7/`):

| finding | fix |
|---|---|
| `HumanState` memo ignored grad mode: a `no_grad` read poisoned later grad-enabled reads | memo keyed by grad mode (`body.py`), test `test_state_memo_does_not_leak_no_grad_results` |
| `smplx_to_classic` on `hands=False` returned `-hand_mean` and the round trip raised | `_hand_pose` returns `None` for dropped hands, test `test_body_without_hands_round_trips_classic` |
| Model gravity was BR's `-z` while the families are y-up | `ModelBuilder(gravity=)` in BR; both rigs set gravity against `spec.up_axis`; free-fall test |
| `project` broadcast a per-sequence camera against the wrong axis when B == T | camera axes are left-aligned and validated; test |
| `root_wrench` / `contact_wrenches` accepted wrong shapes with cryptic errors; empty contact list crashed | explicit checks, forces broadcast, long dtype for an empty index |
| streamed joint shift re-bakes joint placements only (frames/inertias keep bake-time values) | no effect on current assets (shift of pelvis and foot frames is exactly 0); documented as a limitation below |
| axis-angle round trip is exact only up to rotation equivalence (norm > π comes back < π) | expected; documented in `docs/contracts.md` |

Attacks that found nothing: `contact_wrenches` vs an independent root wrench (4e-14), static equilibrium with
12 contact frames (3.5e-13), free fall with a random root rotation (1e-10), identity × pose batching vs loops
(1e-16), float32 vs float64 (2.7e-7 m), SMPL-X classic round trips with PCA hands (≤ 1.8e-7 in float32).

## Studio (M5)

Commit `c2caa2f`. `HumanStudio` is a client of the BR `Viewer` (`Viewer(theme=replace(DARK, up="+y"))`):
scene geometry is BR layers only (the BetterHuman mesh/frames layers, a `SkeletonLayer` with public per-joint
colours, and two studio layers `SegmentationLayer` and `PartsLayer` in `viz/studio/layers.py`); posing uses
`better_robot.viewer.tools.PoseGizmo` (rotation-only handles for non-root joints, 6-DoF root); picking uses the
BR `ray_hit` / `vertices_in_sphere` / `vertices_in_rect` tools through one scene pointer handler. The studio's own
viser layers and picking code were trashed. `StudioSession` is the pure-state hub on `Body`/`Identity`/`HumanState`.

Feature checklist from the old `docs/studio.md` (every item has a headless test unless marked):

| feature | status | test |
|---|---|---|
| Body tab: family/gender/LOD load, shape sliders (first 10 per group) with Reset/Random, density (SMPL table dropdown: Naive/Chandler/Dempster/Clauser/Custom; number input otherwise) + Apply density, total mass, pose Save/Load with saved-pose dropdown, Deselect joint | kept | `test_studio_app.py` |
| Posing: click joint sphere or mesh -> gizmo at the joint, rotation-only vs 6-DoF, fixed MHR joint reports in status, mesh dimmed and sphere highlighted while selected | kept | `test_studio_gizmo.py`, `test_studio_app.py` |
| Layers tab: visibility toggles, Color by segmentation, opacity, joint radius, explode | kept | `test_studio_layers.py` |
| Segment tab: active segmentation (argmax / built-in json / saved), sphere brush, rectangle marquee, add part, Build bodies with per-part + total mass table (closed meshes; total equals `total_mass` within 1 % for the default segmentation) | kept | `test_studio_layers.py`, `test_studio_app.py` |
| Frames tab: add at nearest vertex, select by click, rename, parent-joint override, delete, save/load, Export as config (`contact_frames={...}` for `bh.load`) | kept; export format changed with the API | `test_studio_app.py`, `test_studio_persistence.py` |
| `studio_data/{segmentations,frames,poses}/<model_key>/` formats | unchanged | `test_studio_persistence.py` |
| Fast (client-side skinned) toggle, wireframe overlay | dropped: the BR `MeshLayer` is one exact per-frame mesh; `human_layers(mesh="skinned")` exists for clips | — |
| Click a selected frame again to clear the highlight | not restored (documented) | — |

Manual checks: `python -m better_human.viz.studio --family {smpl,smplx,mhr --lod 1}` each print
`http://127.0.0.1:<port>` and build the GUI without a traceback (killed after 20–25 s). Browser pointer
interactions (drag, brush, marquee) are exercised through their handlers with synthetic events only.
Studio tests: 65 (old: 65) — 7 app, 22 persistence, 7 posing, 7 gizmo, 8 layers, 2 state, 4 human layers, 2
contract, plus 6 added in the finishing pass; dropped tests are listed in `subagents/m5_part1_state.md` and the
part 1b/2 hand-backs (own-viser layer internals, the gizmo worker thread, picking moved to BR's own tests).

## Suite

`CUDA_VISIBLE_DEVICES="" BETTERHUMAN_MODELS_DIR=$PWD/models uv run pytest tests -q`: 298 passed, 1 skipped, 1 failed. One known
failure is pre-existing and unrelated: `tests/core/test_reduction.py::test_corrective_slice_is_exact_when_dropped_joint_is_at_rest`
(exact float compare, off by 1 ulp). Examples 01–04 pass `--check`. `ruff check` is clean on `src tests examples
benchmarks`.

## Assumptions (made off-hours)

- Expression is a stream group (per frame), exact joints by default (`stream_joints=True`), switchable per `Body`.
- FLAME: no asset on this machine; the loader follows the documented npz layout and is tested on a synthetic file.
- Contact frames are SMPL-family only; MHR never had them. Posing a dropped subtree raises.
- `bh.load` + `families.<name>` replace the registry classes; classic converters are functions per family;
  `mhr_from_classic(body, classic)` takes a `Body` and returns `(identity, q, stream)` because MHR's classic
  parameters include identity.
- BR received small public additions (listed under M2) instead of BetterHuman reaching into private modules.
- Gravity follows the family's `up_axis` (−y for every family here). Code that assumed BR's −z default must
  rotate the root instead.

## Open items

- `core/pipeline.py` still has the old `lbs` / `skinning_transforms*` functions, used by tests only (kept as
  oracles for `lbs_affines` / `skin_affines`); trash them when the tests move.
- MHR tests were not ported for gradcheck, CRBA PSD and CUDA; the MHR parity test file is not `ruff format` clean.
- `ruff format --check` reports a handful of pre-existing files; `ruff check` is clean.
- Streamed joint shift: contact-frame placements and inertias stay at bake-time values (no effect on shipped
  assets, see review).
- B=8192 vertex benchmarks need a higher memory limit (`--memory-gb`) or chunking; not measured.
- MHR on CUDA is 1.02–1.15× the TorchScript time (CPU is 0.25–0.59×); the LOD1 authored rig path was not
  profiled on GPU.
- Studio: browser interactions are not covered by tests (headless fake server only); see `docs/studio.md`.
