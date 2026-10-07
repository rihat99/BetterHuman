# STATE — BetterHuman / 2026-10-07_09-36_p4_betterhuman (updated 2026-10-07 15:30 Dubai) — DONE

## Pointers
- Plan: /home/rikhat.akizhanov/better/dev/BetterHuman/agent_notes/tasks/2026-10-07_09-36_p4_betterhuman/PLAN.md
- API contract: same folder, `M1_DESIGN.md`. Subagent context: `subagents/context_m1.md`.
- Owner is asleep: work automatically, ask nothing, record assumptions. Never push/merge; commit on `dev`.
- Never bind servers to 0.0.0.0; GPU only when FREE; no new dependencies; trash, never rm.
- `models/` and `references/` in the worktree are gitignored symlinks to the main checkout.
- Run tests: `CUDA_VISIBLE_DEVICES="" uv run pytest tests -q` (studio tests `tests/viz/test_studio_*.py`
  still fail at import until M5). viser is installed in the BH venv (`uv sync --extra studio --extra viewer`).

## Status (BH dev 69571ce; BR dev 12d1287 = `Model.expand_configuration`, camera `up` axis)
- M0–M4 DONE and committed. Core `core/{spec,rig,body,pipeline}`; families smpl/smplh/smplx/mano/flame/mhr
  (`families/`), classic converters `families/classic.py` + `families/mhr.py`; `bh.load(name, path, **opts)`;
  `better_human.terms` (markers, landmarks, reprojection, ground, priors, shape fit, `root_wrench` ID term with
  contact forces, window=3, check_jacobian 1.6e-7); `viz/human_layers.py` (`human_layers`, `HumanView`) on the BR
  viewer; examples 01–04 rewritten (`--check` all pass); contract test `tests/contract/test_br_imports.py`.
- Numbers: SMPL-X parity worst 5.1e-7 m (gate 1e-6), expression streamed with exact joints; `stream_joints=False`
  off by 1.2 mm; MHR parity all LODs within tolerance, LOD1 vertices 22 ms (B=1) / 112 ms (B=64) CPU = old speed;
  SMPL-X marker fit ~90 ms/LM iter (B=4); viewer push 0.9 ms/frame; GIF artifact `artifacts/smplx_swing.gif`.
- Suite (non-studio): 231 passed, 1 skipped, 1 known failure `tests/core/test_reduction.py::
  test_corrective_slice_is_exact_when_dropped_joint_is_at_rest` (1-ulp exact compare, pre-existing).
- M6 DONE (BH 9ad4093): `benchmarks/run.py` + `report.py`, results `benchmarks/results/p4_bodies_{cpu,cuda}.json`,
  notes `subagents/m6_bench_{cpu,cuda}.md`. CPU B=1: SMPL 0.69×, SMPL-X 0.42–0.50×, MHR 0.59× of upstream time;
  B=1024: 0.95× / 0.73–0.86× / 0.25×. CUDA ~0.5–1.0×, MHR 1.0–1.15× (not faster than TorchScript). B=8192 vertex
  cells skipped (memory estimate > 8 GB).
- BR dc0dae3: `PoseGizmo(**options)` passthrough (rotation-only handles) for the studio.
- M7 core fixes + docs DONE (BH 5cfa5de): review found 2 bugs + 3 contract gaps, all fixed with tests
  (`tests/test_terms.py`, hands=False round trip); BR fc49fc2 `ModelBuilder(gravity=)`, 5f4d393
  `SkeletonLayer.colors`, dc0dae3 `PoseGizmo(**options)`. Docs rewritten (architecture, contracts, README,
  root CLAUDE.md). REPORT.md drafted (placeholders STUDIO_SECTION, SUITE_COUNTS).
- M5 studio DONE (BH c2caa2f): 65 studio tests, SMPL/SMPL-X/MHR launches on 127.0.0.1.
- FINAL: full suite 298 passed, 1 skipped, 1 known failure; REPORT.md written; index line dropped.

## Next steps
None — task complete. Owner review of `REPORT.md`; merge is the owner's call (never merged here).

## Assumptions (made off-hours)
- Expression is a stream group (per frame), exact joints by default, switchable per Body.
- FLAME: no asset here; loader against the documented npz layout, synthetic-file test only.
- Contact frames SMPL-family only; MHR never had them. Pose for a dropped subtree raises.
- `bh.load` + `families.<name>` replace the registry classes; classic converters are functions per family.
- BR got two small public additions (expand_configuration, camera up axis) instead of BH reaching in.
