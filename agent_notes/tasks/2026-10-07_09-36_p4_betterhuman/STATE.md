# STATE — BetterHuman / 2026-10-07_09-36_p4_betterhuman (updated 2026-10-07 13:40 Dubai)

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
- RUNNING: mars_top_opus benchmark agent (`benchmarks/run.py`, `report.py` written, uncommitted; results to
  `benchmarks/results/p4_bodies_cpu.json` + `subagents/m6_bench_cpu.md`; cuda only if a GPU is FREE).

## Next steps
1. Collect the benchmark agent's report; commit `benchmarks/` + results (M6).
2. M5 studio on the BR `Viewer` (3,171 lines, 65 tests in `tests/viz/test_studio_*`): delegate in two parts
   (StudioSession + layers/HumanView + posing/gizmo; then segment/frames editors + persistence + tests); keep every
   feature of `docs/studio.md`; picking from `better_robot.viewer.tools.picking`; trash `viz/studio/layers`.
3. M7: docs (`docs/architecture.md`, `contracts.md`, README, `CLAUDE.md`), mars_aux review (core + terms +
   classic converters), REPORT.md, drop the BH index line.

## Assumptions (made off-hours)
- Expression is a stream group (per frame), exact joints by default, switchable per Body.
- FLAME: no asset here; loader against the documented npz layout, synthetic-file test only.
- Contact frames SMPL-family only; MHR never had them. Pose for a dropped subtree raises.
- `bh.load` + `families.<name>` replace the registry classes; classic converters are functions per family.
- BR got two small public additions (expand_configuration, camera up axis) instead of BH reaching in.
