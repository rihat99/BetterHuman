# STATE — BetterHuman / 2026-10-07_09-36_p4_betterhuman (updated 2026-10-07 12:10 Dubai)

## Pointers
- Plan: /home/rikhat.akizhanov/better/dev/BetterHuman/agent_notes/tasks/2026-10-07_09-36_p4_betterhuman/PLAN.md
- API contract: same folder, `M1_DESIGN.md` (M0 facts + FamilySpec/Rig/Body/Identity/HumanState + loaders).
- Subagent context: `subagents/context_m1.md`.
- Design: /home/rikhat.akizhanov/better/dev/BetterRobot/agent_notes/tasks/2026-10-06_20-13_architecture_plan/DESIGN.md (D5, Q5–Q8)
- Owner is asleep: work automatically, ask nothing, record assumptions. Never push/merge; commit on `dev`.
- Never bind servers to 0.0.0.0; GPU only when FREE; no new dependencies; trash, never rm.
- `models/` in the worktree is a gitignored symlink to the main checkout's assets.

## Status
- M0 done (facts in M1_DESIGN.md). BR dev got `Model.expand_configuration` (uncommitted in BR; commit with M2).
- M1 part 1 committed (fcb425d): `core/{spec,rig,body,pipeline}.py`, `families/{smpl_family,classic,tables,assets}`,
  SMPL/SMPL-H/SMPL-X/MANO/FLAME loaders, `bh.load`. SMPL-X parity on the new API: worst 5.1e-7 m (gate 1e-6),
  expression streamed per frame with exact joints; `stream_joints=False` is off by ≤1.2 mm.
- Running: mars_top ports MHR (`families/mhr.py`, `mhr_rig.py`, parity + family tests); mars_top_opus ports the
  SMPL-family tests to `tests/families/`, adds SMPL-H/MANO/FLAME tests.
- Old `bodies/`, `core/classic` leftovers, `tests/bodies` to be trashed after both agents finish.

## Next steps
1. Integrate both agents; trash `bodies/`; full test run; commit M1.
2. M2: contract test "no private BR imports"; commit BR `Model.expand_configuration`.
3. M3 terms (`better_human/terms.py` on optim2), M4 `viz/human_layers.py`, M5 studio, M6 benchmarks, M7 docs/review.

## Assumptions (made off-hours)
- Expression is a stream group (per frame), exact joints by default (`Body(stream_joints=True)`), switchable.
- FLAME: no asset on this machine; loader written against the documented npz layout, tested on a synthetic file.
- Contact frames stay SMPL-family only (regressed rig); MHR has pre-rotated joints and never had them.
- Pose given for a dropped subtree (hands=False) raises instead of being ignored.
- `bh.load(name, path, **options)` replaces the registry classes `bh.SMPL/SMPLX/MHR`.
