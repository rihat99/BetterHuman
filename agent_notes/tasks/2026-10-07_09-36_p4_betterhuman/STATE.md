# STATE — BetterHuman / 2026-10-07_09-36_p4_betterhuman (updated 2026-10-07 09:40 Dubai)

## Pointers
- Plan: /home/rikhat.akizhanov/better/dev/BetterHuman/agent_notes/tasks/2026-10-07_09-36_p4_betterhuman/PLAN.md
- Design: /home/rikhat.akizhanov/better/dev/BetterRobot/agent_notes/tasks/2026-10-06_20-13_architecture_plan/DESIGN.md
  (D5, Q5, Q6, Q8) and `subagents/scan_better_human.md`
- Done before this: P1 optim2, P2 kin/dyn, P3 viewer in BR dev (reports under
  /home/rikhat.akizhanov/better/dev/BetterRobot/agent_notes/tasks/2026-10-07_*/REPORT.md). BH's editable
  dependency points at that BR worktree.
- Owner is asleep: work automatically, ask nothing, record assumptions. Never push/merge; commit on `dev`.
- Never bind servers to 0.0.0.0; GPU only when FREE; no new dependencies; trash, never rm.

## Status
- Task folder created, PLAN.md written. No code yet.

## Next steps
1. M0: run BH tests against BR dev, list breakage; list assets; list BR private imports (grep `better_robot.`).
2. M1 FamilySpec + bake/stream with parity gate; then M2–M7 per PLAN.md. Commit after each milestone.

## Assumptions (made off-hours)
- Keep MHR's corrective MLP as a family-specific corrective kind behind the same `FamilySpec` interface.
- SMPL-H/MANO/FLAME only if their assets exist under /home/rikhat.akizhanov/better/better_human; otherwise
  loaders are written against the documented npz layout and marked untested in the report.
