# PLAN — P4 BetterHuman (families as data, streams, public BR API, terms, viewer layers, studio)

Task id `2026-10-07_09-36_p4_betterhuman`, branch `dev`, repo `/home/rikhat.akizhanov/better/dev/BetterHuman`
(editable dependency on `../BetterRobot` = the BR `dev` worktree, so P1–P3 code is available). Owner asleep: no
intake interview; the plan follows DESIGN.md D5 + roadmap P4 and owner decisions Q5 (no retargeting), Q6 (keep all
studio functionality), Q8 (full SMPL-X and FLAME expression, switchable) in
`/home/rikhat.akizhanov/better/dev/BetterRobot/agent_notes/tasks/2026-10-06_20-13_architecture_plan/DESIGN.md`;
the code scan is `.../subagents/scan_better_human.md` (§3 BR coupling, §5 problems, §7 keep list).

## Objective
BetterHuman becomes a thin, fast layer on BetterRobot's public API: body families are data (`FamilySpec`),
identity is baked once (`Body.bake`), per-frame groups (pose, expression, per-frame scale) stream without
re-baking, outputs are requested (joints / markers / vertices), human terms run on `optim2`, the BH viewer is BH
layers on the BR `Viewer`, and the studio keeps every editor on that core. Benchmarks vs upstream smplx, JSON.

## Facts to check first (M0)
- BH tests/examples run against BR dev today? (`uv run pytest tests -q`): expect breakage from P1–P3 (old BR
  `viewer.RenderContext` in `viz/skinned_mesh.py`, old `optim`/`tasks` use, `Data` API). Record the list.
- BR public surface BH may use: `br.load/Model/ModelValues/with_values`, `kinematics.forward_kinematics_state`,
  `dynamics.*`, `optim2`, `terms`, `viewer` (Layer, SkinnedMeshLayer, SkeletonLayer, …), `lie`, `spatial`. What BH
  needs that is private in BR (`io.builders.kinematic_tree`, `io.ir.IRFrame`, `reduced_coordinates`,
  `JOINT_KIND_CODES`) → add a small public BR builder API in BR (commit on BR dev) rather than keep reaching in.
- Assets available under `/home/rikhat.akizhanov/better/better_human` (SMPL, SMPL-X, MANO, FLAME?, MHR) — list.

## Milestones and checks
M1 `FamilySpec` + `Body.bake`/`forward` (`core/`): spec = template, shape/expression/pose-corrective bases,
   joint regressor, top-K skin weights, kintree, corrective kind enum, per-group stream definitions. SMPL, SMPL-H,
   SMPL-X, MANO, FLAME loaders become tables + a short loader each; MHR keeps its own corrective MLP behind the
   same interface. Check: vertex parity vs the current BH pipeline and upstream smplx (gate 6e-7 m SMPL family,
   1.4e-6 m MHR, `tools/parity/*`); bake once, stream T frames with expression without re-bake; `outputs=`
   skins only needed vertices for markers.
M2 Public-API-only: BH imports no BR private module (contract test); BR gets `io.build_tree(...)`-style public
   builder if needed (BR dev commit).
M3 Terms on optim2 (`better_human.terms`): pose prior (limits + simple GMM/VPoser-free prior), markers / vertex
   landmarks, 2D reprojection, ground contact / penetration, plus inverse dynamics with a contact-force split
   using BR `dynamics`. Check: `examples/03_ik_reach.py` rewritten on optim2; a marker-fit example converges.
M4 Viewer: `better_human.viz.human_layers(body, params_seq, mesh=, skeleton=, contacts=)` → BR layers
   (`SkinnedMeshLayer` with bone poses, `SkeletonLayer`, `ArrowsLayer`); delete `viz/skinned_mesh.py`.
   Check: a clip plays in the BR viewer and records offline through `viewer.record`.
M5 Studio on the BR `Viewer`: editors (segment painting, contact frames, posing, persistence) unchanged in
   behaviour; `StudioSession` observer rewrites T=1 layer arrays and calls `viewer.show()`; picking from
   `better_robot.viewer.tools.picking`. Delete the studio's own server/layers. Check: every studio feature from the
   current `viz/studio/app.py` has a test or a manual checklist in the report; studio tests green.
M6 Benchmarks (`benchmarks/`): {bake, vertices fwd, fwd+bwd, FK, RNEA} × {SMPL, SMPL-X, MHR} × B ∈ {1, 64, 1024,
   8192} × {cpu, cuda (only if FREE)} with upstream smplx baseline, JSON + table in the report.
M7 Docs (`docs/`), `CLAUDE.md`s, mars_aux review, REPORT.md, drop the index line.

## Rules
- Commit on BH `dev` (and BR `dev` for the public builder API only); never push/merge. `trash`, never rm.
- No new dependencies; GPU only when `gpu_free.py` says FREE; servers bind 127.0.0.1 only.
- Retargeting is out of scope (Q5). Keep parity tools and the vertex gate; never weaken them.
