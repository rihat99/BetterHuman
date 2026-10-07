# Context for M1 subagents (BetterHuman rewrite on the new core)

Repo: `/home/rikhat.akizhanov/better/dev/BetterHuman` (git worktree, branch `dev`, HEAD fcb425d). Run everything
from that directory with `CUDA_VISIBLE_DEVICES="" uv run ...` (CPU only; no GPU jobs). BetterRobot is the editable
dependency `../BetterRobot` (its `dev` worktree): read it, never edit it. No new dependencies. Never `rm`: move
files away with `python3 ~/.claude/skills/autoresearch/scripts/trash.py <paths>`. Do not commit; the main agent
commits. Never bind a server. Never skip, xfail or loosen a test to get green; if a gate fails, report the number.

Read first: `agent_notes/tasks/2026-10-07_09-36_p4_betterhuman/M1_DESIGN.md` (the API contract), then the new code:
`src/better_human/core/{spec,rig,body,pipeline}.py`, `src/better_human/families/{smpl_family,classic,tables}.py`,
`src/better_human/__init__.py`. The old code in `src/better_human/bodies/` is the reference for behaviour and is
going to be trashed at the end of M1 (do not import from it in new code or tests).

Key facts of the new core:
- `FamilySpec` is data (template, faces, top-K skin weights, identity/stream blend groups, `Corrective`, `Parts`,
  landmarks, `meta`) plus one behavioural member, `rig` (`core/rig.py`: `Rig` protocol, `RegressedRig`).
- `Body(spec, dtype=, device=, mass=, density=, contact_frames=, stream_joints=)` builds the BR model once;
  `body.bake(**identity) -> Identity` (v_shaped pelvis-centred, rest_joints, pelvis, model with values);
  `identity.forward(q, stream=) -> HumanState`; `state.vertices()`, `state.joints`, `state.joint_poses`,
  `state.points(ids)`, `state.landmarks()`, `state.kinematics` (BR `KinematicState`), `state.model`.
- The SMPL family passes its parity gate on this core (worst 5.1e-7 m over the 29 SMPL-X goldens, float32 and
  float64, expression streamed per frame with exact joints). Script: `/tmp/claude-1003/-home-rikhat-akizhanov-better/d9220856-907e-4a64-b1a1-f6cd72ec6355/scratchpad/parity_smplx.py`.
- Assets: `models/` symlink in the repo (SMPL, SMPL-X, SMPL-H `smplh/smplh/{neutral,male}/model.npz`, MANO
  `mano/MANO_{LEFT,RIGHT}.npz`, MHR `MHR/converted/mhr_lod{0..6}.npz`, `MHR/assets/mhr_model.pt`). No FLAME asset.
- `better_robot` public surface to use: `better_robot.Model`, `ModelBuilder`, `better_robot.io.build_model`,
  `Model.expand_configuration(q)` (new, public), `better_robot.kinematics.forward_kinematics_state`,
  `better_robot.lie.{so3, se3}`, `better_robot.spatial.Inertia`, `better_robot.dynamics` (`rnea` etc. on the
  `Data` API still exist; prefer `better_robot.dynamics.state.inverse_dynamics(model, state, v, a)` for new code).
  Never import `better_robot.io.builders.*`, `better_robot.io.ir`, `better_robot.data_model.*` privates.
- Style: ruff (line length 120), type hints, docstrings with shapes, no comments that narrate. Tests are pytest;
  skip only when a licensed asset is absent (existing pattern in `tests/parity/*`).
