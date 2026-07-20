# Plan — BetterHuman rewrite: scaffold, io, SMPL, SMPL-X, mass, viewer

State at planning time (2026-07-20, branch `dev`): the repository was just
renamed `better_human` → `BetterHuman` and the full rewrite begins. Twelve
reference analyses plus a BetterRobot readiness audit live in
`references/design/` (local, untracked — read them; they are the evidence
base). The audit's verdict: the riskiest assumption — betas → batched,
differentiable `ModelValues` → FK/IK/dynamics with gradients — **already
works in BetterRobot and is tested**
(`BetterRobot/tests/optim/test_parametric_model_values.py`; note the test
is a small revolute model in float64 — it proves the mechanism, and order
03's float32 gradchecks on the real spherical/free-flyer tree complete the
validation); BR also ships
`make_smpl_like_model` (24-joint preset, nq=99/nv=75), spherical+free-flyer
joints in every lane including Warp, `Inertia.from_mesh` (differentiable,
batched), and `SwingTwistLimitResidual`. Everything left is human-mesh
machinery that is legitimately ours: LBS/blendshape/corrective pipeline,
betas→ModelValues regressors, weight-file io, converters, a skinned-mesh
viewer mode.

Design is settled in `docs/vision.md`, `docs/architecture.md`,
`docs/contracts.md`, `docs/design_principles.md`. The old code
(`src/better_human/*`, `scripts/`, `tests/`, `temp/`) is dead on arrival —
order 01 deletes it; nobody builds on it.

## The six orders

| Order | What | Acceptance in one line |
|---|---|---|
| `for_agents/01_scaffold.md` | Wipe old code; uv project; BR path dep; package + test skeleton | `uv run pytest` green incl. BR smoke FK; old tree gone |
| `for_agents/02_io_weights.md` | npz schemas, validated loader, path resolution, chumpy→npz converter CLI | real SMPL/SMPL-X weights convert + load + validate; skip-if-absent discipline |
| `for_agents/03_core_smpl.md` | core pipeline + `ParametricBody` + SMPL + converters + parity harness | SMPL vertices/joints ≤1e-6 (f64) vs smplx oracle goldens; round-trip + gradchecks green |
| `for_agents/04_smplx_partial.md` | SMPL-X + generic partial-model reduction (hands/face/num_betas) | full SMPL-X parity; partial spawns match oracle-with-rest-DOFs; no new classes |
| `for_agents/05_mass_dynamics.md` | skin-weight mesh segmentation → `Inertia.from_mesh` → `body_inertias`; dynamics smoke | total mass/COM plausible; rnea/crba run batched + differentiable w.r.t. betas |
| `for_agents/06_viewer_demo.md` | skinned-mesh render mode on BR viewer; demo scripts; README rewrite | posed SMPL-X deforms live in viser demo; unit-tested mode update path |

Run in order; each order ends with the full test gate green and docs true.
01→03 are strictly sequential. 04 needs 03; 05 and 06 both need 03 and are
independent of each other (may run in either order, not concurrently in one
tree).

## Decisions taken in this plan (owner may veto before launch)

1. **v1 abstraction budget is SMPL-family-sized.** The stage seams
   (`ShapeModel`/`JointRegressor`/`PoseCorrective`/`ParameterTransform`)
   are documented in `docs/architecture.md` but ship as plain functions and
   dataclasses; they become protocols when a third model family (MHR or
   Anny) lands. No speculative interfaces.
2. **Golden-file parity, not live-oracle parity.** A pinned uv venv under
   `tools/parity/` runs the original `smplx` package once to dump goldens;
   CI tests compare against committed golden tensors and skip (visibly)
   when local weights are missing. Originals never become runtime deps.
3. **Chumpy conversion via pickle-stub shim** (fake `chumpy` module at
   unpickle time, SOMA-X pattern) in the main venv; the museum-NumPy
   converter venv is the documented fallback only if the shim fails on real
   files.
4. **Weights resolution**: explicit `model_path` arg, else
   `$BETTERHUMAN_MODELS_DIR/<family>/<file>`, else actionable error. The
   repo-local `models/` folder (untracked, already populated on this host)
   is the conventional local root — orders must NEVER modify or delete
   `models/`.
5. **Public API**: per-model classes (`SMPL`, `SMPLX`) + registry
   `better_human.load(name, **cfg)`. Package import name stays
   `better_human`.
6. **Keypoint/landmark output heads (OpenPose-style 127/144 joints) are
   deferred** — they are an output convention, not kinematics (SMX §3.6).
   Roadmap, not v1.
7. **`temp/` is deleted with the old code.** Its experiments are superseded
   by the parity harness; git history keeps them recoverable from `main`.

## Ground rules

1. **Read `docs/design_principles.md`, `docs/architecture.md`, and
   `docs/contracts.md` before writing code.** They are binding; the
   reference analyses in `references/design/` are the evidence behind them.
2. **No backward compatibility.** This is a ground-up rewrite on `dev`;
   nothing references the old API. No shims, no aliases.
3. **The safety net only grows.** Parity goldens, converter round-trips,
   float32 gradchecks, and schema-validation tests are never weakened to
   land a change.
4. **BetterRobot is a dependency, not a fork.** Gaps found in BR (missing
   builder ergonomics, viewer hooks) are reported in the results file and,
   if small and general, fixed in BR on its `dev` — never worked around by
   duplicating BR functionality in BetterHuman. Do not modify BetterRobot
   without listing the change in the results file.
5. **CUDA is agent-runnable here** (8 RTX 6000 Ada GPUs; pin one via
   `CUDA_VISIBLE_DEVICES`, 2/3/7 usually free). Device-conditional code is
   validated on both CPU and CUDA or explicitly reported as not.
6. **Docs stay true per order** — every page and docstring an order
   falsifies is updated in the same order.
7. **Honest results files.** Each order writes `for_agents/NN_results.md`:
   delivered, deviations-as-findings, exact test counts, parity numbers as
   measured (never rounded to "green").
8. **Subagents run on Opus.** Everything on `dev`; no feature branches, no
   worktrees unless asked.
9. **Never touch `models/` (weights) or `references/`** except reading.

## Roadmap beyond these orders (not scheduled)

MANO / SMPL-H / FLAME (same family, mostly schema + tables); STAR / SUPR
(quaternion pose features, β-dependent correctives — first real test of the
corrective seam); MHR (authored skeleton, parameter transform, MLP
correctives — promotes the seams to protocols); Anny (CC0 default model,
phenotype shape space); keypoint/landmark heads; human residuals on BR optim
(swing/twist tables, pose priors, mink's task taxonomy); human→robot
retargeting (`better_human.retarget`); cross-model conversion (barycentric
wrap + staged fitting); differentiable measurements (`better_human.measure`).
