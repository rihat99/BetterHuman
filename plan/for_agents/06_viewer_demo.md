# Order 06 — Skinned-mesh viewer mode, demos, README

> **Agent log — 2026-07-20:** Completed on `dev`; Viser 1.0.30 sustained
> 59.7 server-side updates/s and the full gate passed with 102 tests. The mode
> stayed in BetterHuman and uses BR's public same-name `add_mesh` fallback;
> the unexposed Viser vertex setter is documented in [`06_results.md`](06_results.md).

Read first: `docs/architecture.md` (§Visualization),
`references/design/betterrobot_readiness.md` §6 (exact BR viewer gap and
the `add_mesh` backend hook), BR's `src/better_robot/viewer/` render-mode
API and its nearest `CLAUDE.md`. Ground rules in `plan/README.md` apply.
Depends on order 03 (04 enriches the demo but is not required to start).

## Context

BR's viewer re-poses rigid geometry only; a parametric human needs its
LBS-posed vertices re-uploaded per frame. This order adds that render mode,
ships the first real demos (replacing the old `*_viser_demo.py` scripts),
and rewrites the README into an honest front page for the new library.

## Deliverables

1. **`src/better_human/viz/skinned_mesh.py`** — a render mode following
   BR's render-mode protocol (`attach`/`update` shape): `attach` uploads
   the body's rest mesh via the backend's `add_mesh`; `update(data)`
   recomputes vertices from the body + `Data` and pushes them. Check the
   installed viser version for an efficient vertex-update path before
   falling back to re-adding the mesh each frame; record which path viser
   offers. Follow BR's optional-dependency rule: viser imports stay at the
   integration boundary.
   - If the mode can be implemented cleanly as a BR-side contribution
     instead (a generic "deforming mesh" mode taking a vertices callback),
     prefer that and note it in the results file — BR is the dependency;
     BetterHuman keeps only the human-specific glue.
2. **`examples/`**:
   - `01_smpl_shape_pose.py` — viser sliders for betas (re-bake via
     `with_shape`) and a few joints (`q` via `Model.integrate`), live
     deforming mesh;
   - `02_smplx_partial.py` — spawn full vs `use_hands=False, use_face=False`
     side by side (order 04);
   - `03_ik_reach.py` — the payoff demo: BR `solve_ik` driving the SMPL
     wrist frame to a draggable target, skinned mesh following — the
     robotics-core story in one script.
3. **Unit tests**: drive the render mode against a fake/recording backend
   (no viser server): attach uploads once, update pushes new vertices,
   dtype/device respected. No screenshot tests.
4. **README rewrite** — the real front page: what BetterHuman is (one
   paragraph from `docs/vision.md`), install via uv (incl. the BR path
   dependency caveat during alpha), weight download + `models/` layout +
   `$BETTERHUMAN_MODELS_DIR`, converter usage, a minimal code example
   (spawn → `with_shape` → `vertices`), supported-models table (SMPL,
   SMPL-X + spawn flags), pointer to `docs/` and the examples.

## Acceptance

- Examples run on this host against local weights (visual check by owner;
  agent verifies they launch, render one frame, and shut down cleanly —
  e.g. a `--frames N --headless`-style guard or timeout, documented).
- Render-mode unit tests green; full gate green; ruff clean.
- README contains no stale conda-era or old-API text.

## Results file

`plan/for_agents/06_results.md`: viser update path used (and version),
frame-rate observed for the SMPL demo, whether the mode landed in BH or BR,
deviations.
