# Order 05 — Mass properties from the shaped mesh; dynamics on the human tree

> **Agent log — 2026-07-20:** Completed on `dev`; neutral SMPL measured
> `73.43093872070312` kg with `1.778485059823366%` partition error, and the
> full gate passed with 98 tests. The shipped canonical segmentation and the
> assumed BetterRobot topology validation both deviated from the plan; see
> [`05_results.md`](05_results.md).

Read first: `docs/architecture.md` (§Mass and dynamics),
`references/design/betterrobot_readiness.md` §4 (what `Inertia.from_mesh`
gives and what is ours), `docs/contracts.md`. Ground rules in
`plan/README.md` apply. Depends on order 03 (SMPL); works for every body
that has skinning weights, so implement it in `core`, not per model.

## Context

The vision says "closer to robotics than graphics": dynamics on a human
body must work out of the box, with inertias that follow shape. BR ships
the differentiable, batched mesh integrator (`Inertia.from_mesh`) and all
dynamics algorithms; BetterHuman supplies the vertex→body-part segmentation
and density policy, and wires per-part inertias into `with_shape`.

## Deliverables

1. **`core/mass.py`**:
   - `segment_vertices(weights_vk, indices_vk) -> part id per vertex`
     (argmax skinning influence, one part per BR body). **Before writing
     this from scratch, inspect `models/star/smpl_3d_segmentation.npy`** —
     it ships a canonical SMPL 24-part segmentation (`body_vertices`,
     `body_faces`, and `border_vertices`, i.e. the exact cut loops) — and
     `models/star/body_densities.json` (named per-part density models:
     Naive=1010, Dempster, Chandler, …). Prefer these assets for SMPL
     (read-only!) with the argmax path as the model-agnostic fallback;
     cross-check argmax against them and report the diff;
   - `part_meshes(v_shaped, faces, part_ids)` → per-part **closed**
     triangle meshes: keep faces whose vertices agree on the part
     (majority rule for mixed faces — document the rule), then cap each
     boundary loop (the `border_vertices` loops for SMPL) with a triangle
     fan about the loop centroid. `Inertia.from_mesh` validates
     closedness/winding — let its checks be the arbiter, don't duplicate
     them;
   - `body_inertias_from_shape(...)` → `(B?, nbodies, 10)` packed
     inertias via `Inertia.from_mesh`, differentiable w.r.t. `v_shaped`;
   - density: single scalar default `1000.0` kg/m³, overridable with a
     per-part tensor (the per-segment Dempster-style table is a data
     option, not new code paths).
2. **Wire into shape baking**: `with_shape` (order 03) now also computes
   `body_inertias` and passes them through `Model.with_values`. Make the
   mass step optional at spawn (`compute_mass=True` default) only if its
   cost is material — measure first, add the knob only with evidence
   (design principle 1).
3. **Tests** (`tests/core/test_mass.py`, `tests/dynamics/`):
   - conservation: Σ part masses ≈ whole-mesh `from_mesh` mass of the same
     shaped body (tolerance reflects cap-fan closure error — measure and
     pin it); total ≈ 70–90 kg for neutral SMPL at ρ=1000 (record the
     number);
   - COM plausibility (inside the body, near pelvis height at rest);
   - dynamics smoke on the human tree: `rnea` (gravity-only τ finite,
     zero-velocity consistency), mass matrix via `crba` SPD, batched over
     `(B,)` bodies with different betas;
   - differentiability: float32 gradcheck of `betas → total mass` and a
     `betas → rnea τ` backward producing finite nonzero grads;
   - CPU + CUDA (pin a free GPU).
4. **Docs**: extend `docs/architecture.md` §Mass and dynamics with the
   implemented segmentation/closure rule and measured accuracy; record the
   density default in `docs/contracts.md` (§Units) if it becomes a
   contract.

## Constraints

- No trimesh dependency for the inertia path — BR's integrator is
  torch-only; mesh capping is ours and stays torch/numpy-simple.
- Segmentation runs at `with_shape` time on shaped vertices (inertias are
  shape-dependent); the part assignment itself is shape-independent and
  computed once per spawn.

## Acceptance

- Conservation, plausibility, dynamics smoke, and gradcheck tests green on
  CPU and CUDA; measured mass/COM numbers in the results file.
- Full gate green; ruff clean; docs updated in the same change.

## Results file

`plan/for_agents/05_results.md`: measured masses/COM, closure-error
numbers, per-part table (name, mass at neutral), any BR-side findings,
deviations.
