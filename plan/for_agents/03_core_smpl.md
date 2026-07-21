# Order 03 — Core pipeline, `ParametricBody`, SMPL, converters, parity harness

> **Agent log — 2026-07-20:** Completed on `dev`; 13-case parity peaked at
> `1.0412031070661243e-07` (float64) and `7.078051567077637e-07` (float32).
> Both float32 gradchecks and the GPU-1 CUDA smoke passed; BetterRobot stayed
> unchanged and there were no plan deviations. See [`03_results.md`](03_results.md).

Read first: `docs/architecture.md` (whole page — it specifies this order),
`docs/contracts.md`, `docs/design_principles.md`,
`references/design/smplx.md` (§3 math pipeline, §3.4 root wart),
`references/design/BEST_PRACTICES.md` §1–§4,
`references/design/betterrobot_readiness.md` §1–§3 (builder API, batched
`with_values`, the existing BR test to mirror). Ground rules in
`plan/README.md` apply; BR is a dependency, not a fork.

## Context

This is the heart of the rewrite: the shared seven-stage pipeline, the
`ParametricBody` composition, SMPL as the first concrete model, the
SMPL-family classic converters, and the parity harness that makes
correctness a test instead of a hope. SMPL first because it is the smallest
full exercise of every stage (24 joints, betas, correctives, root wart).

## Deliverables

1. **`core/body.py`** — `BodyStructure`, `BodyValues`, `ParametricBody` as
   specified in `docs/architecture.md` (§Core objects): body holds
   `structure`, `values`, `robot` (a `better_robot.Model`); functionally
   immutable; `with_shape(**named_groups)` bakes shape (batch-bearing,
   differentiable) and rebinds BR `ModelValues` via `Model.with_values`;
   `fk(q)` delegates to BR; `vertices(q)` (and `vertices_from_data(data)`)
   run correctives + skinning. Keep it thin — pipeline math lives in
   `core/pipeline.py`.
2. **`core/pipeline.py`** — the shared math, plain functions on tensors:
   - `blend_shapes(coeffs, basis)` (serves shape and expression),
   - `regress_joints(regressor, v_shaped)` (sparse-aware),
   - `pose_feature_rotmat_minus_identity(q_joint_quats)` (SMPL's `(R−I)`,
     root excluded),
   - `linear_corrective(feature, posedirs)`,
   - `lbs(v_rest, skin_transforms, weights_vk, indices_vk)` — padded
     `[V,K]` influences, never dense `[V,J]` gathers,
   - skinning transforms from BR world joint poses + rest joints
     (`T_world · T_rest⁻¹`, the smplx `A` matrices, but built from BR
     `Data.joint_pose_world`).
   All batched (leading batch axes, trailing event axes), dtype/device
   preserving, autograd-clean.
3. **`core/classic.py`** — shared converter algebra: axis-angle ↔
   scalar-last quaternion (batched), and the SMPL-family root-wart algebra
   (pelvis pivot `J₀(β)` + post-skinning `transl` ↔ free-flyer `q[0:7]`),
   as pure functions; see `references/design/anny.md` §3.3 and
   `references/design/SOMA-X.md` §3.3 for worked matrix-form references.
4. **`bodies/smpl.py`** — `SMPL(model_path=None, gender="neutral",
   num_betas=10, dtype=torch.float32, device=None)`:
   - loads via order 02's io; builds the BR tree with
     `build_kinematic_tree_model` (free-flyer root + 23 spherical joints,
     parents from `kintree_table`, parent-relative offsets from neutral
     `J`). **Pass `preserve_joint_order=True`** — BR's default DFS
     topo-sort reorders joints and would silently misalign `weights`,
     `posedirs`, and `J_regressor`, which are all native-order. Standing
     convention: BR joint `j+1` ↔ native joint `j` (universe at 0); add an
     explicit alignment test (BR joint names in native order);
   - **root-frame convention** (docs/architecture.md §Classic conversion,
     binding): at shape-bake time, pelvis-center the rest data — shift
     `v_shaped` and rest joints by `−J₀(β)` so the pelvis is at the origin
     — with zero free-flyer root placement. Then `q[0:3]` is the pelvis
     world position and `transl = q[0:3] − J₀(β)` with no
     rotation-coupled term. Do NOT keep the MPI-world template as the LBS
     rest frame; that changes the conversion to `J₀ + transl − R₀·J₀` and
     breaks parity exactly on the root-wart grid case;
   - `SMPLClassic` dataclass (`global_orient`, `body_pose`, `transl`,
     each optional/batched) + `from_classic`/`to_classic` (exact inverses;
     shape-dependent via the body's own `J₀`);
   - no pipeline math in this file — data binding and policy only.
5. **`bodies/registry.py`** — decorator registration +
   `better_human.load(name, **cfg)`; public API exported from
   `better_human/__init__.py`: `SMPL`, `ParametricBody`, `load`.
6. **Parity harness** — `tools/parity/smplx/`: a pinned uv project
   installing the original `smplx` package; a dump script producing golden
   `.npz` (vertices+joints, float64) for a fixed grid: zero pose; ~8
   random poses (seeded, axis-angle magnitudes up to ~1.5 rad); betas ∈
   {0, ±2 sampled}; nonzero `global_orient` + `transl` (the root wart
   case). SMPL goldens are small (~0.17 MB/case at float64) — commit the
   neutral-gender set and record its total size. (SMPL-X goldens in order
   04 are larger and regenerate-on-demand; the split is deliberate.)
   `tests/parity/test_smpl_parity.py` compares our
   `from_classic → vertices` against goldens: float64 ≤ 1e-6 m; float32
   tolerance measured, pinned, and recorded in the results file.
7. **Tests** (beyond parity): converter round-trips both directions
   (contract tolerances); float32 gradchecks `betas → vertices` and
   `q → vertices` on a small vertex subset (document the subset);
   batched-shape checks (`betas (B,10)` × `q (B, nq)` and broadcast
   `q (nq,)`); a CUDA smoke run of FK+vertices (pin a free GPU).

## Constraints

- No `nn.Module`, no `nn.Parameter` state, no batch size in constructors
  (SMX anti-patterns). No magic constants — counts come from loaded shapes.
- One FK traversal per query (never recompute the chain for skinning).
- If a BR gap blocks you (e.g. builder ergonomics), the fix goes in BR on
  its `dev` with a note in the results file — no duplicated robotics code
  in BetterHuman.

## Acceptance

- SMPL parity vs oracle goldens: float64 ≤ 1e-6 across the whole grid;
  float32 number reported.
- Round-trips, gradchecks, batching, CUDA smoke green; full gate green;
  ruff clean.
- `better_human.load("smpl", ...)` returns a working body.

## Results file

`plan/for_agents/03_results.md`: parity table (per grid case, max error),
float32 tolerance chosen and why, gradcheck runtimes, any BR-side changes,
deviations.
