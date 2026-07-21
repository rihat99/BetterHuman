# Order 03 — the MHR body

> **Implementation log (2026-07-21): complete.** Added all seven data-driven
> LODs, the 132/131-DoF revolute+mimic tree, shape/scale bake, MLP correctives,
> physical-part mass, and lossless classic conversion. Static FBX inverse binds
> remain authoritative (the plan's recompute-bind instruction was rejected by
> parity); LOD1 `combined_00` vertex max is `2.26498e-6 m`. Full suite:
> **199 passed, 4 skipped**. Details and deviations: `03_results.md`.

`bodies/config_mhr.py` + `bodies/mhr.py`: spawn, BetterRobot tree, shape bake,
forward path, classic converters. Depends on orders 01 (converted npz +
goldens) and 02 (core stages). The SMPL-X module is the style template.

## Ground rules (binding)

- Read `docs/design_principles.md`, `docs/architecture.md`, `docs/contracts.md`,
  `references/design/MHR.md`, `references/design/momentum.md` first. Also read
  `~/.claude/handoffs/betterhuman_mhr_context.md` — it carries the BR
  capability verification with file:line citations.
- **No BetterRobot modifications.** If BR blocks something, measure, report in
  the results file, and stop — do not work around by duplicating BR
  functionality in BetterHuman.
- Never modify `models/MHR/assets/` or `references/`.
- End green: full BetterHuman suite passes.
- Write `plan/for_agents/03_results.md`: delivered, deviations, exact test
  counts, measured numbers.

## Owner-settled design (do not re-litigate)

1. **q = free-flyer root + the 125 named MHR rotation params** (nq = 132,
   nv = 131). Typed revolute sub-nodes, **no spherical joints** in this tree.
2. **Proportions are shape**: the 71 translation params + 2 hand scales form a
   73-wide `proportions` group in `with_shape` — never q.
3. All 7 LODs, default `lod=1`. Spawn flags: `lod`, `use_expression=True`,
   `use_correctives=True`, `compute_mass=True`, plus `device`/`dtype`.
4. Everything topology-related is **data-driven from the converted npz** — no
   hand-written 127-joint tables in Python.

## 1. Config (`bodies/config_mhr.py`)

Static conventions only: unit/up/forward (as recorded by the converter),
`native_pose_format="euler_xyz"`, segmentation part-grouping reference,
density-table name, LOD list, asset filename pattern. No landmark tables
(MHR ships none; roadmap). Reuse/extend `BodyConfig` only where its fields
genuinely fit — a small MHR-specific config dataclass is fine if cleaner.

## 2. BetterRobot tree construction

Built from the npz via `ModelBuilder`'s **public typed helpers**
(`add_free_flyer_root`, `add_revolute_x/y/z`, `add_fixed`, `add_frame` —
see `io/parsers/programmatic.py:184-357`; avoid `_push_joint`).

- **Free-flyer root** for momentum joint 0 (its 6 root channels).
- **Per driven joint**: first sub-node placement = `T(base_offset) · Rpre`
  (origin quaternions **pre-normalized** — `build_model` does not normalize
  build-time origins), then revolute sub-nodes in momentum application order
  **rz, ry, rx** for the driven rotation channels only. Intermediate sub-nodes
  get identity placements and zero mass.
- **Spine — the 12 multi-driver rows** (`c_spine0..3.{rx,ry,rz}`, each
  `1.0·flexible + a·shared0 [+ b·shared1]`): same-axis stacked sub-nodes —
  `Rx(flex + a·t0 + b·t1)` becomes `Rx(flex)·Rx(a·t0)·Rx(b·t1)`, which is
  exact. One sub-node per driver: the flexible param's node is public; each of
  the 6 shared params (`spine_twist/lean/bend{0,1}`) gets exactly **one public
  host sub-node** at its first row (its q is the *scaled contribution*, e.g.
  `0.01·spine_twist0` — document this q semantic in the class docstring) and
  its other spine sub-nodes **mimic the host** with ratio multipliers
  (`0.12/0.01`, …). ≈ +18 sub-nodes; this is what makes the 125-param
  bijection exact with single-source mimic only.
- **Fan-outs** (8 limb twists ×5 rows, `neck_twist` ×3, `scale_knee_knock`
  ×2): host = the 1.0-weight (or first) row as a public revolute; other rows
  mimic with ratio multipliers. BR constraints (verified): mimic source AND
  target must be scalar Euclidean joints; chained mimic by name works; cycle
  detection exists.
- **Undriven joints** (14): fixed joints or frames — both have batchable
  placements through `ModelValues`. Expose **all 127 native joint world
  transforms in native order** (frames or sub-node poses) for parity and user
  access.
- `preserve_joint_order=True` with the converter-asserted topological order —
  note BR silently *repairs* non-topological input, so the converter assertion
  is the guard.
- Expected scale: ~175 moving sub-nodes (125 public + ~50 mimic targets).

## 3. Shape bake — `with_shape(identity=…, expression=…, proportions=…)`

Widths 45 / 72 / 73; named groups, batched, differentiable:

- `v_shaped = template + basis @ [identity; expression]` (mesh-only — the
  converter asserted blendshapes never move joints).
- **Placements**: base translation offsets + (sparse translation block of the
  parameter transform) @ proportions — parent-frame, meters, written batched
  through `Model.with_values(joint_placements=…)` (BR normalizes quats there).
- **Hand scale**: the 2 `sc` channels (wrists) with momentum's `S = exp2(sc)`.
  Per-joint accumulated scale vector `s_j` = product of ancestor `2^{sc}`
  (≠ 1 only in the wrist subtrees). Fold into (a) descendant placement
  translations (× accumulated parent scale) and (b) the per-joint skinning
  scale passed to the order-02 scaled LBS. The skinning transform
  `world ∘ inverse(bind)` is invariant to any constant right-factor, which is
  why baking works — verify against goldens, not by trusting this note.
- **Bind poses**: recompute by one rest-pose FK during bake (batched over
  shape). Rest q = free-flyer identity + zeros… **verify** whether momentum's
  bind pose is the zero-parameter pose; the converter/results of order 01
  record this.
- **Mass**: argmax part meshes from grouped skin weights (config grouping) on
  `v_shaped` → `body_inertias`, densities via the existing named tables;
  respects `compute_mass=False`.
- `use_expression=False`: drop the expression group/columns (reference
  `exclude_expression`). `use_correctives=False`: skip loading the corrective
  tensors entirely (they are 21 MB–2.6 GB; LOD0 needs ~2.6 GB — document).

## 4. Forward path

FK → full per-channel values via **BR's**
`reduced_coordinates.expand_configuration` (public q → all 157 channel values
incl. mimic targets; do not duplicate this matrix in BetterHuman) → gather the
corrective joint list (`joints[2:]` as recorded by the converter) → 6D feature
(order 02) → MLP offsets if enabled → generalized skinning with bind poses +
scale vector → vertices. Functional throughout; no in-place aliasing.

## 5. Classic converters

`MHRClassic(identity_coeffs[45], model_parameters[204], expression[72] | None)`
— mirrors the reference forward signature exactly.

- `from_classic`: split the 204-vector **by channel type** — root 6 →
  free-flyer (the ×10 root-translation factor and cm→m come from the transform
  weights themselves; Euler→quat for root orientation), 125 rotation params →
  public q entries (apply the documented scale factors for spine host
  sub-nodes), 71 translation + 2 scale params → the `proportions` group. The
  conversion is shape-aware (like SMPL's pelvis-dependent conversion): decide
  and document whether it returns `(q, proportions)` or applies `with_shape`.
- `to_classic`: exact inverse. Round-trip contract: **exact bijection both
  directions** (float64 ≈ 1e-12); root Euler↔quaternion exact on the principal
  domain (same caveat class as SMPL axis-angle — document).

## Acceptance

- Spawn + forward run on **all 7 LODs** (report LOD0 memory).
- Classic round-trips exact (f64 ≈ 1e-12; f32 measured + pinned).
- Masses positive, total plausible (report the neutral total in kg at
  density 1000).
- **Early smoke parity**: one LOD1 golden case within 1 mm (loose, to catch
  gross composition errors before order 04 tightens).
- Mimic coupling behaves: posing a twist param moves its helper rows in ratio
  (quick test; full coverage in order 04).
- Full suite green; ruff clean.
