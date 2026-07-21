# Order 01 — pymomentum environment, asset converter, parity goldens

> **Implementation log (2026-07-21, `dev`):** Complete. All 7 converter
> archives and 48 float32 goldens were generated and verified. Important source
> corrections: native joint 0 is fixed `body_world` and joint 1 is `root`; 350
> means `.model` assignments, while the sparse transform has 295 unique driven
> rows / 368 entries; the static FBX inverse bind is authoritative (zero-FK
> differs by `4.57763672e-05 cm`). PyMomentum exposes neither FBX axes nor pose
> constraints, so those are parsed from the read-only FBX/`.model` inputs.

Everything that needs `pymomentum` lives in this order, under `tools/`, behind
a pinned environment. The shipped library never imports it.

## Ground rules (binding)

- Read `docs/design_principles.md`, `docs/architecture.md`, `docs/contracts.md`,
  `references/design/MHR.md`, `references/design/momentum.md` before coding.
- **No BetterRobot modifications.** Report BR gaps in the results file.
- **Never modify `models/MHR/assets/` or `references/`** (read-only). Converter
  output goes to `models/MHR/converted/` (untracked, inside gitignored `models/`).
- End green: full BetterHuman suite passes (`uv run pytest tests/ -q`).
- Write `plan/for_agents/01_results.md`: delivered, deviations-as-findings,
  exact commands, measured numbers as measured.

## Inputs on disk

`models/MHR/assets/`: `compact_v6_1.model` (plain text, "Momentum Model
Definition V1.0"), `lod{0..6}.fbx`, `corrective_blendshapes_lod{0..6}.npz`
(2.6 GB … 21 MB), `corrective_activation.npz`, `mhr_model.pt` (TorchScript,
LOD1, pure torch). Reference code: `references/parametric_body/MHR/` (the
`mhr` package + tools) and `references/parametric_body/momentum/`.

Measured ground truth (verify, don't re-derive): 127 joints, 113 driven, 350
driven DOFs = 157 rotation + 191 translation + 2 scale channels; 204 model
params; only the 12 spine rows (`c_spine0..3.{rx,ry,rz}`) have 2–3 drivers;
`root.tx = 10.0 * root_tx` (×10); units centimeters.

## 1. Pinned environment (`tools/parity/mhr/`)

- Try **pixi** first (installs user-level; `references/parametric_body/MHR/pixi.lock`
  is the guide — pymomentum ≥ 0.1.90 + torch + numpy). Fallback: conda env from
  conda-forge `pymomentum` (box has miniconda). **pymomentum is NOT on PyPI**
  (the PyPI package of that name is unrelated) — do not try uv/pip.
- Pin whichever works: commit the lockfile/env spec + a README with exact
  commands (pattern: `tools/parity/smplx/README.md`).
- Sanity check: load LOD6 (smallest FBX) via
  `pym_geometry.Character.load_fbx(lod6.fbx, compact_v6_1.model, load_blendshapes=True)`.

## 2. Converter CLI (`tools/convert/mhr/`)

Dumps **one self-contained npz per LOD** to `models/MHR/converted/mhr_lod{n}.npz`.
All lengths converted **cm → meters**. Store f32 as stored upstream. Contents:

- **Skeleton** (identical across LODs, duplicated per file for self-containment):
  joint names (127), parents, pre-rotation quaternions (**scalar-last,
  normalized** — BR does not normalize build-time origins), translation offsets.
- **Parameter transform**: nonzero `(row, col, weight)` triplets of the
  `[889, 321]` matrix, the 204 param names, per-row `(joint_index, channel)`
  labels where channel ∈ {tx,ty,tz,rx,ry,rz,sc}. Length-bearing weights
  (translation rows) converted to meters.
- **Mesh**: vertices `(V,3)`, faces `(F,3)`; **skinning**: indices + weights
  `[V, ≤8]` padded.
- **Blendshapes**: basis split identity `(V,3,45)` / expression `(V,3,72)`, meters.
- **Correctives**: first-layer sparse indices/values + mask (from
  `corrective_activation.npz`), output layer from
  `corrective_blendshapes_lod{n}.npz` reshaped `[3000, V·3]` and scaled cm→m;
  the exact corrective joint list (reference uses joints `[2:]`, 125 joints)
  and feature ordering, recorded as index arrays.
- **Limits, parameter sets, `tpose`** pose-constraint values from the text
  `.model`.
- **Segmentation**: joint-name → part-group table (adapt the grouping from
  `references/parametric_body/MHR/tools/mhr_create_segmentation/example.py`).
- **Conventions**: up/forward axes read from the character (don't guess), native
  unit, native pose format (`euler_xyz`).

**Converter assertions (fail loudly, all of them):**

1. exactly 350 transform rows; row classification matches 157/191/2;
2. blendshape columns (204:321) of the transform are identically zero
   (identity/expression must not move joints);
3. joint order is topological (`parents[i] < i`) — required for BR
   `preserve_joint_order`;
4. pre-rotation quats unit-norm; skin weights rows sum to 1 (or renormalize
   and record);
5. **FK composition confirmed against source**: read
   `momentum/character/joint_state.h` (and the pymomentum tensor path) and
   record in the results file the exact per-joint order — expected
   `world = parent · T(offset + t_params) · Rpre · R · S`,
   `R = Rz(rz)·Ry(ry)·Rx(rx)`, `S = exp2(sc)` — including where the fixed
   `translationOffset` enters relative to the parameter translations. Parity
   depends on this; do not trust the plan, verify.

## 3. Goldens (`tools/parity/mhr/dump_mhr.py` → `tests/parity/goldens/mhr_lod{n}/`)

Run the **reference `mhr` package** (import read-only from
`references/parametric_body/MHR`, e.g. via `sys.path`) inside the pinned env.
Per case save: all inputs, output vertices, `skel_state` `(127, 8)`
(trans3+quat4+scale1), flags. Native units recorded in the file (tests
convert). Oracle is **float32-only** (pymomentum stores floats) — dump f32 and
say so in the README.

LOD1 grid ≈ 30 cases, seeded RNG, covering at minimum:

- all-zeros, correctives on AND off;
- the `.model` `tpose` (note `l/r_elbow_bend = -0.5`);
- random poses within `[Limits]` (several seeds), zero shape;
- random identity coeffs (e.g. U[-2,2]), zero pose;
- random expression only; expression excluded (`None`);
- random proportions within limits (the 71 translation params);
- **hand-scale extremes** (`scale_l_hands`/`scale_r_hands` at limit bounds —
  the only true scale channels; a known trap);
- root translation+rotation exercised (the ×10 factor is a known trap);
- fully-random combined cases (several seeds).

Other LODs: 3 smoke cases each (zeros; one fully random; one
proportions+hand-scale). LOD0 cases are ~880 KB each — keep to 3.

## Acceptance

- Converter runs end-to-end for **all 7 LODs on this box**; every assertion
  passes; per-LOD npz sizes reported.
- Goldens exist for the full grid; a quick self-check reloads one golden and
  re-runs the reference to confirm reproducibility.
- Environment pinned + documented; nothing under `models/MHR/assets/` or
  `references/` modified.
- Full BetterHuman suite still green (this order adds no library code).
