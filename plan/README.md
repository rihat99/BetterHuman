# Plan — MHR support

Add Meta's **MHR** (Momentum Human Rig, arXiv 2511.15586) as the third body
model after SMPL and SMPL-X: reimplemented in our style on BetterRobot, no
`pymomentum` at runtime, exact mesh/joint parity, `from_classic`/`to_classic`,
all seven LODs, masses/dynamics, and a performance check against the
reference. Everything on `dev`; alpha, no back-compat.

MHR is not another SMPL — it is an animation rig: 127-joint skeleton, a sparse
linear map from 204 named parameters to per-joint DOFs, mostly 1-DoF Euler
rotation channels, translational bone-lengthening channels, two true
uniform-scale channels (hands), mesh-only blendshapes (45 identity + 72
expression), and an **MLP pose corrective** instead of linear posedirs.
Background studies: `references/design/MHR.md` and `momentum.md`.

## Settled decisions (owner-confirmed 2026-07-21)

1. **q = the native named MHR rotation parameters** — free-flyer root + 125
   named params (nq 132). Typed revolute sub-nodes, coupled channels through
   BR's existing single-source mimic; the overcomplete spine block is solved
   with same-axis stacked sub-nodes, making `from_classic`/`to_classic` an
   exact bijection.
2. **No BetterRobot changes.** (The pre-authorized multi-source-mimic edit
   proved unnecessary.) BR gaps get reported, never worked around.
3. **Proportions are shape, not pose**: the 71 bone-length params + 2 hand
   scales form a `proportions` group in `with_shape`, baked into placements
   and per-joint skinning scales — so inertias stay correct.
4. **All 7 LODs**, default `lod=1`. Spawn flags: `use_expression`,
   `use_correctives`, `compute_mass`.
5. **`pymomentum` only in `tools/`** (it is conda/pixi-only, not on PyPI):
   a pinned env powers the one-time converter and the parity oracle. The
   reference TorchScript `mhr_model.pt` is the pymomentum-free secondary
   oracle and the performance yardstick.
6. **Mass via the argmax segmentation** (MHR ships no authored closed-3D
   asset); parts grouped per the reference segmentation tool.

## Orders

| Order | File | What | Depends on |
|---|---|---|---|
| 01 | [`for_agents/01_convert_goldens.md`](for_agents/01_convert_goldens.md) | pinned pymomentum env, FBX+`.model` → per-LOD npz converter, golden grids | — |
| 02 | [`for_agents/02_core_stages.md`](for_agents/02_core_stages.md) | io schema + core stages (6D feature, MLP corrective, scale-aware skinning); SMPL untouched | — (parallel with 01) |
| 03 | [`for_agents/03_mhr_body.md`](for_agents/03_mhr_body.md) | the `MHR` body: BR tree, shape bake, forward, classic converters | 01, 02 |
| 04 | [`for_agents/04_parity.md`](for_agents/04_parity.md) | parity vs goldens (all LODs) + TorchScript cross-check, gradchecks, commit goldens | 03 |
| 05 | [`for_agents/05_perf_docs.md`](for_agents/05_perf_docs.md) | benchmark vs TorchScript, segmentation renders, example, docs | 04 |

Each order ends with the full BetterHuman test gate green and writes an honest
`for_agents/0N_results.md`. Subagents run on Opus; CUDA is available (RTX 6000
Ada).

## Targets

- **Parity**: measured against the float32-only pymomentum oracle, then pinned
  (~1e-5 m expected; the f64 ≤ 1e-6 standard is impossible against an f32
  oracle — contracts get an honest f32-oracle clause).
- **Classic round-trip**: exact both directions (float64 ≈ 1e-12).
- **Speed**: ≤ 2× the reference TorchScript at CUDA batch ≥ 16 (LOD1).
- SMPL/SMPL-X baselines in `docs/contracts.md` §Measured baselines must not
  move.

## Beyond this plan (roadmap, not scheduled)

MANO / SMPL-H / FLAME; STAR / SUPR (quaternion features, β-dependent
correctives); Anny (CC0, phenotype sliders); MHR follow-ups (authored closed
segmentation, landmarks, SMPL↔MHR conversion via the reference barycentric
maps, LOD remap tools); keypoint output heads; human residuals on the BR
optimizer; human→robot retargeting; differentiable measurements.
