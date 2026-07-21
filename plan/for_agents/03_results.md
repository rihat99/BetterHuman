# Order 03 results — MHR body

Completed on branch `dev` on 2026-07-21. BetterRobot, `models/MHR/assets/`,
and `references/` were not modified.

## Delivered

- `MHR_CONFIG` with SI/axis/Euler conventions, seven LODs, default LOD1,
  canonical archive pattern, and the zero-weight segmentation group.
- `MHR` / `MHRClassic`, registry integration, and the public `better_human.MHR`
  export.
- A model-driven BetterRobot tree with a free-flyer native `root`, 125 public
  named revolutes, and 47 concrete mimic targets. The public dimensions are
  `nq=132`, `nv=131`; full dimensions are `179/178`.
- Shape baking for 45 identity, optional 72 expression, and 73 proportion
  parameters. The proportion path covers 71 joint translations and both
  `exp2` hand scales, including accumulated descendant scale.
- Authoritative static inverse-bind skinning, all 127 native joint poses in
  native order, BR-expanded mimic channels for corrective Euler features, and
  the baked-dense 750→3000→`V*3` corrective MLP.
- Data-driven 19-part segmentation and inertias scattered only to physical
  terminal bodies; all intermediate bodies remain massless.
- Shape-aware classic conversion: `from_classic` returns `(shaped_body, q)`;
  `shaped_body.to_classic(q)` is its inverse. Public documentation now records
  this model-specific boundary.
- Focused coverage for all LODs, tree and parameter maps, scaled host limits,
  mimic expansion, float32/float64 and negative-quaternion round trips,
  expression modes, batching, gradients, LOD1 parity, and mass.

## Measured results

### Tree and conversion

| Quantity | Measured |
|---|---:|
| BetterRobot joints / bodies / frames | `203 / 203 / 203` |
| Public / full rotation nodes | `125 / 172` |
| Mimic targets | `47` |
| Deterministic classic round trip, float64 | `5.551115123125783e-17` max |
| Deterministic classic round trip, float32 | `2.9802322387695312e-08` max |
| Review random-q round trip, float64 | `5.97e-15` max over 10,000 roots |

All 125 pose coordinates and their sign-aware public limits were checked
against the archive. The six non-unit spine host weights use the documented
scaled-contribution q semantics.

### Early LOD1 parity (`combined_00`, float32)

| Output | Max absolute error |
|---|---:|
| Vertices | `2.2649765014648438e-06 m` |
| Native joint positions | `6.556510925292969e-07 m` |
| Native quaternions, sign invariant | `3.5762786865234375e-07` |
| Accumulated joint scale | `0` |

The Order 03 acceptance threshold was `1e-3 m`; Order 04 owns the complete
48-golden tight parity grid.

### Mass and LOD0 memory

At density 1000 kg/m³, LOD1 has 19 positive physical bodies totaling
`81.481849670 kg`. Neutral totals across LOD0→6 are:

```text
81.093444824, 81.481849670, 72.890838623, 76.809577942,
70.775901794, 59.534843445, 57.649929047 kg
```

An isolated LOD0 expression-on, correctives-off, mass-on spawn plus forward
peaked at `661004 KiB` RSS. The LOD0 archive is `2769297616` bytes; its
corrective output tensor alone is `2651004000` bytes (`2.469 GiB`) in
float32 and twice that in float64.

## Verification

- MHR body tests: **14 passed**.
- MHR plus core-mass focused gate: **29 passed**, 1 expected float32-gradcheck
  warning.
- Existing SMPL/SMPL-X parity gate: **84 passed**.
- Full suite: **199 passed, 4 skipped, 15 warnings** in `59.81 s`. The skips
  are unavailable-CUDA cases; warnings are the existing float32 gradcheck and
  SciPy deprecation warnings.
- `.venv/bin/ruff check .` and `git diff --check` pass.
- The Order 03 Python files pass `ruff format --check`. A repository-wide
  formatter check still identifies six pre-existing files outside this
  order's formatting scope; they were not bulk-reformatted.

## Deviations and findings

1. **Static FBX binds remain authoritative.** The plan proposed recomputing
   binds during shape bake, but zero-parameter FK differs from the stored bind
   by `4.57763671875e-05 cm`, and recomputation cancels intended proportion
   deformation. The implementation composes world poses with the stored
   inverse binds; LOD6 fully-random parity is `1.55e-6 m`.
2. **Native joint 0 is not the free-flyer.** Archive evidence identifies
   `body_world` (joint 0) as fixed universe and `root` (joint 1) as the owner
   of the six root channels. The tree follows the data rather than the
   conflicting sentence in the plan.
3. **Numeric density only.** MHR accepts a scalar or 19-vector density. The
   existing named table has 24 SMPL-indexed rows and no validated mapping to
   MHR's 19 groups, so `MHR_CONFIG.density_table` is `None` instead of silently
   assigning incorrect material densities.
4. **Undriven leaves are fixed joints, not frames.** This keeps one uniform
   native-joint gather and yields 203 total FK joints. The plan permits fixed
   joints, but converting 20 terminal leaves to frames could reduce the tree
   to 183 joints if Order 05 measurements show it is necessary.
5. **Expression absence is canonicalized.** With expression enabled,
   `expression=None` and an explicit zero tensor are reference-equivalent;
   `to_classic` emits the explicit zero representation. With expression
   disabled, non-`None` input is rejected and output remains `None`.
6. **Coarse-LOD centroid caps are approximate.** Every LOD has 19 positive
   parts, but total mass falls substantially at LOD5/6 because the fallback
   part meshes are built from very coarse triangulations. LOD1 is the default
   and is the pinned acceptance value.
7. **Dtype support is explicit.** MHR rejects float16/bfloat16 at construction
   because BetterRobot FK supports float32/float64 only. CUDA was unavailable,
   so the CUDA path remains unverified rather than inferred from CPU results.

## Primary commands

```bash
.venv/bin/pytest tests/bodies/test_mhr.py tests/core/test_mass.py -q
.venv/bin/pytest tests/parity/test_smpl_parity.py \
  tests/parity/test_smplx_parity.py -q
.venv/bin/pytest tests/ -q
.venv/bin/ruff check .
git diff --check
```
