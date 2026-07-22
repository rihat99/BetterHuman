# Order 02 — matrix-blend LBS rewrite

> **Implementation log (2026-07-21):** Complete. All bodies now use one
> dense matrix-blend LBS formulation; existing parity pins remain unchanged.
> CUDA: 226 passed, 0 skipped. Warp pose-only is 0.899×/1.056× reference at
> batch 16 and 0.744×/0.736× at batch 256. See `02_results.md`.

The owner's batch-256 slowness is a measured algorithmic formulation issue
in `core/pipeline.py::lbs`, not a logic bug and not a missing-Warp problem.
This order replaces the formulation, re-measures parity honestly, and
re-runs the MHR benchmark. Depends on order 01.

## Ground rules (binding)

- Read `~/.claude/handoffs/betterhuman_mhr_context.md` (§"LBS perf root
  cause" carries the measured evidence), `docs/contracts.md`, and
  `docs/architecture.md` §MHR performance first.
- **No BetterRobot modifications.** Never modify `models/MHR/assets/` or
  `references/`.
- **No Warp/CUDA kernels in BetterHuman.** Measured unnecessary: with this
  rewrite the Warp-FK pose-only batch-256 forward is ≈ 9 ms vs the
  reference TorchScript's 12.7 ms. If some configuration still misses its
  target afterwards, report the split — do not add kernels or caching.
- Honest numbers; medians as measured.
- Write `plan/for_agents/02_results.md`: parity before/after tables, full
  benchmark tables, exact test counts.

## 1. The rewrite

Current `lbs` (`core/pipeline.py`) gathers `(B, V, K, 7)` transforms and
runs quaternion `se3.act` (plus a second `so3.act` for the scale
correction) per vertex per influence — memory-bound. Per-influence output:

```
p' = R_s·p + t_s + (s − 1)·R_s·(p − t_bind)
   = (s·R_s)·p + (t_s − (s − 1)·R_s·t_bind)
```

i.e. each joint contributes a plain affine `M_j = s_j·R_j`,
`b_j = t_j − (s_j − 1)·R_j·t_bind_j` (with `M_j = R_j`, `b_j = t_j` when
unscaled). So:

1. Convert the `(…, J, 7)` skinning transforms to per-joint 3×4 affines
   once (J = 127 for MHR — negligible), folding `joint_scales` and
   `bind_joint_positions` in as above.
2. Blend affines with a **dense** `(V, J)` weight matrix via one GEMM
   (`einsum("vj,...jx->...vx")`), then apply the blended affine once per
   vertex.

Measured on the real LOD1 data (V=18439, K=8, J=127, RTX 6000 Ada, f32,
median of 30, agreement between formulations ≤ 1.7e-6 on aggressive random
poses — algebraically identical, rounding only):

| Config | current | matrix-dense |
|---|---:|---:|
| B=16, scaled | 1.200 ms | 0.790 ms |
| B=256, scaled | 30.545 ms | **5.151 ms (5.9×)** |
| B=256, unscaled | 15.183 ms | 5.012 ms |

Implementation constraints:

- **One formulation for all models** (SMPL, SMPL-X, MHR). Delete the
  per-influence gather path entirely — no dual code paths, no flags.
- Archive/schema stay unchanged (padded sparse `(V, K)` storage). Build the
  dense `(V, J)` weight matrix once at spawn via `scatter_add_` (sizes are
  trivial: 37 MB at LOD0, far less elsewhere). Where to store it
  (`BodyValues` vs structure) is your call — smallest change wins; the
  spawn-time argmax segmentation keeps consuming the sparse form.
- Callers to update: `core/body.py::vertices_from_data`,
  `bodies/mhr.py::_bake_mhr_shape` (neutral-vertex LBS), and the
  stage-instrumentation `_transform_and_lbs` in
  `benchmarks/benchmark_mhr.py` (it deliberately mirrors the production
  stage — keep it mirroring the new one).
- The exact-reduction test
  (`test_generalized_skinning_reduces_exactly_on_smpl_fixture`) compared
  the generalized path against the legacy per-influence path; after the
  rewrite the equivalent invariant is: scaled path at `s = 1` is
  tensor-equal to the unscaled path on the SMPL fixture. Rewrite the test
  to pin that (this is a permitted baseline-formulation swap, not a
  weakening — state it in the results file).

## 2. Parity policy — measure, never silently re-pin

The formulation is algebraically identical; only float rounding moves.

- **float64: must stay within every existing pin** (headroom is ~1e-15 vs
  pins of 2e-6/5e-7 — if f64 moves materially, you have a bug, stop).
- **float32: re-measure every parity suite** (SMPL, SMPL-X, MHR all LODs,
  TorchScript cross-check) and put before/after maxima in the results file.
  Known risk: SMPL f32 vertex parity is pinned at 1e-6 with measured worst
  7.08e-7 — rounding shift may cross the pin. **If any measured maximum
  exceeds its pinned tolerance: STOP, report the numbers, and wait for the
  owner's decision.** Do not bump any pin yourself.
- Existing gradcheck/batching/dtype/device tests must pass through the new
  path unmodified.

## 3. Re-benchmark and update docs

Re-run `benchmarks/benchmark_mhr.py` with the protocol already in the file
(LOD1, f32, batch 1/16/256, correctives on/off, CPU and CUDA and
`--use-warp`, synchronized medians of ≥30, GPU 7 if free). Expected:
batch-256 `transforms + LBS` drops ~30 → ~5-6 ms; Warp pose-only meets
≤ 2× TorchScript at batch ≥ 16 (likely ≈ 1× or better); CPU batch-256
improves substantially. Update the tables in `docs/architecture.md`
§MHR performance and the perf row in `docs/contracts.md` with the new
measured numbers (replace, don't append). Known and accepted: the fair
shape+pose CUDA path stays Torch-FK-bound (BR Warp lane rejects batched
joint placements with unbatched frames — already documented as a BR
finding; do not work around).

## Acceptance

- Full suite green; parity within pins (or stopped-and-reported per §2).
- Benchmark tables regenerated with real medians; the ≤2× CUDA batch ≥ 16
  target met on the Warp pose-only path, or the miss explained by split.
- Single LBS formulation in the codebase; `ruff check .` clean.
