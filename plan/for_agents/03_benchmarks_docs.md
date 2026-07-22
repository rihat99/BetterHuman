# Order 03 — SMPL/SMPL-X benchmarks vs original `smplx`, final docs

> **Implementation log (2026-07-21):** Complete. Canonical per-batch upstream
> comparisons agree within `5.9604644775390625e-7 m`; Warp pose-only is
> 0.800–1.463× upstream at CUDA batch 16/256, while fair Torch-FK shape+pose
> is 4.605–13.559×. CUDA: 226 passed, 0 skipped. See `03_results.md`.

The owner's ask: "measure speed for smpl and smplx, us vs original code,
and document in docs." Plus a final docs sweep. Depends on order 02 (so the
numbers are final).

## Ground rules (binding)

- **No BetterRobot modifications.** Never modify `models/` weights or
  `references/` (importing reference code read-only is fine).
- Honest numbers: medians as measured, report device + torch version +
  shared-host caveats.
- Write `plan/for_agents/03_results.md` with the full tables.

## 1. Benchmark SMPL and SMPL-X vs the original package

Yardstick: the vendored original `smplx` package at
`references/parametric_body/smplx` — verified to import cleanly in our venv
via `sys.path.insert(0, "references/parametric_body/smplx")` (pure torch,
no install needed). Use `smplx.create`/`SMPL`/`SMPLX` with the weights
under `models/smpl` and `models/smplx` (pick the same neutral models our
parity tests use; disable its extras that we don't model — no DMPL, and
match hand/face flags to what we spawn).

Protocol — mirror `benchmarks/benchmark_mhr.py` in a new
`benchmarks/benchmark_smpl.py` (shared helpers may be factored out only if
it stays obviously simple):

- SMPL and SMPL-X; CPU and CUDA; batch 1 / 16 / 256; float32.
- Reference row: original `smplx` full forward (`betas` + `body_pose` +
  `global_orient`, which recomputes shape blendshapes + joint regression +
  pose correctives + LBS every call).
- Our rows, with the fairness note stated in the table: **pose-only**
  (`vertices(q)` after a prior `with_shape`) and **shape+pose**
  (`with_shape` + `vertices(q)` per call — the apples-to-apples row).
- Feed equivalent inputs (same axis-angle pose converted through
  `from_classic`; same betas) and sanity-check outputs agree to parity
  tolerance before timing — a benchmark of mismatched computations is
  worthless.
- Warm-ups, then median over ≥ 30 iterations; `torch.cuda.synchronize()`
  around timings; try BR's Warp FK lane for the pose-only rows where
  applicable; report the FK / corrective / LBS split for ours.

No pass/fail target: measure and document. If we are > 2× the original at
CUDA batch ≥ 16, report it as a finding with the stage split (a BR-FK-bound
result is a known finding, not something to work around).

## 2. Final docs sweep

- `docs/architecture.md`: add the SMPL/SMPL-X benchmark table next to the
  MHR one (same format, same honesty about fairness rows); one paragraph on
  the matrix-blend LBS formulation replacing the per-influence one.
- `docs/contracts.md`: add the SMPL/SMPL-X perf baselines; fix the wording
  gap found in review — line ~140 claims the MHR TorchScript baseline
  "skeleton exact", but the enforced test pins the skeleton at the f32
  tolerance class (`1e-6 m` position, `1e-6` quaternion, `1e-7` scale);
  state the pinned values and note the *measured* agreement was bit-exact.
- Check the MHR architecture section still records the durable facts from
  the deleted results files (LOD0 archive ≈ 2.6 GB with the corrective
  tensor ≈ 2.5 GiB, neutral LOD1 mass 81.48 kg at density 1000, CRBA
  PSD-not-SPD rationale). Add whatever is missing; the full historical
  record stays reachable via `git show 8c32ec7:plan/for_agents/`.
- `README.md`: the line pointing at `plan/for_agents/` for "measured
  results" (≈ line 146) is stale after the plan rewrite — point it at the
  docs (architecture/contracts) instead.
- Every doc claim these orders falsified gets fixed here at the latest.

## Acceptance

- Both benchmark scripts run reproducibly with documented commands; tables
  in results + docs carry real medians.
- Output-agreement sanity check passed before timing (state the measured
  max difference).
- Docs true to code (spot-check every changed claim); full suite green;
  `ruff check .` clean.
