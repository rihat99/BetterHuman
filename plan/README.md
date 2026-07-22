# Plan — MHR follow-up: simplify, fix LBS, benchmark the SMPL family

MHR support landed on `dev` (commits `1814394`, `8c32ec7`) and was verified
after the fact by two independent read-only reviews: **zero undeclared
deviations, zero false self-reported claims**, 228/228 tests green with CUDA,
parity reproduced independently (worst LOD1 float32 vertex error
`2.92e-6 m` vs the float32 pymomentum oracle). The historical order/results
files live in git: `git show 8c32ec7:plan/for_agents/`.

Three things remain, in order:

1. **Trim overengineering.** The code is sound, but an adversarial review
   found ~90–120 src lines + ~40 test lines of fat, concentrated in
   validation of impossible states, constructor plumbing, and one
   speculative API. Findings with file:line are in the order file.
2. **Fix the LBS formulation.** The batch-256 slowness is a measured
   algorithmic issue, not a logic bug: `lbs` transforms every vertex once
   *per influence* through quaternion `se3.act` (memory-bound). The
   algebraically identical matrix-blend formulation (per-joint 3×4 affines,
   scale folded in, one dense-weight GEMM) is **5.9× faster at CUDA
   batch 256 (30.5 → 5.2 ms)**, measured on our data. With it, the Warp
   pose-only batch-256 forward lands at ≈9 ms vs the reference TorchScript's
   12.7 ms — so **no Warp/CUDA kernels in BetterHuman**; the idea is
   settled as unnecessary.
3. **Benchmark SMPL and SMPL-X** against the original `smplx` package
   (vendored at `references/parametric_body/smplx`, imports cleanly in our
   venv) and document all performance numbers in docs.

## Orders

| Order | File | What | Depends on |
|---|---|---|---|
| 01 | [`for_agents/01_simplify.md`](for_agents/01_simplify.md) | remove the reviewed overengineering; pure refactor, parity byte-identical | — |
| 02 | [`for_agents/02_lbs_matrix.md`](for_agents/02_lbs_matrix.md) | matrix-blend LBS rewrite; re-measure parity; re-run the MHR benchmark | 01 |
| 03 | [`for_agents/03_benchmarks_docs.md`](for_agents/03_benchmarks_docs.md) | SMPL/SMPL-X benchmark vs original `smplx`; final docs sweep | 02 |

Each order ends with the full test gate green and writes an honest
`for_agents/0N_results.md`. Subagents run on Opus; CUDA is available
(RTX 6000 Ada; GPU 7 was free at planning time).

## Targets

- Order 01 is a pure refactor: every measured parity maximum byte-identical
  before/after.
- Order 02: CUDA batch-256 `transforms + LBS` ≤ ~6 ms; the Warp pose-only
  path meets **≤ 2× TorchScript at CUDA batch ≥ 16** (expected ≈ 1×). All
  parity re-measured; if any measured maximum exceeds its pinned tolerance,
  stop and report — never bump a pin silently.
- Order 03: honest SMPL/SMPL-X vs original-`smplx` tables in
  `docs/architecture.md`; measure and document — no pass/fail target, but
  a >2× gap at CUDA batch ≥ 16 gets reported as a finding.

## Standing decisions (owner-confirmed)

- No BetterRobot changes; BR gaps are reported, never worked around
  (known: per-joint Torch FK cost, Warp lane rejecting batched joint
  placements with unbatched frames, CRBA rank limited by the overcomplete
  spine — all documented).
- MHR CRBA is PSD, not SPD, by construction; documented, not "fixed".
- Alpha: breaking changes allowed, no back-compat shims. Everything on `dev`.

## Beyond this plan (roadmap, not scheduled)

MANO / SMPL-H / FLAME; STAR / SUPR; Anny; MHR follow-ups (authored closed
segmentation, landmarks, SMPL↔MHR conversion, LOD remap tools); keypoint
heads; human residuals on the BR optimizer; human→robot retargeting;
differentiable measurements.
